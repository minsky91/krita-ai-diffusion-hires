from __future__ import annotations
import asyncio
import json
import os
import struct
import uuid
from dataclasses import dataclass
from enum import Enum
from collections import deque
from itertools import chain, product
from typing import Any, Optional, Sequence

from .api import WorkflowInput
from .client import Client, CheckpointInfo, ClientMessage, ClientEvent, DeviceInfo, ClientModels
from .client import SharedWorkflow, TranslationPackage, ClientFeatures, TextOutput
from .client import MissingResources, filter_supported_styles, loras_to_upload
from .files import FileFormat
from .image import Image, ImageCollection, ImageFileFormat
from .network import RequestManager, NetworkError
from .websockets.src import websockets
from .style import Styles
from .resources import ControlMode, ResourceId, ResourceKind, Arch
from .resources import CustomNode, UpscalerName, resource_id
from .settings import PerformanceSettings, settings
from .localization import translate as _
from .util import client_logger as log
from .workflow import create as create_workflow
from . import resources, util

# minsky91
from datetime import datetime
from .comfy_workflow import image_uid
from base64 import a85encode, b85encode
#from base64 import z85encode  # supported from v3.13 on
from PyQt5.QtCore import QByteArray
import PyQt5.QtCore
import json
from .settings import Verbosity, ServerMode

# global timestamp for logging of timing operations
import sys
last_timestamp = datetime.now()

class UploadImageFileFormat(Enum):
    png_95 = ("png", 95)  # max fast, largest files (near uncompr)
    png_85 = ("png", 85)  # fast, large files
    png_50 = ("png", 50)  # slow, smaller files
    webp_80 = ("webp", 80)
    webp_lossless = ("webp", 100)
    jpeg_85 = ("jpeg", 85)
    jpeg_95 = ("jpeg", 95)

# end of minsky91 additions

if util.is_macos:
    import os

    if "SSL_CERT_FILE" not in os.environ:
        os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"

# minsky91: a few utilities for verbose logging 

def get_time_diff(start_time: datetime = None):
    if datetime is None:
        return 0, ""
    time_now = datetime.now()
    time_diff: float = (time_now - start_time).microseconds / 10.0**6
    time_diff = time_diff + (time_now - start_time).seconds
    return time_diff,  f"time {time_diff:.2f} sec."

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

@dataclass
class JobStats:
    metadata: list[str]
    workflow: dict[str, Any]
    submit_time: datetime = None
    exec_time: datetime = None
    prep_elapsed_time: float = 0.0 
    exec_elapsed_time: float = 0.0
    batch_active_time: float = 0.0
    batch_size: int = 0
    last_in_the_batch: bool = False
    
    def set_submit_time(self):
        self.submit_time = datetime.now()
        return self.submit_time
    
    def set_exec_time(self):
        self.exec_time = datetime.now()
        return self.exec_time
    
    def set_prep_elapsed_time(self):
        if self.submit_time:
            self.prep_elapsed_time, _ = get_time_diff(self.submit_time)
        else:
            self.prep_elapsed_time = 0.0
        self.metadata.append(f"    Preparation time: {self.prep_elapsed_time:.2f} sec.")
        return self.prep_elapsed_time
    
    # lines in stat_metadata are indented to indicate their verbosity level
    
    def set_total_active_time(self):
        total_active_time = 0.0
        if self.prep_elapsed_time:
            total_active_time += self.prep_elapsed_time
        if self.exec_elapsed_time:
            total_active_time += self.exec_elapsed_time
        self.metadata.append(f"    Total active time: {total_active_time:.1f} sec.")
        return total_active_time
            
    def set_exec_elapsed_time(self, is_end: bool = True):
        if self.exec_time:
            self.exec_elapsed_time, _ = get_time_diff(self.exec_time)
        else:
            self.exec_elapsed_time = 0.0
        self.metadata.append(f"Execution time: {self.exec_elapsed_time:.1f} sec.")
        if is_end:
            total_active_time = self.set_total_active_time()
            lifetime, _ = get_time_diff(self.submit_time)
            self.metadata.append(f"    Total lifetime: {lifetime:.1f} sec.")
            return total_active_time
        return self.exec_elapsed_time

    def add_batch_active_time(self, add_time: float = 0.0):
        self.batch_active_time += add_time
        return self.batch_active_time
        
    def get_batch_active_time(self):
        return self.batch_active_time

    def set_job_batch_size(self, n_items:int=0):
        if n_items > self.batch_size:
           self.batch_size = n_items
        return self.batch_size
        
    def set_last_in_the_batch(self):
        self.last_in_the_batch = True
        
    def set_upload_params(self, upload_time, n_images, files_mb_size, dim_x, dim_y, cached_images):
        res_str = f" of {dim_x}x{dim_y} pixels"
        if files_mb_size > 0.0:
            self.metadata.append(f"  Input files total size: {files_mb_size:.2f} MB in {n_images} image(s){res_str} ({cached_images} cached)")
        else:
            if n_images == cached_images:
                self.metadata.append(f"  {n_images} cached input image(s){res_str}")
            else:
                self.metadata.append(f"  {n_images} input image(s){res_str} ({cached_images} cached)")
        if upload_time > 0.0:
            self.metadata.append(f"    Input files upload time: {upload_time:.2f} sec.")
    
    def set_wf_params(self, post_time, wf_size, wf_nodes):
        self.metadata.append(f"  Workflow size: {wf_size:.2f} MB, {wf_nodes} nodes")
        self.metadata.append(f"    Workflow upload time: {post_time:.2f} sec.")
    
    def set_download_params(self, download_time, n_images, img_mb_size, img_format, dim_x, dim_y):
        self.metadata.append(f"  Output files total size: {img_mb_size:.2f} MB in {n_images} {img_format} image(s) of {dim_x}x{dim_y} pixels")
        self.metadata.append(f"    Output files download time: {download_time:.2f} sec.")

    def save_workflow(self, wf:dict):
        self.workflow = wf
    
    def metadata_from_stats(self, system_info):
        def print_gb(k, v):
            if "free" in k:
                ram_gb = int(v) / 1024**3
                m_str = f"{ram_gb:.1f}"
            else:
                ram_gb = round(int(v) / 1024**3)
                m_str = f"{ram_gb}"
            return f"{k}: {m_str} GB"
        self.metadata.append(f"    Batch size: {max(self.batch_size, 1)}")
        if self.last_in_the_batch:
            self.metadata.append(f"    Batch active time: {self.get_batch_active_time():.1f} sec.")
        if system_info:
            self.metadata.append("    ")
            self.metadata.append("    System info:")
            if "system" in system_info:
                for k,v in system_info["system"].items():
                    if k not in ["embedded_python", "argv"]:
                        if k == "python_version":
                            self.metadata.append(f"    {k}: {v[:8]}")
                        elif "ram_" in k:
                            self.metadata.append("    "+print_gb(k, v))
                        else:
                            self.metadata.append(f"    {k}: {v}")
            if "devices" in system_info:
                self.metadata.append("    GPU device features:")
                for device_data in system_info["devices"]:
                    for k,v in device_data.items():
                        if k in ["name", "vram_total", "vram_free"]:
                            if k == "name":
                                v_str, _ = v.split(' : ', 1)
                                self.metadata.append(f"    {k}: {v_str}")
                            elif "ram_" in k:
                                self.metadata.append("    "+print_gb(k, v))
                            else:
                                self.metadata.append(f"    {k}: {v}")
        return self.metadata
    
    def retrieve_workflow(self):
        return self.workflow
    
# end of minsky91 additions


@dataclass
class JobInfo:
    local_id: str
    work: WorkflowInput
    # minsky91: include job stats data
    stats: JobStats
    # end of minsky91 additions
    front: bool = False
    remote_id: str | asyncio.Future[str] | None = None
    node_count: int = 0
    sample_count: int = 0

    def __str__(self):
        return f"Job[local={self.local_id}, remote={self.remote_id}]"

    @staticmethod
    def create(work: WorkflowInput, front: bool = False):
        return JobInfo(str(uuid.uuid4()), work, front)

    async def get_remote_id(self):
        if isinstance(self.remote_id, asyncio.Future):
            self.remote_id = await self.remote_id
        return self.remote_id



class Progress:
    _nodes = 0
    _samples = 0
    _info: JobInfo

    def __init__(self, job_info: JobInfo):
        self._info = job_info

    def handle(self, msg: dict):
        id = msg["data"].get("prompt_id", None)
        if id is not None and id != self._info.remote_id:
            return
        if msg["type"] == "executing":
            self._nodes += 1
        elif msg["type"] == "execution_cached":
            self._nodes += len(msg["data"]["nodes"])
        elif msg["type"] == "progress":
            self._samples += 1

    @property
    def value(self):
        # Add +1 to node count so progress doesn't go to 100% until images are received.
        node_part = self._nodes / (self._info.node_count + 1)
        sample_part = self._samples / max(self._info.sample_count, 1)
        return 0.2 * node_part + 0.8 * sample_part


class ComfyClient(Client):
    """HTTP/WebSocket client which sends requests to and listens to messages from a ComfyUI server."""

    default_url = "http://127.0.0.1:8188"

    def __init__(self, url):
        self.url = url
        self.models = ClientModels()
        self._requests = RequestManager()
        self._id = str(uuid.uuid4())
        self._active: Optional[JobInfo] = None
        self._features: ClientFeatures = ClientFeatures()
        self._supported_archs: dict[Arch, list[ResourceId]] = {}
        self._messages = asyncio.Queue()
        self._queue = asyncio.Queue()
        self._jobs: deque[JobInfo] = deque()
        self._is_connected = False
        
        # minsky91: additional objects for timing and unique image tracing
        self._input_image_uids: list[int] = []  # uids of uploaded input images cached on the server side
        # minsky91: added storing of full system info returned by the server
        self._system_info: dict[str, Any] | None = None 
        # end of minsky91 additions



    @staticmethod
    async def connect(url=default_url, access_token=""):
        client = ComfyClient(parse_url(url))
        # minsky91: modified
        log.info(f"Connecting to ComfyUI server at {client.url}")

        # Retrieve system info
        # minsky91: added extended system and job stats & metadata 
        client._system_info = await client._get("system_stats")
        client.device_info = DeviceInfo.parse(client._system_info)

        # Try to establish websockets connection
        wsurl = websocket_url(client.url)
        try:
            async with websockets.connect(f"{wsurl}/ws?clientId={client._id}"):
                pass
        except Exception as e:
            msg = _("Could not establish websocket connection at") + f" {wsurl}: {str(e)}"
            raise Exception(msg)

        # Check custom nodes
        nodes = await client._get("object_info")
        missing = _check_for_missing_nodes(nodes)
        if len(missing) > 0:
            raise MissingResources(missing)

        client._features = ClientFeatures(
            ip_adapter=True,
            translation=True,
            languages=await _list_languages(client),
            wave_speed="ApplyFBCacheOnModel" in nodes,
        )

        # Check for required and optional model resources
        models = client.models
        models.node_inputs = {name: nodes[name]["input"] for name in nodes}
        available_resources = client.models.resources = {}

        clip_models = nodes["DualCLIPLoader"]["input"]["required"]["clip_name1"][0]
        available_resources.update(_find_text_encoder_models(clip_models))
        if clip_gguf := nodes.get("DualCLIPLoaderGGUF", None):
            clip_gguf_models = clip_gguf["input"]["required"]["clip_name1"][0]
            available_resources.update(_find_text_encoder_models(clip_gguf_models))

        vae_models = nodes["VAELoader"]["input"]["required"]["vae_name"][0]
        available_resources.update(_find_vae_models(vae_models))

        control_models = nodes["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        available_resources.update(_find_control_models(control_models))

        clip_vision_models = nodes["CLIPVisionLoader"]["input"]["required"]["clip_name"][0]
        available_resources.update(_find_clip_vision_model(clip_vision_models))

        ip_adapter_models = nodes["IPAdapterModelLoader"]["input"]["required"]["ipadapter_file"][0]
        available_resources.update(_find_ip_adapters(ip_adapter_models))

        style_models = nodes["StyleModelLoader"]["input"]["required"]["style_model_name"][0]
        available_resources.update(_find_style_models(style_models))

        models.upscalers = nodes["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        available_resources.update(_find_upscalers(models.upscalers))

        inpaint_models = nodes["INPAINT_LoadInpaintModel"]["input"]["required"]["model_name"][0]
        available_resources.update(_find_inpaint_models(inpaint_models))

        loras = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]
        available_resources.update(_find_loras(loras))

        # Retrieve list of checkpoints
        checkpoints = await client.try_inspect("checkpoints")
        diffusion_models = await client.try_inspect("diffusion_models")
        diffusion_models.update(await client.try_inspect("unet_gguf"))
        client._refresh_models(nodes, checkpoints, diffusion_models)

        # Check supported base models and make sure there is at least one
        client._supported_archs = {ver: client._check_workload(ver) for ver in Arch.list()}
        supported_workloads = [
            arch for arch, miss in client._supported_archs.items() if len(miss) == 0
        ]
        log.info("Supported workloads: " + ", ".join(arch.value for arch in supported_workloads))
        if len(supported_workloads) == 0:
            raise MissingResources(client._supported_archs)

        # Workarounds for DirectML
        if client.device_info.type == "privateuseone":
            # OmniSR causes a crash
            for n in [2, 3, 4]:
                id = resource_id(ResourceKind.upscaler, Arch.all, UpscalerName.fast_x(n))
                available_resources[id] = models.default_upscaler

        _ensure_supported_style(client)
        
        return client

    async def _get(self, op: str, timeout: float | None = 30):
        return await self._requests.get(f"{self.url}/{op}", timeout=timeout)

    async def _post(self, op: str, data: dict):
        return await self._requests.post(f"{self.url}/{op}", data)

    async def enqueue(self, work: WorkflowInput, front: bool = False):
        job = JobInfo.create(work, front=front)
        await self._queue.put(job)
        return job.local_id

    async def _report(self, event: ClientEvent, job_id: str, value: float = 0, **kwargs):
        await self._messages.put(ClientMessage(event, job_id, value, **kwargs))

    async def _run(self):
        assert self._is_connected
        try:
            while self._is_connected:
                job = await self._queue.get()
                try:
                    await self._run_job(job)
                except Exception as e:
                    log.exception(f"Unhandled exception while processing {job}")
                    await self._report(ClientEvent.error, job.local_id, error=str(e))
        except asyncio.CancelledError:
            pass


    async def _run_job(self, job: JobInfo):
        await self.upload_loras(job.work, job.local_id)
        
        # minsky91: job timing stats
        job.stats = JobStats([], {})
        time_job_start = job.stats.set_submit_time()
        # end of minsky91 additions

        workflow = create_workflow(job.work, self.models)
        job.node_count = workflow.node_count
        job.sample_count = workflow.sample_count
        if settings.debug_dump_workflow:
            workflow.dump(util.log_dir)

        # minsky91: save workflow for metadata and upload input images
        if settings.upload_method != "None":        
            job.stats.save_workflow(workflow.root)  # won't save a wf with BASE64-encoded images
            result = await self._upload_input_images(job, workflow.get_image_uids())
        # end of minsky91 additions

        data = {"prompt": workflow.root, "client_id": self._id, "front": job.front}
        job.remote_id = asyncio.get_running_loop().create_future()
        self._jobs.append(job)

        seed_str = "" if job.work is None or job.work.sampling is None else f" seed {job.work.sampling.seed}"
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_run_job: posting prompt workflow, node_count {job.node_count}, client_id {self._id}{seed_str}")
        elif Verbosity(settings.logfile_verbosity) in [Verbosity.medium]: 
            log.info(f"posting prompt workflow with {job.node_count} nodes")
        upl_time = datetime.now()
        # end of minsky91 additions
        
        try:
            result = await self._post("prompt", data)
            job.remote_id.set_result(result["prompt_id"])
        except Exception as e:
            job.remote_id.set_result("ERROR")
            if self._jobs[0] == job:
                self._jobs.popleft()
            raise e
            
        # minsky91: job stats & timing data 
        global get_size
        wf_mb_size = get_size(data, None) / 1024.0**2 
        time_taken, time_diff_str = get_time_diff(upl_time) 
        job_prep_elapsed_time = job.stats.set_prep_elapsed_time()
        job.stats.set_wf_params(time_taken, wf_mb_size, len(workflow.root))
        job_prep_elapsed_time_str = f"time {job_prep_elapsed_time:.2f} sec."
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_run_job: workflow submitted, size {wf_mb_size:.2f} MB, upload {time_diff_str}, total prep. {job_prep_elapsed_time_str}, client_id {self._id}")
        elif Verbosity(settings.logfile_verbosity) in [Verbosity.medium]: 
            log.info(f"workflow submitted, size {wf_mb_size:.2f} MB, upload {time_diff_str}")
        else:
            log.info(f"submitted workflow")
        # end of minsky91 additions

    async def _listen(self):
        url = websocket_url(self.url)
        # minsky91: setting-based extended log info
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_listen: websockets.connect with {url}/ws?clientId={self._id} max_size {2**30} ping_timeout {180}")
        async for websocket in websockets.connect(
            # minsky91: extended timeout to 180, to accomodate for extra large files 
            f"{url}/ws?clientId={self._id}", max_size=2**30, ping_timeout=180, 
        ):
            try:
                await self._subscribe_workflows()
                # minsky91: reset transient http storage
                if settings.server_mode is ServerMode.external:
                    await self._reset_transient_storage()        
                await self._listen_websocket(websocket)
            except websockets.exceptions.ConnectionClosedError as e:
                log.warning(f"Websocket connection closed: {str(e)}")
            except OSError as e:
                msg = _("Could not connect to websocket server at") + f"{url}: {str(e)}"
                await self._report(ClientEvent.error, "", error=msg)
            except asyncio.CancelledError:
                await websocket.close()
                self._active = None
                self._jobs.clear()
                break
            except Exception as e:
                log.exception("Unhandled exception in websocket listener")
                await self._report(ClientEvent.error, "", error=str(e))
            finally:
                await self._report(ClientEvent.disconnected, "")

    async def _listen_websocket(self, websocket: websockets.ClientConnection):
        progress: Progress | None = None
        images = ImageCollection()
        last_images = ImageCollection()
        result = None

        # minsky91
        global last_timestamp
        last_workflow_name = ""

        async for msg in websocket:
            
            if isinstance(msg, bytes):
                # minsky91: added progress preview
                image, is_preview = _extract_message_png_image(memoryview(msg), settings.progress_preview)
                if image is not None:
                    if is_preview:
                        job = self._active
                        if job is not None and (progress is None or progress.value > 0.5):
                            await self._report(
                                ClientEvent.progress_preview, job.local_id, preview=image
                            )
                    else:
                        images.append(image)
                # end of minsky91 modifications

            elif isinstance(msg, str):

                # minsky91
                last_timestamp = datetime.now()

                msg = json.loads(msg)

                if msg["type"] == "status":
                    await self._report(ClientEvent.connected, "")

                if msg["type"] == "execution_start":
                    id = msg["data"]["prompt_id"]
                    self._active = await self._start_job(id)
                    if self._active is not None:
                        progress = Progress(self._active)
                        images = ImageCollection()
                        result = None
                        # minsky91: job logging, timing & stats
                        self._active.stats.set_exec_time()
                        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                            log.info(f"_listen_websocket: websocket event execution_start: starting a job")
                        # end of minsky91 additions

                if msg["type"] == "execution_interrupted":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        # minsky91: job logging, timing & stats
                        exec_elapsed_time = job.stats.set_exec_elapsed_time(is_end=True)
                        exec_time_str = f", active time {exec_elapsed_time:.2f} sec." if exec_elapsed_time != 0.0 else ""
                        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                            log.info(f"_listen_websocket: websocket executing msg: job {job.local_id} has been interrupted{exec_time_str}")
                        # end of minsky91 additions
                        self._clear_job(job.remote_id)
                        await self._report(ClientEvent.interrupted, job.local_id)
                        
                # minsky91: process a SendImageHttp or SaveTempImage node's message and receive the generated image
                if msg["type"] in ("sending_images", "images_ready"):
                    n_images = msg["data"]["n_images"]
                    if n_images:
                        if msg["type"] == "sending_images":
                            image_format = msg["data"]["format"]
                            uid = msg["data"]["uid"]
                            append_str = f"uid {uid}, image_format {image_format}"
                        elif msg["type"] == "images_ready":
                            filepath_prefix = msg["data"]["prefix"]
                            append_str = f"prefix {filepath_prefix}"
                        job = self._active
                        if job is not None:
                            extent = max(job.work.images.extent.desired, job.work.images.extent.target)
                        else:
                            extent = None
                        msg_str = msg["type"]
                        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                            log.info(f"_listen_websocket: websocket received {msg_str} msg: {append_str}, n_images {n_images}, extent {extent}, client_id {self._id}")
                        if msg["type"] == "sending_images":
                            await self._receive_images_http(job, image_format, n_images, images, uid, extent)
                        elif msg["type"] == "images_ready":
                            await self._receive_transient_images(job, filepath_prefix, n_images, images, extent)
                # end of minsky91 additions
                
                if msg["type"] == "executing" and msg["data"]["node"] is None:
                    job_id = msg["data"]["prompt_id"]
                    # minsky91: job logging, timing & stats
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        exec_elapsed_time = job.stats.set_exec_elapsed_time(is_end=True)
                        batch_exec_time = self._update_batch_timings(finished_job=job)
                        try:
                            job_stat_metadata = job.stats.metadata_from_stats(self._system_info)
                        except:
                            pass
                        job_workflow = job.stats.retrieve_workflow()
                    else:
                        exec_elapsed_time = 0.0
                        batch_exec_time = 0.0
                        job_stat_metadata = None
                        job_workflow = ""
                    # end of minsky91 additions
                    if local_id := self._clear_job(job_id):
                        if len(images) == 0:
                            # It may happen if the entire execution is cached and no images are sent.
                            images = last_images
                        if len(images) == 0:
                            # Still no images. Potential scenario: execution cached, but previous
                            # generation happened before the client was connected.
                            err = "No new images were generated because the inputs did not change."
                            await self._report(ClientEvent.error, local_id, error=err)
                        else:
                            last_images = images
                            # minsky91: stats, timing of jobs and batch execution, setting-based logs
                            exec_time_str = ""
                            if exec_elapsed_time > 0.0:
                                exec_time_str = f", exec time {exec_elapsed_time:.1f} sec." 
                            batch_timing_str = ""
                            if batch_exec_time > 0.0:
                                batch_timing_str = f", batch exec time {batch_exec_time:.1f} sec."
                            if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                                log.info(f"_listen_websocket: websocket executing msg: finished job {job_id}{exec_time_str}{batch_timing_str}")
                            else: 
                                log.info(f"finished gen job{exec_time_str}{batch_timing_str}")
                            if job_stat_metadata is not None:
                                await self._report(
                                    ClientEvent.finished, 
                                    local_id, 
                                    1, 
                                    images=images, 
                                    result=result, 
                                    metadata=job_stat_metadata, 
                                    workflow=json.dumps(job_workflow, indent=4)
                                )
                                continue
                            # end of minsky91 additions
                            await self._report(
                                ClientEvent.finished, local_id, 1, images=images, result=result
                            )

                elif msg["type"] in ("execution_cached", "executing", "progress"):
                    if self._active is not None and progress is not None:
                        progress.handle(msg)
                        await self._report(
                            ClientEvent.progress, self._active.local_id, progress.value
                        )
                    else:
                        log.error(f"Received message {msg} but there is no active job")

                if msg["type"] == "executed":
                    # minsky91: setting-based extended log info
                    if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                        log.info(f"_listen_websocket: websocket event executed: {msg}")
                    if job := self._get_active_job(msg["data"]["prompt_id"]):
                        text_output = _extract_text_output(job.local_id, msg)
                        if text_output is not None:
                            await self._messages.put(text_output)
                        pose_json = _extract_pose_json(msg)
                        if pose_json is not None:
                            result = pose_json

                if msg["type"] == "execution_error":
                    job = self._get_active_job(msg["data"]["prompt_id"])
                    if job:
                        # minsky91: job logging, timing & stats
                        job.stats.set_exec_elapsed_time(is_end=True)
                        # end of minsky91 additions
                        error = msg["data"].get("exception_message", "execution_error")
                        traceback = msg["data"].get("traceback", "no traceback")
                        log.error(f"Job {job} failed: {error}\n{traceback}")
                        self._clear_job(job.remote_id)
                        await self._report(ClientEvent.error, job.local_id, error=error)

                if msg["type"] == "etn_workflow_published":
                    name = f"{msg['data']['publisher']['name']} ({msg['data']['publisher']['id']})"
                    # minsky91: job logging, timing & stats
                    if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose] and last_workflow_name != name: 
                        log.info(f"_listen_websocket: websocket etn_workflow_published msg: name {name}")
                    last_workflow_name = name
                    workflow = SharedWorkflow(name, msg["data"]["workflow"])
                    await self._report(ClientEvent.published, "", result=workflow)


    async def listen(self):
        self._is_connected = True
        self._job_runner = asyncio.create_task(self._run())
        self._websocket_listener = asyncio.create_task(self._listen())

        try:
            while self._is_connected:
                yield await self._messages.get()
        except asyncio.CancelledError:
            pass

    async def interrupt(self):
        await self._post("interrupt", {})

    async def clear_queue(self):
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
                await self._report(ClientEvent.interrupted, job.local_id)
            except asyncio.QueueEmpty:
                break

        await self._post("queue", {"clear": True})
        self._jobs.clear()

    async def disconnect(self):
        if self._is_connected:
            self._is_connected = False
            self._job_runner.cancel()
            # minsky91: reset transient http storage
            if settings.server_mode is ServerMode.external:
                await self._reset_transient_storage()        
            self._websocket_listener.cancel()
            await asyncio.gather(
                self._job_runner,
                self._websocket_listener,
                self._unsubscribe_workflows(),
            )

    async def try_inspect(self, folder_name: str) -> dict[str, Any]:
        try:
            return await self._get(f"api/etn/model_info/{folder_name}")
        except NetworkError:
            return {}  # server has old external tooling version

    @property
    def queued_count(self):
        return len(self._jobs) + self._queue.qsize()

    @property
    def is_executing(self):
        return self._active is not None  

    # minsky91: methods to upload input images using server's transient storage, 
    # as a replacement for the workflow-embedding scheme which causes an unreasonable overhead  

    async def _upload_image_http(self, im_uid:int, image: Image, do_reset_storage: str = "no"):
        image_uid: str = f"{im_uid}"
        base_query = "api/etn/transient_image"
        do_post: bool = False if settings.upload_method == "Binary" else True
        if do_post:        # a POST method-based upload using a variety of encodings
            query = base_query + "/post"
        else:              # a binary PUT method-based upload with zero overhead
            query = f"{self.url}/" + base_query + "/upload"
        query += f"/{image_uid}/reset_storage={do_reset_storage}"
        if (max(image.width, image.height) >= settings.jpeg_res_threshold * 1024): 
           compr_format  = UploadImageFileFormat.jpeg_95
        else:
           compr_format  = UploadImageFileFormat.png_85
        format_str, quality = compr_format.value
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_upload_image_http: compressing a {image.width}x{image.height} image as {format_str}-{quality} and submiting query {query}")
        # compress to png or jpeg with a low compression setting
        compr_time_start = datetime.now()
        image_bytes = image.to_bytes(format=compr_format)
        compr_time, _ = get_time_diff(compr_time_start) 

        if do_post:
            img_compr_mb = len(image_bytes) / 1024.0**2
            time = datetime.now()
            if settings.upload_method == "Base64":
                image_bytes = image_bytes.toBase64()
            elif settings.upload_method == "a85":
                image_bytes = QByteArray(a85encode(b=image_bytes, pad=False))
            elif settings.upload_method == "b85":
                image_bytes = QByteArray(b85encode(b=image_bytes, pad=False))
            #elif settings.upload_method == "z85":
            #    image_bytes = QByteArray(z85encode(b=image_bytes, pad=False))  # supported from v3.13 on
            image_bytes = image_bytes.data().decode("utf-8")
            post_data = {"image_data": f"{image_bytes}", "encode": settings.upload_method} 
            _, enc_time_str = get_time_diff(time) 
            img_enc_mb = len(image_bytes) / 1024.0**2
            enc_ratio = (img_enc_mb-img_compr_mb) * 100 / img_compr_mb
            if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                enc_str = f"{settings.upload_method}-encoded data"
                mb_size_str = f"{img_enc_mb:.2f} MB size"
                enc_size_str = f"(from {img_compr_mb:.2f} MB unencoded, {enc_ratio:.1f}% overhead)"
                enc_time_str = f"encoding {enc_time_str}"
                log.info(f"_upload_image_http: posting {enc_str} of {mb_size_str} {enc_size_str}, {enc_time_str}")
        try:
            if do_post:
                result = await self._post(f"{query}", post_data)
            else:
                async for sent, total in self._requests.upload(query, image_bytes):
                    progress = sent / max(sent, total)
            return im_uid, len(image_bytes), compr_time
        except Exception as e:
            log.error(f"_upload_image_http: ERROR when submitting query {query} to the server: {str(e)}")
    
        return 0, 0, 0.0

                
    async def _upload_input_images(self, job: JobInfo, wf_image_uids: list[int]):
        total_uploaded_size = 0
        total_compr_time = 0.0
        uploaded_images = 0
        cached_images = 0
        non_cached_images = 0
        image_uploaded_size = 0
        work = job.work
        new_uids: list[int] = []
        begin_time = datetime.now()
        do_reset_storage = "no"
        kind_str = "upload" if settings.upload_method == "Binary" else "post" 
        reset_str = ""
            
        # use the image's unique ID attribute to avoid unnecessary uploads
        def check_and_append(the_image):
            nonlocal input_images, new_uids, cached_images, non_cached_images, wf_image_uids
            if the_image is not None:
                im_uid = image_uid(the_image)
                if len(wf_image_uids) == 0 or (im_uid in wf_image_uids):
                    if im_uid not in self._input_image_uids:
                        if im_uid not in new_uids:
                            new_uids.append(im_uid)
                            input_images.append(the_image)
                        non_cached_images += 1
                    else:
                        cached_images += 1
                    
        im_w = 0
        im_h = 0
        if work.images.initial_image is not None:
            im_w = work.images.initial_image.width
            im_h = work.images.initial_image.height
        input_images = ImageCollection()
        check_and_append(work.images.initial_image)
        check_and_append(work.images.hires_image)
        check_and_append(work.images.hires_mask)
        if work.conditioning is not None:
            for control in work.conditioning.control:
                if im_w == 0:
                    im_w = control.image.width
                    im_h = control.image.height
                check_and_append(control.image)
            for region in work.conditioning.regions:
                check_and_append(region.mask)
                for control in region.control:
                    check_and_append(control.image)
            
        if non_cached_images == 0:
            if cached_images == 0:
                skipped_str = "no input images" 
            else: 
                skipped_str = f"all {non_cached_images+cached_images} input image(s) cached, uids: {self._input_image_uids}"
            if len(input_images) == 0:
                n_images = cached_images
            else:
                n_images = len(input_images)
            job.stats.set_upload_params(0.0, n_images, 0.0, im_w, im_h, cached_images)
            if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                log.info(f"_upload_input_images: {skipped_str}, skipping upload")
            return True

        if cached_images == 0 and not (self.is_executing or len(self._jobs) > 0):
            # if there are no cached images needed for this job, submit resetting of the transient storage
            if len(input_images) == 1:
                # an inline reset done at the 1st image upload 
                do_reset_storage = "yes"   
                self._input_image_uids = []
            else:
                # with multiple concurrent uploads we do the storage reset as a separate preceding op 
                do_reset_storage = "no"
                if len(self._input_image_uids) > 0:
                    await self._reset_transient_storage()
            reset_str = " and resetting transient storage"
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_upload_input_images: preparing {non_cached_images} input image(s) for {kind_str}ing{reset_str}, current cached uids: {self._input_image_uids}")

        task_launch_time = datetime.now()
        upload_tasks: list[asyncio.Task] = []
        for input_image in input_images:
            im_uid = image_uid(input_image)
            if im_uid not in self._input_image_uids:
                upload_tasks.append(asyncio.create_task(self._upload_image_http(im_uid, input_image, do_reset_storage)))
        task_result = await asyncio.gather(*[t for t in upload_tasks])

        max_compr_time = 0.0
        for upload_task in upload_tasks:
            im_uid, image_uploaded_size, compr_time = task_result[upload_tasks.index(upload_task)]
            max_compr_time = max(compr_time, max_compr_time)
            if image_uploaded_size > 0:
                self._input_image_uids.append(im_uid)
                uploaded_images +=1
                total_uploaded_size += image_uploaded_size

        result = True
        max_compr_time_str = f"compr {max_compr_time:.1f} sec."
        time_taken, time_diff_str = get_time_diff(begin_time) 
        upload_mb_size: float = total_uploaded_size / 1024.0**2
        cached_str = ""
        uploaded_str = ""
        total_compr_time_str = ""
        if total_uploaded_size > 0: 
            uploaded_str = f"{kind_str}ed {uploaded_images} image(s) of {upload_mb_size:.2f} MB total size, total {time_diff_str}"
        if cached_images != 0:
            cached_str = f"{cached_images}/{len(self._input_image_uids)} image(s) cached"
            if total_uploaded_size > 0:
                cached_str = ", " + cached_str
            else:
                result = False
        if total_uploaded_size > 0:
            max_compr_time_str = ", " + max_compr_time_str
        job.stats.set_upload_params(time_taken, len(input_images), upload_mb_size, im_w, im_h, cached_images)
        current_uids_str = f"cached uids: {self._input_image_uids}"
        if total_uploaded_size > 0 or cached_images > 0:
            current_uids_str = ", " + current_uids_str
        if image_uploaded_size > 0 or cached_images > 0:
            if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                log.info(f"_upload_input_images: {uploaded_str}{max_compr_time_str}{cached_str}{current_uids_str}")
            elif Verbosity(settings.logfile_verbosity) in [Verbosity.medium]: 
                log.info(f"{uploaded_str}")
        #await asyncio.sleep(0.75)  # prevent hasty actions by the server 

        return result
    
# minsky91: methods to download output images using server's transient storage,
# as a replacement for slow websocket-based transfers
    
    async def _receive_image_http(self, format: str, uid: str, image_index:int = 1):
        try:
            query = f"api/etn/transient_image/download/{uid}/{image_index}/{format}"
            if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                log.info(f"_receive_image_http: submiting query {query}")
            result = await self._get(query)
            #fullfilename = f"C:\\Download\\received_image.{format}"
            #with open(fullfilename, "wb") as f:
            #    f.write(result)
            return result
        except NetworkError:
            log.error(f"_receive_image_http: server response to query {query}: not found or other error")
            
        return None  # not found or unexpected response
        
    
    async def _receive_images_http(self, job: JobInfo, image_format: str, n_images: int, images: ImageCollection, uid: str, extent: Extent):
        if extent is not None and (max(extent.width, extent.height) >= settings.jpeg_res_threshold * 1024): 
            image_format = "JPEG"
        img_total_mb_size = 0
        total_time_taken = 0.0
        img_w = 0
        img_h = 0
        for i in range(1, n_images+1, 1):
            time = datetime.now()
            image_received = await self._receive_image_http(image_format, uid, i)
            time_taken, time_diff_str = get_time_diff(time)
            total_time_taken += time_taken
            if image_received is not None:
                img_mb_size: float = len(image_received) / 1024.0**2
                img_total_mb_size += img_mb_size
                if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                    log.info(f"_receive_images_http: server returned a {img_mb_size:.2f} MB image, {time_diff_str}")
                img = Image.from_bytes(image_received)
                images.append(img)
                if i == 1:
                    img_w = img.width
                    img_h = img.height
            else:
                break
        if job is not None:
            job.stats.set_download_params(download_time=total_time_taken, n_images=n_images, img_mb_size=img_total_mb_size, img_format=image_format, dim_x=img_w, dim_y=img_h)
                

    async def _receive_view_image(self, subfolder: str, filename: str, do_preview: bool=False):
        query: str = f"/view?filename={filename}&type=temp&subfolder={subfolder}"
        if do_preview:
            query += "&preview=jpeg;95"
        else:
            query += "&channel=rgba"
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_receive_view_image: submiting query {query}")
        try:
            time = datetime.now()
            result = await self._get(query)
            #with open("C:\\Download\\"+filename, "wb") as f:
            #    f.write(result)
            return result
        except NetworkError:
            log.error(f"_receive_view_image: server response to query {query}: not found or other error")
            return None 
            
    async def _receive_transient_images(self, job: JobInfo, filepath_prefix: str, n_images: int, images: ImageCollection, extent: Extent):
        i: int = len(filepath_prefix) - 1
        subfolder = filepath_prefix
        do_preview = False
        if extent is not None and (max(extent.width, extent.height) >= settings.jpeg_res_threshold * 1024): 
            do_preview = True
        while i >= 0:
            if filepath_prefix[i] == '/' or filepath_prefix[i] == '\\':
                filename_prefix = filepath_prefix[i+1:]
                subfolder = filepath_prefix[:i]
                break
            i = i - 1
        for i in range(1, n_images+1, 1):
            filename = filename_prefix + f"_{i:05}_.png"  # following Comfy filenaming conventions
            if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                log.info(f"_receive_transient_images requesting image file: {subfolder}/{filename}, jpeg preview mode {do_preview}")
            time = datetime.now()
            image_received = await self._receive_view_image(subfolder, filename, do_preview)
            _, time_diff_str = get_time_diff(time)
            img_mb_size: float = 0.0
            if image_received is not None:
                img_mb_size = len(image_received) / 1024.0**2
                if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
                    log.info(f"_receive_transient_images: server returned a {img_mb_size:.2f} MB image, {time_diff_str}")
            else:
                log.info(f"_receive_transient_images ERROR: failure while receiving image, {time_diff_str}")
            if image_received is not None:
               images.append(Image.from_bytes(image_received))
            else:
                break

    async def _reset_transient_storage(self):
        self._input_image_uids = []
        await self._get("api/etn/transient_image/reset_storage")        
        
    # forward-propagate the job's active exec time within the batch 
    def _update_batch_timings(self, finished_job: JobInfo):
        if finished_job is None:
            return 0.0
        add_active_time = finished_job.stats.prep_elapsed_time + finished_job.stats.exec_elapsed_time
        result = finished_job.stats.get_batch_active_time() + add_active_time  # this is returned if the job is last in the batch
        batch_jobs = 0
        if finished_job.work.sampling != None:
            next_seed = finished_job.work.sampling.seed
            # follow consecutive sample seeds in the jobs to update the batch's timing data
            for job in self._jobs:
                if job.work.sampling.seed == next_seed + settings.batch_size:
                    batch_jobs += 1
                    job.stats.add_batch_active_time(add_active_time)
                    next_seed += settings.batch_size
                    result = 0.0   # some jobs in the batch are still pending
                else:
                    break
        for job in self._jobs:
            job.stats.set_job_batch_size(batch_jobs+1)
        
        if result != 0.0:
            finished_job.stats.set_last_in_the_batch()
            finished_job.stats.set_job_batch_size(batch_jobs+1)
            finished_job.stats.add_batch_active_time(add_active_time)
            
        return result
        
# end of minsky91 additions

    async def refresh(self):
        nodes, checkpoints, diffusion_models, diffusion_gguf = await asyncio.gather(
            self._get("object_info"),
            self.try_inspect("checkpoints"),
            self.try_inspect("diffusion_models"),
            self.try_inspect("unet_gguf"),
        )
        diffusion_models.update(diffusion_gguf)
        self._refresh_models(nodes, checkpoints, diffusion_models)

    def _refresh_models(self, nodes: dict, checkpoints: dict | None, diffusion_models: dict | None):
        models = self.models

        def parse_model_info(models: dict, model_format: FileFormat):
            parsed = (
                (
                    filename,
                    Arch.from_string(info["base_model"], info.get("type", "eps")),
                    info.get("is_inpaint", False),
                    info.get("is_refiner", False),
                )
                for filename, info in models.items()
            )
            return {
                filename: CheckpointInfo(filename, arch, model_format)
                for filename, arch, is_inpaint, is_refiner in parsed
                if not (arch is None or (is_inpaint and arch is not Arch.flux) or is_refiner)
            }

        if checkpoints:
            models.checkpoints = parse_model_info(checkpoints, FileFormat.checkpoint)
        else:
            models.checkpoints = {
                filename: CheckpointInfo.deduce_from_filename(filename)
                for filename in nodes["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            }
        if diffusion_models:
            models.checkpoints.update(parse_model_info(diffusion_models, FileFormat.diffusion))

        models.vae = nodes["VAELoader"]["input"]["required"]["vae_name"][0]
        models.loras = nodes["LoraLoader"]["input"]["required"]["lora_name"][0]

        if gguf_node := nodes.get("UnetLoaderGGUF", None):
            for name in gguf_node["input"]["required"]["unet_name"][0]:
                if name not in models.checkpoints:
                    models.checkpoints[name] = CheckpointInfo(name, Arch.flux, FileFormat.diffusion)
        else:
            # minsky91: setting-based extended log info
            if Verbosity(settings.logfile_verbosity) in [Verbosity.medium, Verbosity.verbose]: 
                log.info("GGUF support: node is not installed.")

    async def translate(self, text: str, lang: str):
        try:
            return await self._get(f"api/etn/translate/{lang}/{text}")
        except NetworkError as e:
            log.error(f"Could not translate text: {str(e)}")
            return text

    async def _subscribe_workflows(self):
        # minsky91: setting-based extended log info
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_subscribe_workflows: client_id {self._id}")
        try:
            await self._post("api/etn/workflow/subscribe", {"client_id": self._id})
        except Exception as e:
            log.error(f"Couldn't subscribe to shared workflows: {str(e)}")

    async def _unsubscribe_workflows(self):
        # minsky91: setting-based extended log info
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_unsubscribe_workflows: client_id {self._id}")
        try:
            await self._post("api/etn/workflow/unsubscribe", {"client_id": self._id})
        except Exception as e:
            log.error(f"Couldn't unsubscribe from shared workflows: {str(e)}")

    @property
    def missing_resources(self):
        return MissingResources(self._supported_archs)

    @property
    def features(self):
        return self._features

    @property
    def performance_settings(self):
        return PerformanceSettings(
            batch_size=settings.batch_size,
            resolution_multiplier=settings.resolution_multiplier,
            max_pixel_count=settings.max_pixel_count,
            tiled_vae=settings.tiled_vae,
            dynamic_caching=settings.dynamic_caching and self.features.wave_speed,
        )

    async def upload_loras(self, work: WorkflowInput, local_job_id: str):
        # minsky91: setting-based extended log info
        log_issued: bool = False;
        for file in loras_to_upload(work, self.models):
            # minsky91: setting-based extended log info
            if not log_issued:
                log.info(f"Uploading lora models to {self.url}")
                log_issued = True
            try:
                assert file.path is not None
                url = f"{self.url}/api/etn/upload/loras/{file.id}"
                # minsky91: setting-based extended log info
                if Verbosity(settings.logfile_verbosity) in [Verbosity.medium, Verbosity.verbose]: 
                    log.info(f"Uploading lora model {file.id} to {url}")
                data = file.path.read_bytes()
                async for sent, total in self._requests.upload(url, data):
                    progress = sent / max(sent, total)
                    await self._report(ClientEvent.upload, local_job_id, progress)

                await self.refresh()
            except Exception as e:
                raise Exception(_("Error during upload of LoRA model") + f" {file.path}: {str(e)}")

    def _get_active_job(self, remote_id: str) -> Optional[JobInfo]:
        if self._active and self._active.remote_id == remote_id:
            return self._active
        elif self._active:
            log.warning(f"Received message for job {remote_id}, but job {self._active} is active")
        if len(self._jobs) == 0:
            log.warning(f"Received unknown job {remote_id}")
            return None
        active = next((j for j in self._jobs if j.remote_id == remote_id), None)
        if active is not None:
            return active
        return None

    async def _start_job(self, remote_id: str):
        if self._active is not None:
            log.warning(f"Started job {remote_id}, but {self._active} was never finished")
        if len(self._jobs) == 0:
            log.warning(f"Received unknown job {remote_id}")
            return None

        if await self._jobs[0].get_remote_id() == remote_id:
            return self._jobs.popleft()

        log.warning(f"Started job {remote_id}, but {self._jobs[0]} was expected")
        for job in self._jobs:
            if await job.get_remote_id() == remote_id:
                self._jobs.remove(job)
                return job
        return None

    def _clear_job(self, job_remote_id: str | asyncio.Future | None):
        if self._active is not None and self._active.remote_id == job_remote_id:
            result = self._active.local_id
            self._active = None
            return result
        return None

    def _check_workload(self, sdver: Arch) -> list[ResourceId]:
        models = self.models
        missing: list[ResourceId] = []
        for id in resources.required_resource_ids:
            if id.arch is not Arch.all and id.arch is not sdver:
                continue
            if models.find(id) is None:
                missing.append(id)
        has_checkpoint = any(cp.arch is sdver for cp in models.checkpoints.values())
        if not has_checkpoint and sdver not in [Arch.illu, Arch.illu_v]:
            missing.append(ResourceId(ResourceKind.checkpoint, sdver, "Diffusion model checkpoint"))
        if len(missing) > 0:
            # minsky91: setting-based extended log info
            if Verbosity(settings.logfile_verbosity) in [Verbosity.medium, Verbosity.verbose]: 
                log.info(f"{sdver.value}: missing {len(missing)} models")
        return missing


def parse_url(url: str):
    url = url.strip("/")
    url = url.replace("0.0.0.0", "127.0.0.1")
    if not url.startswith("http"):
        url = f"http://{url}"
    return url


def websocket_url(url_http: str):
    return url_http.replace("http", "ws", 1)


def _check_for_missing_nodes(nodes: dict):
    def missing(node: str, package: CustomNode):
        if node not in nodes:
            log.error(f"Missing required node {node} from package {package.name} ({package.url})")
            return True
        return False

    return [
        package
        for package in resources.required_custom_nodes
        if any(missing(node, package) for node in package.nodes)
    ]


def _find_model(
    model_list: Sequence[str],
    kind: ResourceKind,
    sdver: Arch,
    identifier: ControlMode | UpscalerName | str,
):
    search_paths = resources.search_path(kind, sdver, identifier)
    if search_paths is None:
        return None

    def sanitize(p):
        return p.replace("\\", "/").lower()

    def match(filename: str, pattern: str):
        filename = sanitize(filename)
        pattern = pattern.lower()
        return all(p in filename for p in pattern.split("*"))

    matches = (m for p in search_paths for m in model_list if match(m, p))
    # if there are multiple matches, prefer the one with "krita" in the path
    prio = sorted(matches, key=lambda m: 0 if "krita" in m else len(m))
    found = next(iter(prio), None)
    model_id = identifier.name if isinstance(identifier, Enum) else identifier
    model_name = f"{kind.value} {model_id}"

    if found is None and resources.is_required(kind, sdver, identifier):
        log.warning(f"Missing {model_name} for {sdver.value}")
        log.info(f"-> No model matches search paths: {', '.join(p.lower() for p in search_paths)}")
        log.info(f"-> Available models: {', '.join(sanitize(m) for m in model_list)}")
    elif found is None:
        # minsky91: setting-based extended log info
        if Verbosity(settings.logfile_verbosity) in [Verbosity.medium, Verbosity.verbose]: 
            log.info(
                f"Optional {model_name} for {sdver.value} not found (search path:"
                f" {', '.join(search_paths)})"
            )
    else:
        # minsky91: setting-based extended log info
        if Verbosity(settings.logfile_verbosity) in [Verbosity.medium, Verbosity.verbose]: 
            log.info(f"Found {model_name} for {sdver.value}: {found}")
    return found


def find_model(model_list: Sequence[str], id: ResourceId):
    return _find_model(model_list, id.kind, id.arch, id.identifier)


def _find_text_encoder_models(model_list: Sequence[str]):
    kind = ResourceKind.text_encoder
    return {
        resource_id(kind, Arch.all, te): _find_model(model_list, kind, Arch.all, te)
        for te in ["clip_l", "clip_g", "t5"]
    }


def _find_control_models(model_list: Sequence[str]):
    kind = ResourceKind.controlnet
    return {
        resource_id(kind, ver, mode): _find_model(model_list, kind, ver, mode)
        for mode, ver in product(ControlMode, Arch.list())
        if mode.is_control_net
    }


def _find_ip_adapters(model_list: Sequence[str]):
    kind = ResourceKind.ip_adapter
    return {
        resource_id(kind, ver, mode): _find_model(model_list, kind, ver, mode)
        for mode, ver in product(ControlMode, Arch.list())
        if mode.is_ip_adapter
    }


def _find_clip_vision_model(model_list: Sequence[str]):
    clip_vision_sd = ResourceId(ResourceKind.clip_vision, Arch.all, "ip_adapter")
    model = find_model(model_list, clip_vision_sd)
    clip_vision_flux = ResourceId(ResourceKind.clip_vision, Arch.flux, "redux")
    clip_vision_illu = ResourceId(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
    return {
        clip_vision_sd.string: model,
        clip_vision_flux.string: find_model(model_list, clip_vision_flux),
        clip_vision_illu.string: find_model(model_list, clip_vision_illu),
    }


def _find_style_models(model_list: Sequence[str]):
    redux_flux = ResourceId(ResourceKind.ip_adapter, Arch.flux, ControlMode.reference)
    return {redux_flux.string: find_model(model_list, redux_flux)}


def _find_upscalers(model_list: Sequence[str]):
    kind = ResourceKind.upscaler
    models = {
        resource_id(kind, Arch.all, name): _find_model(model_list, kind, Arch.all, name)
        for name in UpscalerName
    }
    default_id = resource_id(kind, Arch.all, UpscalerName.default)
    if models[default_id] is None and len(model_list) > 0:
        models[default_id] = models[resource_id(kind, Arch.all, UpscalerName.fast_4x)]
    return models


def _find_loras(model_list: Sequence[str]):
    kind = ResourceKind.lora
    common_loras = list(product(["hyper", "lcm", "face"], [Arch.sd15, Arch.sdxl]))
    sdxl_loras = [("lightning", Arch.sdxl)]
    flux_loras = [(ControlMode.depth, Arch.flux), (ControlMode.canny_edge, Arch.flux)]
    return {
        resource_id(kind, arch, name): _find_model(model_list, kind, arch, name)
        for name, arch in chain(common_loras, sdxl_loras, flux_loras)
    }


def _find_vae_models(model_list: Sequence[str]):
    kind = ResourceKind.vae
    return {
        resource_id(kind, ver, "default"): _find_model(model_list, kind, ver, "default")
        for ver in Arch.list()
    }


def _find_inpaint_models(model_list: Sequence[str]):
    kind = ResourceKind.inpaint
    ids: list[tuple[Arch, str]] = [
        (Arch.all, "default"),
        (Arch.sdxl, "fooocus_head"),
        (Arch.sdxl, "fooocus_patch"),
    ]
    return {
        resource_id(kind, ver, name): _find_model(model_list, kind, ver, name) for ver, name in ids
    }


def _ensure_supported_style(client: Client):
    styles = filter_supported_styles(Styles.list(), client)
    if len(styles) == 0:
        supported_checkpoints = (
            cp.filename
            for cp in client.models.checkpoints.values()
            if client.supports_arch(cp.arch)
        )
        checkpoint = next(iter(supported_checkpoints), None)
        if checkpoint is None:
            log.warning("No checkpoints found for any of the supported workloads!")
            if len(client.models.checkpoints) == 0:
                raise Exception(_("No diffusion model checkpoints found"))
            return
        log.info(f"No supported styles found, creating default style with checkpoint {checkpoint}")
        default = next((s for s in Styles.list() if s.filename == "default.json"), None)
        if default:
            default.checkpoints = [checkpoint]
            default.save()
        else:
            Styles.list().create("default", checkpoint)


async def _list_languages(client: ComfyClient) -> list[TranslationPackage]:
    try:
        result = await client._get("api/etn/languages")
        return TranslationPackage.from_list(result)
    except NetworkError as e:
        log.error(f"Could not list available languages for translation: {str(e)}")
        return []


def _extract_message_png_image(data: memoryview, need_preview: bool):
    # minsky91: added progress preview and extended logs
    global last_timestamp
    s = struct.calcsize(">II")
    if len(data) > s:
        event, format = struct.unpack_from(">II", data)

        # minsky91: setting-based extended log info & job stats
        image = None
        _, time_diff_str = get_time_diff(last_timestamp) 
        img_mb_size: float = len(data) / 1024.0**2
        preview_str = "" if event == 1 and format == 2 else " progress preview"
        if format == 1:
            preview_str += " JPEG"
        else:
            preview_str += " PNG"
        if event == 1 and (need_preview or format == 2):  # format: JPEG=1, PNG=2
            image = Image.from_bytes(data[s:])
        size_str = f" of {img_mb_size:.2f} MB size"
        dim_str = f" {image.width}x{image.height}" if image is not None else ""
        if Verbosity(settings.logfile_verbosity) in [Verbosity.verbose]: 
            log.info(f"_extract_message_png_image: event {str(event)} format {str(format)}, received a{dim_str}{preview_str} image{size_str}, {time_diff_str}")
        last_timestamp = datetime.now()
        return image, (format == 1)
            
    return None, (format == 1)
    # end of minsky91 additions and modifications


def _extract_pose_json(msg: dict):
    try:
        output = msg["data"]["output"]
        if output is not None and "openpose_json" in output:
            return json.loads(output["openpose_json"][0])
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
    return None


def _extract_text_output(job_id: str, msg: dict):
    try:
        output = msg["data"]["output"]
        if output is not None and "text" in output:
            key = msg["data"].get("node")
            payload = output["text"]
            name, text, mime = (None, None, "text/plain")
            if isinstance(payload, list) and len(payload) >= 1:
                payload = payload[0]
            if isinstance(payload, dict):
                text = payload.get("text")
                name = payload.get("name")
                mime = payload.get("content-type", mime)
            elif isinstance(payload, str):
                text = payload
                name = f"Node {key}"
            if text is not None and name is not None:
                result = TextOutput(key, name, text, mime)
                return ClientMessage(ClientEvent.output, job_id, result=result)
    except Exception as e:
        log.warning(f"Error processing message, error={str(e)}, msg={msg}")
    return None
