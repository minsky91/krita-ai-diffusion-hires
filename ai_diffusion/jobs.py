from __future__ import annotations
from collections import deque
from dataclasses import dataclass, fields, field
from datetime import datetime
from enum import Enum, Flag
from typing import Any, NamedTuple, TYPE_CHECKING
from PyQt5.QtCore import QObject, pyqtSignal

from .image import Bounds, ImageCollection
from .settings import settings
from .style import Style
from .util import ensure

if TYPE_CHECKING:
    from . import control

import json

# minsky91:
from .resources import Arch
from .util import client_logger as log
from .style import StyleSettings, SamplerPresets
from .settings import Verbosity
from .util import flatten
from textwrap import wrap
from .image import multiple_of
# end of minsky91 additions

class JobState(Flag):
    queued = 0
    executing = 1
    finished = 2
    cancelled = 3


class JobKind(Enum):
    diffusion = 0
    control_layer = 1
    upscaling = 2
    live_preview = 3
    animation_batch = 4  # single frame as part of an animation batch
    animation_frame = 5  # just a single frame
    animation = 6  # full animation in one job


@dataclass
class JobRegion:
    layer_id: str
    prompt: str
    bounds: Bounds
    is_background: bool = False

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["bounds"] = Bounds(*data["bounds"])
        return JobRegion(**data)


@dataclass
class JobParams:
    bounds: Bounds
    name: str  # used eg. as name for new layers created from this job
    regions: list[JobRegion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    has_mask: bool = False
    frame: tuple[int, int, int] = (0, 0, 0)
    animation_id: str = ""

    @staticmethod
    def from_dict(data: dict[str, Any]):
        data["bounds"] = Bounds(*data["bounds"])
        data["regions"] = [JobRegion.from_dict(r) for r in data.get("regions", [])]
        if "metadata" not in data:  # older documents before version 1.26.0
            data["name"] = data.get("prompt", "")
            data["metadata"] = {}
            _move_field(data, "prompt", data["metadata"])
            _move_field(data, "negative_prompt", data["metadata"])
            _move_field(data, "strength", data["metadata"])
            _move_field(data, "style", data["metadata"])
            _move_field(data, "sampler", data["metadata"])
            _move_field(data, "checkpoint", data["metadata"])
        return JobParams(**data)

    @classmethod
    def equal_ignore_seed(cls, a: JobParams | None, b: JobParams | None):
        if a is None or b is None:
            return a is b
        field_names = (f.name for f in fields(cls) if not f.name == "seed")
        return all(getattr(a, name) == getattr(b, name) for name in field_names)

    def set_style(self, style: Style, checkpoint: str):
        self.metadata["style"] = style.filename
        self.metadata["style_preset_name"] = style.name
        self.metadata["checkpoint"] = checkpoint
        self.metadata["loras"] = style.loras
        self.metadata["style_prompt"] = style.style_prompt
        self.metadata["style_negative_prompt"] = style.negative_prompt
        self.metadata["architecture"] = style.architecture.resolve(checkpoint)
        # minsky91: extended the collected metadata code
        #self.metadata["sampler"] = f"{style.sampler} ({style.sampler_steps} / {style.cfg_scale})"
        sampler_str = style.sampler
        preset = SamplerPresets.instance()[sampler_str]
        if " - " in sampler_str:
            before_dash, sampler_str = sampler_str.split(" - ", 1)
        self.metadata["sampler"] = sampler_str
        self.metadata["scheduler"] = preset.scheduler
        self.metadata["steps"] = style.sampler_steps
        self.metadata["cfg_scale"] = style.cfg_scale
        self.metadata["vae"] = style.vae
        self.metadata["clip_skip"] = style.clip_skip
        self.metadata["v_pred"] = style.v_prediction_zsnr
        self.metadata["rescale_cfg"] = style.rescale_cfg
        self.metadata["sag"] = style.self_attention_guidance
        self.metadata["pref_res"] = style.preferred_resolution

    @property
    def prompt(self):
        return self.metadata.get("prompt", "")

    @property
    def style(self):
        return self.metadata.get("style", "")

    @property
    def style_preset_name(self):
        return self.metadata.get("style_preset_name", "")

    @property
    def strength(self):
        return self.metadata.get("strength", 1.0)

    # minsky91: extended the collected metadata code
    def seed(self):
        return self.metadata.get("seed", 0.0)

    @property
    def negative_prompt(self):
        return self.metadata.get("negative_prompt", "")
    
    @property
    def steps(self):
        return self.metadata.get("steps", "")
    
    @property
    def sampler(self):
        return self.metadata.get("sampler", "")
    
    @property
    def scheduler(self):
        return self.metadata.get("scheduler", "")
    
    @property
    def cfg_scale(self):
        return self.metadata.get("cfg_scale", "")
    
    @property
    def checkpoint(self):
        return self.metadata.get("checkpoint", "")

    @property
    def style_prompt(self):
        return self.metadata.get("style_prompt", "")
        
    @property
    def style_negative_prompt(self):
        return self.metadata.get("style_negative_prompt", "")
        
    @property
    def architecture(self):
        arch = self.metadata.get("architecture", "")
        if isinstance(arch, str):
            return arch
        else:
            return arch.value
        
    @property
    def loras(self):
        lora_str = ""
        for lora in self.metadata.get("loras", ""):
            if lora.get("enabled", True):
                lora_strength = lora.get("strength", 0.0)
                if lora_str != "":
                    lora_str += "\n       " 
                lora_str += lora.get("name", "") + f": strength {lora_strength:.1f} "
        return lora_str

    @property
    def vae(self):
        if self.metadata.get("vae", "") != StyleSettings.vae.default:
            return self.metadata.get("vae", "")
        else:
            return ""

    @property
    def clip_skip(self):
        if self.metadata.get("clip_skip", 0) != StyleSettings.clip_skip.default:
            return self.metadata.get("clip_skip", 0)  
        else:
            return 0
    
    @property
    def pref_res(self):
        if self.metadata.get("pref_res", 0) != StyleSettings.preferred_resolution.default:
            return self.metadata.get("pref_res", 0)  
        else:
            return 0
    
    @property
    def v_pred(self):
        if self.metadata.get("v_pred", False) != StyleSettings.v_prediction_zsnr.default:
            return self.metadata.get("v_pred", False) 
        else:
            return None
    
    @property
    def rescale_cfg(self):
        if self.metadata.get("rescale_cfg", 0) != StyleSettings.rescale_cfg.default:
            return self.metadata.get("rescale_cfg", 0)  
        else:
            return 0
    
    @property
    def rescale_cfg(self):
        return self.metadata.get("rescale_cfg", "")
    
    @property
    def sag(self):
        if self.metadata.get("sag", False) != StyleSettings.self_attention_guidance.default:
            return self.metadata.get("sag", False) 
        else:
            return None
    
    @property
    def job_resolution_multiplier(self):
        if self.metadata.get("resolution_multiplier"):
            return self.metadata.get("resolution_multiplier", 1.0)
        else:
            return settings.resolution_multiplier

    @property
    def input_resolution(self):
        if self.metadata.get("input_resolution"):
            return self.metadata.get("input_resolution", [0, 0])
        else:
            return 0, 0

    @property
    def canvas_resolution(self):
        if self.metadata.get("canvas_resolution"):
            return self.metadata.get("canvas_resolution", [0, 0])
        else:
            return 0, 0

    @property
    def output_resolution(self):
        if self.metadata.get("output_resolution"):
            return self.metadata.get("output_resolution", [0, 0])
        else:
            return 0, 0
            
    @property
    def mask_resolution(self):
        if self.metadata.get("mask_resolution"):
            return self.metadata.get("mask_resolution", [0, 0])
        else:
            return 0, 0
            
    @property
    def control_layers(self):
        if self.metadata.get("control_layers"):
            return self.metadata.get("control_layers", [])
        else:
            return None
            
    # recover & restore style from a Krita AI-written png file 
    @staticmethod
    def recover_style_preset(text: str):
        if "Style Preset filename: " in text:
            _, filename_plus_tail = text.split("Style Preset filename: ", 1)
            style_filename, _ = filename_plus_tail.split(".json", 1)
            return style_filename + ".json"
        else:
            return ""
    
    # end of minsky91 additions

class Job:
    id: str | None
    kind: JobKind
    state = JobState.queued
    params: JobParams
    control: "control.ControlLayer | None" = None
    timestamp: datetime
    results: ImageCollection
    in_use: dict[int, bool]
    # minsky91: add extended job metadata stats & basic workflow
    stat_metadata: str
    imageless_workflow: str
    # end of minsky91 additions

    def __init__(self, id: str | None, kind: JobKind, params: JobParams):
        self.id = id
        self.kind = kind
        self.params = params
        self.timestamp = datetime.now()
        self.results = ImageCollection()
        self.in_use = {}
        # minsky91: add extended job metadata stats & basic workflow
        self.stat_metadata = ""
        self.imageless_workflow = ""
        # end of minsky91 additions

    def result_was_used(self, index: int):
        return self.in_use.get(index, False)

    # minsky91: added extended job metadata stats & basic workflow
    
    def get_stat_metadata(self, job: Job, verbosity_setting: Verbosity):
        if verbosity_setting in [Verbosity.medium, Verbosity.verbose]:
            return "Generation stats:\n\n" + "\n".join(ml for ml in job.stat_metadata)
        else:
            return "".join(ml for ml in job.stat_metadata)

    def get_workflow(self, job: Job):
        return job.imageless_workflow
        
    def get_collected_metadata(self, job: Job, verbosity_setting: Verbosity, print_header: str | None, wrap_text: int | None):
        is_verbose: bool = verbosity_setting in [Verbosity.verbose] 
        is_verbose_or_medium: bool = verbosity_setting in [Verbosity.medium, Verbosity.verbose] 
        collected_metadata: list[str] = []
        if print_header is not None and print_header != "":
            # tooltip or clipboard format
            collected_metadata.append(print_header)
        collected_metadata.append(job.params.prompt)
        collected_metadata.append("Negative prompt: " + job.params.negative_prompt + "  ")
        if verbosity_setting in [Verbosity.medium, Verbosity.verbose]:
            collected_metadata.append("Style prompt: " + job.params.style_prompt + "  ")
            collected_metadata.append("Style negative prompt: " + job.params.style_negative_prompt + "  ")
        collected_metadata.append("Steps: " + f"{job.params.steps}")
        collected_metadata.append("Sampler: " + job.params.sampler)
        if len(job.params.scheduler) >= 2:
            collected_metadata.append("Schedule type: " + job.params.scheduler[:1].upper()+job.params.scheduler[1:])
        collected_metadata.append("CFG scale: " + f"{job.params.cfg_scale}")
        collected_metadata.append("Seed: " + f"{job.params.seed}")
        collected_metadata.append("Model: " + job.params.checkpoint + f" ({self.params.architecture})")
        collected_metadata.append("Denoising strength: " + f"{job.params.strength}")
        collected_metadata.append("Style Preset: " + job.params.style_preset_name)
        if is_verbose:
            collected_metadata.append("Style Preset filename: " + job.params.style)
        job_loras = job.params.loras 
        if job_loras != "":
            collected_metadata.append("LoRas: " + job_loras)
        res_mult = job.params.job_resolution_multiplier
        if is_verbose_or_medium:
            if job.params.vae != "":
                collected_metadata.append("VAE: " + job.params.vae)
            if job.params.clip_skip != 0:
                collected_metadata.append("Clip Skip: " + f"{job.params.clip_skip}")
            if job.params.pref_res != 0:
                collected_metadata.append("Preferred Resolution: " + f"{job.params.pref_res}")
            if job.params.v_pred is not None:
                collected_metadata.append("V-Prediction: " + f"{job.params.v_pred}")
            if is_verbose:
                if job.params.rescale_cfg != 0:
                    collected_metadata.append("Rescale CFG: " + f"{job.params.rescale_cfg}")
            if job.params.sag is not None:
                collected_metadata.append("Self-attention Guidance: " + f"{job.params.sag}")
            collected_metadata.append("Resolution multiplier: " + f"{res_mult}")
        image_w, image_h = job.params.canvas_resolution
        if image_w != 0:
            collected_metadata.append("Canvas resolution: " + f"{image_w}x{image_h} pixels")
        image_w, image_h = job.params.input_resolution
        if is_verbose_or_medium:
            if image_w != 0:
                input_res_str = "Input resolution: "
                eff__w = image_w*res_mult
                eff__h = image_h*res_mult
                if res_mult != 1.0:
                    input_res_str = "Effective " + input_res_str
                    eff__w = multiple_of(eff__w, 8)
                    eff__h = multiple_of(eff__h, 8)
                input_res_str += f"{int(eff__w)}x{int(eff__h)} pixels"
                collected_metadata.append(input_res_str)
            image_w, image_h = job.params.output_resolution
            if image_w != 0:
                collected_metadata.append("Output resolution: " + f"{image_w}x{image_h} pixels")
            image_w, image_h = job.params.mask_resolution
            if image_w != 0:
                collected_metadata.append("Mask resolution: " + f"{image_w}x{image_h} pixels")
        if "control_layers" in job.params.metadata:
            for control in job.params.metadata["control_layers"]:
                collected_metadata.append(control)
        if "regions" in job.params.metadata:
            for region in job.params.metadata["regions"]:
                collected_metadata.append(region)

        # gather now the models used, by parsing the workflow seems the only practical way
        the_workflow = job.get_workflow(job)
        model_str = ""
        def gather_models(model_key, model_type):
            nonlocal model_str, the_workflow
            text = the_workflow
            model_kstr = f'\"{model_key}\": \"'
            while model_kstr in text:
                if len(text) <= len(model_key)+1:
                    break
                _, modelname_plus_tail = text.split(model_kstr, 1)
                if not '\"' in modelname_plus_tail:
                    break
                model_filename, text = modelname_plus_tail.split('\"', 1)
                if len(model_str) > 0:
                    model_str += "\n               "
                model_str += model_filename + f" ({model_type})" 
        if is_verbose_or_medium:
            gather_models("control_net_name", "ControlNet")
            gather_models("ipadapter_file", "IPAdapter")
            gather_models("model_name", "Upscale or Inpaint model")
            gather_models("patch", "Inpaint patch model")
            gather_models("style_model_name", "Style model")
            if is_verbose:
                gather_models("unet_name", "UNETLoader")
                gather_models("clip_name", "CLIPVision")
                gather_models("clip_name1", "DualCLIPLoader")
                gather_models("clip_name2", "DualCLIPLoader")
                gather_models("vae_name", "VAE")
        if model_str != "":
            model_str = "Models used: " + model_str
            collected_metadata.append(model_str)

        if job.stat_metadata:
            if verbosity_setting in [Verbosity.medium, Verbosity.verbose]:
                collected_metadata.append("")
                collected_metadata.append("Generation stats:")
            for meta_line in job.stat_metadata:
                # lines in stat_metadata are indented to indicate their verbosity level
                if len(meta_line) >= 4 and meta_line[:4] == "    ":
                    if is_verbose:
                        stat_meta_str = meta_line[4:]
                    else:
                        continue
                elif len(meta_line) >= 2 and meta_line[:2] == "  ":
                    if is_verbose_or_medium:
                        stat_meta_str = meta_line[2:]
                    else:
                        continue
                else:
                    stat_meta_str = meta_line
                collected_metadata.append(stat_meta_str)
                
        if print_header is not None:
            # tooltip or clipboard format: insert newlines and wrap long lines
            meta_strings: list[str] = []
            for ms in collected_metadata:
                if wrap_text is not None:
                    ms = wrap(ms, wrap_text, subsequent_indent=" ")
                meta_strings.append(ms)
            meta_strings = flatten(meta_strings)
            return "\n".join(ms for ms in meta_strings)

        return collected_metadata

    # end of minsky91 additions
        

class JobQueue(QObject):
    """Queue of waiting, ongoing and finished jobs for one document."""

    class Item(NamedTuple):
        job: str
        image: int

    count_changed = pyqtSignal()
    selection_changed = pyqtSignal()
    job_finished = pyqtSignal(Job)
    job_discarded = pyqtSignal(Job)
    # minsky91: added a separate signal to update doc annotations 
    jobs_discarded_all = pyqtSignal(Job)
    # end of minsky91 additions
    result_used = pyqtSignal(Item)
    result_discarded = pyqtSignal(Item)

    def __init__(self):
        super().__init__()
        self._entries: deque[Job] = deque()
        self._selection: list[JobQueue.Item] = []
        self._previous_selection: JobQueue.Item | None = None
        self._memory_usage = 0  # in MB

    def add(self, kind: JobKind, params: JobParams):
        return self.add_job(Job(None, kind, params))

    def add_control(self, control: "control.ControlLayer", bounds: Bounds):
        job = Job(None, JobKind.control_layer, JobParams(bounds, f"[Control] {control.mode.text}"))
        job.control = control
        return self.add_job(job)

    def add_job(self, job: Job):
        self._entries.append(job)
        self.count_changed.emit()
        return job

    def remove(self, job: Job):
        # Diffusion/Animation jobs: kept for history, pruned according to meomry usage
        # Other jobs: removed immediately once finished
        self._entries.remove(job)
        self.count_changed.emit()

    def find(self, id: str):
        return next((j for j in self._entries if j.id == id), None)

    def count(self, state: JobState):
        return sum(1 for j in self._entries if j.state is state)

    def has_item(self, item: Item):
        job = self.find(item.job)
        return job is not None and item.image < len(job.results)

    def set_results(self, job: Job, results: ImageCollection):
        job.results = results
        if job.kind in [JobKind.diffusion, JobKind.animation]:
            self._memory_usage += results.size / (1024**2)
            self.prune(keep=job)

    def notify_started(self, job: Job):
        if job.state is not JobState.executing:
            job.state = JobState.executing
            self.count_changed.emit()

    def notify_finished(self, job: Job):
        job.state = JobState.finished
        self.job_finished.emit(job)
        self._cancel_earlier_jobs(job)
        self.count_changed.emit()

        if job.kind not in [JobKind.diffusion, JobKind.animation]:
            self.remove(job)

    def notify_cancelled(self, job: Job):
        job.state = JobState.cancelled
        self._cancel_earlier_jobs(job)
        self.count_changed.emit()

    def notify_used(self, job_id: str, index: int):
        job = ensure(self.find(job_id))
        job.in_use[index] = True
        self.result_used.emit(self.Item(job_id, index))

    def select(self, job_id: str, index: int):
        self.selection = [self.Item(job_id, index)]

    def toggle_selection(self):
        if self._selection:
            self._previous_selection = self._selection[0]
            self.selection = []
        elif self._previous_selection is not None and self.has_item(self._previous_selection):
            self.selection = [self._previous_selection]

    def _discard_job(self, job: Job):
        self._entries.remove(job)
        self._memory_usage -= job.results.size / (1024**2)
        self.job_discarded.emit(job)

    def prune(self, keep: Job):
        while self._memory_usage > settings.history_size and self._entries[0] != keep:
            self._discard_job(self._entries[0])

    def discard(self, job_id: str, index: int):
        job = ensure(self.find(job_id))
        if len(job.results) <= 1:
            self._discard_job(job)
            return
        for i in range(index, len(job.results) - 1):
            job.in_use[i] = job.in_use.get(i + 1, False)
        img = job.results.remove(index)
        self._memory_usage -= img.size / (1024**2)
        self.result_discarded.emit(self.Item(job_id, index))

    def clear(self):
        jobs_to_discard = [
            job
            for job in self._entries
            if job.kind is JobKind.diffusion and job.state is JobState.finished
        ]
        for job in jobs_to_discard:
            self._discard_job(job)
        # minsky91: added a separate signal to update doc annotations 
        # this radically shortens clear history time when the history is really large
        self.jobs_discarded_all.emit(job)
        # end of minsky91 additions

    def any_executing(self):
        return any(j.state is JobState.executing for j in self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, i):
        return self._entries[i]

    def __iter__(self):
        return iter(self._entries)

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value: list[Item]):
        if self._selection != value:
            self._selection = value
            self.selection_changed.emit()

    @property
    def memory_usage(self):
        return self._memory_usage

    def _cancel_earlier_jobs(self, job: Job):
        # Clear jobs that should have been completed before, but may not have completed
        # (still queued or executing state) due to sporadic server disconnect
        for j in self._entries:
            if j is job:
                break
            if j.state in [JobState.queued, JobState.executing]:
                j.state = JobState.cancelled


    # minsky91: add extended job metadata stats & basic workflow
    def set_stat_metadata(self, job: Job, stat_metadata: listr[str]):
        job.stat_metadata = stat_metadata

    def set_workflow(self, job: Job, workflow: listr[str]):
        job.imageless_workflow = workflow

    # end of minsky91 additions

def _move_field(src: dict[str, Any], field: str, dest: dict[str, Any]):
    if field in src:
        dest[field] = src[field]
        del src[field]
