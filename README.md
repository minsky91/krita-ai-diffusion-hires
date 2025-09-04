# Introducing Krita AI Hires R1.0 (preview), a high resolution version of the Krita AI Diffusion plugin

You might be interested in this version if you:  
 

- want to work with large canvas in Krita AI but are put off by slow processing speed or distortions/artefacts in the output, or   
- need to upscale your works to commercial-grade print sizes (like, 12K and up), or   
- have a beefy GPU (12-16 GB VRAM) and want to use it at its fullest with this tool, or  
- are eager to see the famous Tiled Diffusion implemented in the plugin, or  
- wish for more control over the hires generation process and its logistics.

Among new features implemented in this edition (see the Hires tab screenshot below) are: 

[**Tiled Diffusion**](https://github.com/shiimizu/ComfyUI-TiledDiffusion) and accompanying it Tiled VAE Encode & Decode components, which together allow to speed up processing 1.5 \- 2 times for 4K images, 2.2 times for 6K images, and up to 21 times, for 8K images, as compared to the standard (non-tiled) Generate / Refine option \- with no discernible loss of quality **(\*)**;   

**Hiresfix Guidance**, to customize the plugin’s built-in Hiresfix functionality;

new, **optimized image upload and download** methods, speeding up upload of input PNG images: 3 times for 4K size and 5 times for 8K size, and for receiving generated PNG images: 10 times for 4K size and 24 times for 8K (with much higher speedups for 12K and beyond) **(\*)**; 

support for receiving and saving hires images in **JPEG format**;

**saving full metadata and workflows in PNG images** (see example below) and alongside them;

generation **Progress Preview**;

and a number of various **QoL improvements** & fixes, including a [**fix**](https://github.com/minsky91/krita-ai-diffusion/wiki/5.-Hiresfix-Guidance:-a-few-examples#fixing-hiresfix-for-flux) **for the infamous Flux screen pattern** artifact for resolutions higher than the checkpoint’s native.

![Krita AI Hires tab options](https://github.com/user-attachments/assets/7c482251-b4b7-4b73-95c4-eab458f6b78a)
 
This fork is a product of my extensive 1.5 year-long acquaintance with this amazing tool and, a bit later, with its sources. I find it an unparalleled work in the whole Stable Diffusion field and I wish we had more of such tools to cover various use cases. 

As I discovered rather soon, one thing that the plugin doesn’t do well is high resolution image processing in the regular Generate/Refine workspace. At resolutions from 2K and up, the output becomes more and more plagued with distortions and artifacts, and from 4-6K, processing times grow prohibitive. For the Upscale / Refine workspace that employs a (proprietary) tiled method, things look much better in the hires department, but the Refine functionality is limited and tile seam artifacts in the output are pretty ubiquitous. Many other issues arise when using large images of increasingly higher resolutions in Krita AI (although some of them can be blamed on Comfy and its Python library components), and by sizes approaching 10x10K, it’s no longer usable. (Hence the plugin’s explicit 99 MP capping on the image pixel size, apparently.)

Long story short, after a few months’ work, I succeeded in implementing most of the ideas I had for Krita AI Hires to improve the plugin’s work with large and ultra-large images, ending up at **24K** as the current size limit. I hope you will enjoy using this version just as I do. Full description of the new features can be found on the project’s [wiki pages](https://github.com/minsky91/krita-ai-diffusion/wiki/1.-Krita-AI-Hires-tab-options). 

Note that the Hires version requires Krita AI setup with a [custom Comfy server](https://docs.interstice.cloud/comfyui-setup/). The Hires version includes all functionality of the standard v1.34 of the plugin.

**(\*)** an extensive benchmarking study that supports these claims can be found [here](https://github.com/minsky91/krita-ai-diffusion/wiki/2.-Upscaling-and-refining-from-1K-to-24K-with-Krita-AI-Hires).

# Krita AI Hires installation 

At the moment, there is no automatic installation or update routine like implemented for standard Krita AI, you’ll have to install the Hires version manually, and later on, update it with new releases manually as well. The procedure assumes that you already have a working install of the standard Krita AI plugin on your system. Here are the steps:

1. Using ComfyUI Manager, install Tiled Diffusion, Tiled VAE Encode & Decode nodes.  
2. Download the zip with **Krita AI Hires** files from the [github page](https://github.com/minsky91/krita-ai-diffusion) using the green Code drop-down button.  
3. Extract the zip’s contents to a hard disk, rename the main folder to ai\_diffusion\_hires and move it to the folder tree under the folder where the standard plugin is installed (for Windows, that would be C:\\Users\\yourusername\\AppData\\Roaming\\krita\\pykrita\\ai\_diffusion), so that its location becomes C:\\Users\\yourusername\\AppData\\Roaming\\krita\\pykrita\\ai\_diffusion\_hires.  
4. Download the zip with **ComfyUi tooling nodes Hires** files from the [github page](https://github.com/minsky91/comfyui-tooling-nodes) using the green Code drop-down button.  
5. Extract the zip’s contents to a hard disk, rename the main folder to comfyui-tooling-nodes\_hires and move it to the ComfyUI folder tree under the folder where the standard ComfyUi tooling nodes are installed, similarly to 2\. above.  
6. Create **2 simple batch files** (shell scripts in Linux) to switch between the two versions named to\_hires.bat and to\_baseline.bat and place them in the pykrita folder just above ai\_diffusion. Here are batch files from my system, to be used as a template:

***to\_hires.bat***  
`ren ai_diffusion.desktop ai_diffusion.desktop_baseline`  
`ren ai_diffusion.desktop_hires ai_diffusion.desktop`  
`ren ai_diffusion ai_diffusion_baseline`  
`ren ai_diffusion_hires ai_diffusion`  

`ren "C:\AI\StabilityMatrix\Packages\ComfyUI\custom_nodes\comfyui-tooling-nodes" "comfyui-tooling-nodes_baseline"`    
`ren "C:\AI\StabilityMatrix\Packages\ComfyUI\custom_nodes\comfyui-tooling-nodes_hires" "comfyui-tooling-nodes"`  

***to\_baseline.bat***  
`ren ai_diffusion.desktop ai_diffusion.desktop_hires`  
`ren ai_diffusion.desktop_baseline ai_diffusion.desktop`    
`ren ai_diffusion ai_diffusion_hires`   
`ren ai_diffusion_baseline ai_diffusion`

`ren "C:\AI\StabilityMatrix\Packages\ComfyUI\custom_nodes\comfyui-tooling-nodes" "comfyui-tooling-nodes_hires"`  
`ren "C:\AI\StabilityMatrix\Packages\ComfyUI\custom_nodes\comfyui-tooling-nodes_baseline" "comfyui-tooling-nodes"`

7. In the pykrita folder, make a copy of ai\_diffusion.desktop with the extension ‘desktop\_hires’ and change line 7 to ’Name=AI Image Diffusion Hires’. 
8. In the pykrita\ai_diffusion folder, check if you have the websockets subfolder and it's non-empty (contains subfolders with .py files). If it's not there or empty, *copy* the websockets subfolder tree from the standard version's folder tree (this is due to a github idiosyncrasy, can't think of other ways to fix it).  
9. Run to\_hires.bat to activate the Hires version (alternatively, to switch to the standard version when the Hires version is active, run to\_baseline.bat) and *restart both* the Comfy server and Krita. To check if the switch did what was intended, check the version indicator in the Configure Image Diffusion / Plugin tab, it should have ‘Hires’ in it. 

For instructions on removing cappings on image resolution in the plugin and Comfy server, [click here](https://github.com/minsky91/krita-ai-diffusion/wiki/6.-How-to-remove-size-cappings-on-high-resolution-images-in-ComfyUI-and-Krita-AI). That will allow you to process files of resolutions as high as 24K \- *if* your GPU hardware is up to the task, naturally.

### Example of metadata saved in a file, full verbosity mode:

`a serene landscape with forest and mountains on the horizon, a  colored illustration`  
`Negative prompt:`     
`Steps: 10`  
`Sampler: Euler a (Hyper)`    
`Schedule type: Normal`  
`CFG scale: 3.0`  
`Seed: 1105287894`    
`Model: art\dynavisionXLAllInOneStylized_releaseV0610Bakedvae.safetensors (SD XL)`  
`Denoising strength: 1.0`  
`Style Preset: Hyper with Lora`    
`Style Preset filename: style-5.json`    
`LoRas: LCM-Hyper\Hyper-SDXL-12steps-CFG-lora.safetensors: strength 1.0`   
`Rescale CFG: 0.7`  
`Canvas resolution: 8192x6144 pixels`    
`Output resolution: 8192x6144 pixels`    
`Region 1: prompt <background>`  
`Region 2: prompt <a forest hut>, resolution 3108x2696`  
`Region 3: prompt <a mountain river>, resolution 3851x2856`  
`Region 4: prompt <a forest meadow >, resolution 5207x1473`  
`Region 5: prompt <an elderly forester man is walking to his forest hut, carrying a heavy bundle of woodsticks on his back>, resolution 1656x2680`  
`Models used: xinsirtile-sdxl-1.0.safetensors (ControlNet)`  
`               4x_NMKD-Superscale-SP_178000_G.pth (Upscale or Inpaint model)`

`Generation stats:`  
`6 cached input image(s) of 0x0 pixels`  
`Preparation time: 3.91 sec.`  
`Workflow size: 0.05 MB, 59 nodes`  
`Workflow upload time: 0.14 sec.`  
`Output files total size: 53.43 MB in 1 PNG image(s) of 8192x6144 pixels`  
`Output files download time: 3.28 sec.`  
`Execution time: 164.7 sec.`  
`Total active time: 168.6 sec.`  
`Total lifetime: 353.2 sec.`  
`Batch size: 3`

`System info:`  
`os: nt`  
`ram_total: 32 GB`  
`ram_free: 23.1 GB`  
`comfyui_version: 0.3.19`  
`python_version: 3.10.11`   
`pytorch_version: 2.6.0+cu124`  
`GPU device features:`  
`name: cuda:0 NVIDIA GeForce RTX 4070 Ti SUPER`  
`vram_total: 16 GB`  
`vram_free: 14.7 GB`


# Examples of images upscaled, refined and inpainted with Krita AI Hires

3250x3900
![The snow globe infinity 95% jpgq](https://github.com/user-attachments/assets/b2fc4095-7dbb-45a5-88c8-c132c510120f)

3840x4608
![Wrong Shoes](https://github.com/user-attachments/assets/57a694d1-38df-4a43-a0d7-241512ed03a4)

4096x4864, a scaled-down version (the master file is 8420x10000, 80 MP large, can't be uploaded to github)
![Made in Europe v1 5 4096x4864](https://github.com/user-attachments/assets/360ef1c5-8583-4ffe-afd4-74652e071add)

6528x7840
![The alien artefact](https://github.com/user-attachments/assets/b44df65e-5db7-4c4c-be81-ce59130ac354)

4000x4800, a scaled-down version (the master file is 10000x12000, 114 MP large, can't be uploaded to github)
![Kiss encore](https://github.com/user-attachments/assets/fdded99c-1fe0-4a74-89f1-f80de44266e6)


a 2875x1760 zoomed-in fragment from the master file:
![Kiss encore fragment 2875x1760](https://github.com/user-attachments/assets/730ab235-b33b-4405-a972-35aa32b23915)

