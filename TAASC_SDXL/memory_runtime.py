"""Explicit SDXL phase residency: CPU cache -> text -> U-Net -> VAE -> VLM.

Weights are read from disk exactly once. Moving the SAME modules with .to()
does not construct another model or read another checkpoint. No Accelerate
offload hooks are installed in this mode.
"""
from __future__ import annotations
import gc
import json
import torch
from diffusers import StableDiffusionXLPipeline


class PhasedSDXLPipeline(StableDiffusionXLPipeline):
    @property
    def _execution_device(self):
        # Text encoders/VAE can be on CPU while the denoiser uses CUDA.
        device = getattr(self, '_taasc_device', None)
        return device if device is not None else super()._execution_device


class PhaseMemory:
    names = ('text_encoder', 'text_encoder_2', 'unet', 'vae')

    def __init__(self, pipe, device, phased=True):
        self.pipe = pipe
        self.device = torch.device(device)
        self.phased = phased
        self.phase = 'loaded_on_cpu'
        self.history = []
        pipe._taasc_device = self.device

    def record(self, label):
        row = {'phase':label, 'phased_offload':self.phased}
        if self.device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(self.device)
            scale = 1024**3
            row.update(allocated_gib=round(torch.cuda.memory_allocated(self.device)/scale,3),
                       reserved_gib=round(torch.cuda.memory_reserved(self.device)/scale,3),
                       peak_allocated_gib=round(torch.cuda.max_memory_allocated(self.device)/scale,3),
                       device_free_gib=round(free/scale,3), total_gib=round(total/scale,3))
        self.history.append(row)
        print('[memory] '+json.dumps(row), flush=True)
        return row

    def release_cache(self):
        gc.collect()
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def enter(self, phase, active=()):
        self.phase = phase
        if self.phased:
            # Evict inactive modules BEFORE loading active modules on the GPU.
            for name in self.names:
                if name not in active:
                    getattr(self.pipe,name).to('cpu')
            self.release_cache()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            for name in active:
                getattr(self.pipe,name).to(self.device)
        self.record(phase)

    def release(self, label='sdxl_released_before_vlm'):
        if not self.phased:
            label = 'resident_mode_' + label
            self.release_cache()
        self.enter(label, ())


@torch.inference_mode()
def generate_image(pipe, controller, cache, weights, args, seed):
    """Official SDXL denoising, then separately scheduled VAE decoding."""
    memory = pipe._taasc_memory
    conditioning = None
    result = None
    latents = None
    pixels = None
    original_vae_dtype = pipe.vae.dtype
    try:
        memory.enter('denoise_start', ('unet',))
        conditioning = {name:(tensor.to(memory.device) if tensor is not None else None)
                        for name,tensor in cache.compensate(weights).items()}
        result = pipe(**conditioning, height=args.height,width=args.width,
                      original_size=(args.height,args.width), target_size=(args.height,args.width),
                      num_inference_steps=args.steps,guidance_scale=args.guidance_scale,
                      guidance_rescale=args.guidance_rescale,num_images_per_prompt=1,
                      generator=torch.Generator(device='cpu').manual_seed(seed),output_type='latent')
        latents = result.images.detach().cpu()
        if not torch.isfinite(latents).all():
            raise FloatingPointError('SDXL produced NaN/Inf latents; not saving a corrupted image.')
        result = None
        conditioning = None
        controller.finish()
        # Keep only ordinary Python counters/events for reporting.
        controller.clear_tensor_state()
        memory.record('denoise_finished')
        memory.enter('vae_decode_start', ('vae',))

        # Same latent scaling and safe precision handling as native SDXL.
        # https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        if pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast:
            pipe.upcast_vae()
        decode_dtype = next(pipe.vae.post_quant_conv.parameters()).dtype
        latents = latents.to(device=memory.device,dtype=decode_dtype)
        mean = getattr(pipe.vae.config,'latents_mean',None)
        std = getattr(pipe.vae.config,'latents_std',None)
        if mean is not None and std is not None:
            mu = torch.as_tensor(mean,device=latents.device,dtype=latents.dtype)[None,:,None,None]
            sigma = torch.as_tensor(std,device=latents.device,dtype=latents.dtype)[None,:,None,None]
            latents = latents * sigma / pipe.vae.config.scaling_factor + mu
        else:
            latents = latents / pipe.vae.config.scaling_factor
        pixels = pipe.vae.decode(latents,return_dict=False)[0]
        if not torch.isfinite(pixels).all():
            raise FloatingPointError('VAE produced NaN/Inf pixels; check checkpoint/VAE precision.')
        image = pipe.image_processor.postprocess(pixels,output_type='pil')[0]
        memory.record('vae_decode_finished')
        return image
    except torch.cuda.OutOfMemoryError:
        memory.record('oom_at_'+memory.phase)
        raise
    finally:
        result = conditioning = latents = pixels = None
        controller.clear_tensor_state()
        # Move to CPU before resetting VAE precision, avoiding another GPU copy.
        if memory.phased:
            pipe.vae.to('cpu')
        pipe.vae.to(dtype=original_vae_dtype)
        memory.release()
