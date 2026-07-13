import logging
import torch
import comfy.model_management as mm
import comfy.patcher_extension
import gc
from .block_swap import install_block_swap, install_te_block_swap, _free_to_meta, _has_ggml_params

logger = logging.getLogger(__name__)


def _get_diffusion_model(patcher):
    if patcher is None:
        return None
    model_obj = getattr(patcher, "model", patcher)
    diffusion = getattr(model_obj, "diffusion_model", None)
    if diffusion is not None and isinstance(diffusion, torch.nn.Module):
        return diffusion
    if isinstance(model_obj, torch.nn.Module):
        return model_obj
    inner = getattr(patcher, "model", None)
    if inner is not None and isinstance(inner, torch.nn.Module):
        return inner
    return None


def _get_cond_stage_model(clip_obj):
    """Extract the cond_stage_model from a CLIP wrapper."""
    if clip_obj is None:
        return None
    cond_stage = getattr(clip_obj, "cond_stage_model", None)
    if cond_stage is not None and isinstance(cond_stage, torch.nn.Module):
        return cond_stage
    return None


def _free_block_cleanup(swl):
    """Free swap block memory during ON_CLEANUP.
    Safetensor: _free_to_meta (release to meta, vbar handles restore).
    GGUF: to(offload_device) (quantized data to CPU, GGMLTensor preserved).
    """
    for i in range(swl.non_swap_count, swl.total_count):
        try:
            blk = swl._modules.get(str(i))
            if blk is None:
                continue
            if _has_ggml_params(blk):
                blk.to(swl.offload_device)
            else:
                _free_to_meta(blk)
            for m in blk.modules():
                for attr in ('_v', '_prefetch', '_v_signature',
                             'ggml_weight', 'ggml_weight_data'):
                    if hasattr(m, attr):
                        try:
                            delattr(m, attr)
                        except Exception:
                            pass
        except Exception:
            pass


def clear_comfyui_cache_except(exclude_patcher=None):
    """Clear all models from GPU to CPU (unpatch), except exclude_patcher.
    This frees VRAM used by TE/VAE/etc without touching the DIT model.
    """
    cf_models = mm.loaded_models()
    for pipe in cf_models:
        if exclude_patcher is not None and pipe is exclude_patcher:
            continue
        try:
            pipe.unpatch_model(device_to=torch.device("cpu"))
        except Exception:
            pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")


class UniBlockSwap:
    """Swap blocks one-at-a-time between GPU/CPU to reduce VRAM.
    Supports both safetensor and GGUF models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "num_blocks": ("INT", {
                    "default": -1, "min": -1, "max": 10000, "step": 1,
                    "tooltip": "Blocks from end to swap. -1 = all, 0 = disable",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_swap"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Swap blocks one-at-a-time between GPU/CPU to reduce VRAM."

    def apply_swap(self, model, num_blocks=-1):
        if num_blocks == 0:
            return (model,)

        patcher = model.clone()
        if hasattr(model, 'backup'):
            model.backup.clear()
        patcher.backup = {}
        clear_comfyui_cache_except(patcher)
        diffusion_model = _get_diffusion_model(patcher)
        if diffusion_model is None:
            logger.warning("UniBlockSwap: no diffusion model found")
            return (patcher,)

        compute = mm.get_torch_device()
        offload = mm.unet_offload_device()

        logger.info("UniBlockSwap: %s, compute=%s, offload=%s",
                    type(diffusion_model).__name__, compute, offload)

        mgr, cleanup, _dit_swap_names, _dit_all_swls = install_block_swap(
            diffusion_model, compute, offload,
            num_blocks=num_blocks,
        )

        if mgr is None:
            return (patcher,)

        def _is_dit_swap_key(key):
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in _dit_swap_names and i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part.lstrip("-").isdigit():
                        return True
            return False

        def _on_load(p, device_to, lowvram, force, full):
            try:
                mgr.offload_swap_blocks()
                for key in list(p.backup.keys()):
                    if _is_dit_swap_key(key):
                        p.backup.pop(key, None)
            except Exception:
                pass
            mm.soft_empty_cache()
            gc.collect()

        patcher.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_LOAD,
            "UniBlockSwap", _on_load,
        )

        # Detect if this patcher is a GGUFModelPatcher (which handles GGMLTensor weights).
        _is_gguf = hasattr(patcher, 'mmap_released')

        _orig_patch = patcher.patch_weight_to_device
        def _skip_swap_patch(key, *args, **kwargs):
            if _is_dit_swap_key(key):
                if _is_gguf:
                    # GGUF: completely skip. _load_swap manages GPU loading.
                    return
                # Safetensor: call original, delete backup.
                result = _orig_patch(key, *args, **kwargs)
                if key in patcher.backup:
                    patcher.backup.pop(key, None)
                return result
            return _orig_patch(key, *args, **kwargs)
        patcher.patch_weight_to_device = _skip_swap_patch

        # CRITICAL: _load_list filter for GGUF to prevent load() from
        # iterating over swap blocks and calling m.to(device_to) on each,
        # which would load all GGUF swap blocks to GPU at once (12GB spike).
        if _is_gguf:
            _orig_load_list = patcher._load_list
            def _filtered_load_list(*args, **kwargs):
                raw = _orig_load_list(*args, **kwargs)
                return [item for item in raw if not _is_dit_swap_key(item[-3])]
            patcher._load_list = _filtered_load_list

        def _on_dit_cleanup(p):
            try:
                for swl in _dit_all_swls:
                    _free_block_cleanup(swl)
                for key in list(p.backup.keys()):
                    if _is_dit_swap_key(key):
                        p.backup.pop(key, None)
                for _ in range(3):
                    gc.collect()
            except Exception:
                pass

        patcher.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            "UniBlockSwap", _on_dit_cleanup,
        )

        patcher.model._uniblockswap_cleanup = cleanup
        return (patcher,)


class UniBlockSwapTE:
    """Swap text encoder blocks one-at-a-time between GPU/CPU to save VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "num_blocks": ("INT", {
                    "default": -1, "min": -1, "max": 10000, "step": 1,
                    "tooltip": "Blocks from end to swap. -1 = all, 0 = disable",
                }),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "apply_swap"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Swap text encoder blocks one-at-a-time between GPU/CPU to reduce VRAM."

    def apply_swap(self, clip, num_blocks=-1):
        if num_blocks == 0:
            return (clip,)

        new_clip = clip.clone()
        cond_stage = _get_cond_stage_model(new_clip)
        if cond_stage is None:
            logger.warning("UniBlockSwapTE: no cond_stage_model found")
            return (new_clip,)

        new_clip.patcher.backup = {}
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        compute = new_clip.patcher.load_device
        offload = new_clip.patcher.offload_device

        logger.info("UniBlockSwapTE: %s, compute=%s, offload=%s",
                    type(cond_stage).__name__, compute, offload)

        mgr_list, cleanup, container_names = install_te_block_swap(
            cond_stage, compute, offload,
            num_blocks=num_blocks,
        )

        if not mgr_list:
            logger.info("UniBlockSwapTE: no block containers found in %s",
                        type(cond_stage).__name__)
            return (new_clip,)

        def _is_swap_key(key):
            for mgr in mgr_list:
                cname = getattr(mgr, 'container_name', '')
                if not cname:
                    continue
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == cname and i + 1 < len(parts):
                        next_part = parts[i + 1]
                        if next_part.lstrip("-").isdigit():
                            return True
            return False

        def _purge_swap_from_backup(p):
            if len(p.backup) == 0:
                return
            try:
                keys_to_del = [k for k in p.backup if _is_swap_key(k)]
                for k in keys_to_del:
                    p.backup.pop(k, None)
            except Exception:
                pass

        def _on_load(p, device_to, lowvram, force, full):
            _purge_swap_from_backup(p)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        new_clip.patcher.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_LOAD,
            "UniBlockSwapTE", _on_load,
        )

        _orig_patch = new_clip.patcher.patch_weight_to_device
        def _skip_swap_patch(key, *args, **kwargs):
            if _is_swap_key(key):
                return
            return _orig_patch(key, *args, **kwargs)
        new_clip.patcher.patch_weight_to_device = _skip_swap_patch

        _orig_load_list = new_clip.patcher._load_list
        def _filtered_load_list(*args, **kwargs):
            raw = _orig_load_list(*args, **kwargs)
            return [item for item in raw if not _is_swap_key(item[-3])]
        new_clip.patcher._load_list = _filtered_load_list

        new_clip.patcher.model._uniblockswap_te_cleanup = cleanup

        def _on_cleanup(p):
            try:
                for mgr in mgr_list:
                    _free_block_cleanup(mgr)
                _purge_swap_from_backup(p)
                for _ in range(3):
                    gc.collect()
            except Exception:
                pass

        new_clip.patcher.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            "UniBlockSwapTE", _on_cleanup,
        )

        return (new_clip,)


NODE_CLASS_MAPPINGS = {
    "UniBlockSwap": UniBlockSwap,
    "UniBlockSwapTE": UniBlockSwapTE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniBlockSwap": "UniBlockSwap",
    "UniBlockSwapTE": "UniBlockSwap TE",
}