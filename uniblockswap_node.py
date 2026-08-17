import logging
import torch
import comfy.model_management as mm
import comfy.patcher_extension
import gc
import uuid
from .block_swap import (
    install_block_swap, install_te_block_swap, install_minimax_music_swap,
    _restore_ggml_refs, _restore_param_refs, _is_gguf_block, _is_minimax_music_te,
)

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
    Safetensor: _restore_param_refs (point params back at the original
    CPU/staged data; swap blocks have no vbar _v - they are filtered out of
    _load_list - so meta would leave them unrestorable).
    GGUF: _restore_ggml_refs (point params back at the mmap GGMLTensors, no
    anonymous RAM, no GPU round trip).
    Covers ALL blocks, resident prefix included: the prefix must be released
    too once inference ends, and swap state is reset so the next inference
    re-preloads it.
    """
    for i in range(swl.non_swap_count, swl.total_count):
        try:
            blk = swl._modules.get(str(i))
            if blk is None:
                continue
            if _is_gguf_block(blk):
                _restore_ggml_refs(blk)
            else:
                _restore_param_refs(blk)
            for m in blk.modules():
                for attr in ('ggml_weight', 'ggml_weight_data'):
                    if hasattr(m, attr):
                        try:
                            delattr(m, attr)
                        except Exception:
                            pass
        except Exception:
            pass
    if hasattr(swl, 'reset_swap_state'):
        swl.reset_swap_state()


def _ensure_lora_functions(patcher, swl):
    """Attach LoRA LowVramPatch functions to every swap block module.

    This reuses the LoRA patches ComfyUI already attached (patcher.patches) -
    no weight data is re-read and nothing is re-wrapped. ComfyUI's cast path
    applies module.weight_function (incl. vbar fault restore, ops.py post_cast),
    so LoRA stays effective every time a swap block is loaded into CUDA.
    """
    if getattr(patcher, "mmap_released", False):
        return  # GGUF: keep original dequant/cast path behavior
    import comfy.model_patcher as mp
    patches = getattr(patcher, "patches", None)
    if not patches:
        return
    full_path = None
    for path, mod in patcher.model.named_modules():
        if mod is swl:
            full_path = path
            break
    if full_path is None:
        return
    for i in range(swl.non_swap_count, swl.total_count):
        try:
            blk = swl._modules.get(str(i))
            if blk is None:
                continue
            prefix = f"{full_path}.{i}"
            for mname, m in blk.named_modules():
                base = prefix if not mname else f"{prefix}.{mname}"
                for pname in ("weight", "bias"):
                    key = f"{base}.{pname}"
                    if key not in patches:
                        continue
                    try:
                        _, set_func, convert_func = mp.get_key_weight(
                            patcher.model, key)
                    except Exception:
                        continue
                    fn = mp.LowVramPatch(key, patches, convert_func, set_func)
                    attr_name = pname + "_function"
                    cur = list(getattr(m, attr_name, None) or [])
                    if not any(getattr(f, "key", None) == key for f in cur):
                        cur.append(fn)
                        setattr(m, attr_name, cur)
        except Exception:
            continue


def clear_comfyui_cache_except(exclude_patcher=None):
    """Clear all models from GPU to CPU (unpatch), except exclude_patcher.
    This frees VRAM used by TE/VAE/etc without touching the DIT model.
    """
    cf_models = mm.loaded_models()
    for pipe in cf_models:
        if exclude_patcher is not None and pipe is exclude_patcher:
            continue
        # CRITICAL: do NOT unpatch patchers that share the same underlying
        # model object (e.g. the previous UniBlockSwap output patcher from an
        # earlier inference). GGUFModelPatcher.unpatch_model() wipes `.patches`
        # off the SHARED GGMLTensor parameters and clears `_ggml_patches`,
        # which would silently kill the LoRA on a swap reinstall (num_blocks
        # changed -> node re-runs -> clear() runs while the old patcher is
        # still in loaded_models()).
        if exclude_patcher is not None:
            try:
                if pipe.model is exclude_patcher.model:
                    continue
            except Exception:
                pass
        try:
            pipe.unpatch_model(device_to=torch.device("cpu"))
        except Exception:
            pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")


def _remount_ggml_lora(patcher):
    """Re-mount quantized GGUF LoRA patches after a swap reinstall.

    Belt-and-braces alongside the shared-model skip in
    clear_comfyui_cache_except(): if the shared GGMLTensor `.patches` were ever
    wiped (e.g. by some other unpatch path), the patch *data* still lives in
    `patcher._ggml_patches` (copied onto every clone), so re-mount it via the
    un-wrapped class method, bypassing the `_skip_swap_patch` instance wrapper
    that would otherwise swallow the call.

    Idempotent: re-mounting the same patches over themselves is a no-op in
    effect. Safe on first install too (no-op when _ggml_patches is absent, and
    harmless when present). Works for DIT and TE patchers alike.
    """
    ggml_patches = getattr(patcher, "_ggml_patches", None)
    if not ggml_patches:
        return
    raw_patch = getattr(type(patcher), "patch_weight_to_device", None)
    if raw_patch is None:
        return
    for key in list(ggml_patches.keys()):
        try:
            raw_patch(patcher, key)
        except Exception as e:
            logger.warning("UniBlockSwap: failed to re-mount LoRA on %s: %s", key, e)


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
                    "tooltip": ("前缀常驻块数: 一次性把 block 0..N-1 推送进 CUDA, "
                                "常驻到本次推理结束才释放; 其余块按需逐块懒加载。"
                                "-1 = 单块前缀常驻(最省显存); N = N 块前缀常驻; "
                                ">= 总块数 = 不 swap 全部驻留"),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_swap"
    CATEGORY = "model/loaders"
    DESCRIPTION = "前缀常驻 swap: 前 num_blocks 个 block 一次性推入 CUDA 常驻到推理结束, 其余块逐块懒加载, 降低显存。"

    # NOTE: no IS_CHANGED on purpose. apply_swap() is a one-time install step
    # (wraps the shared model object in SwappableModuleList + attaches LoRA
    # weight_functions). Forcing re-execution every run would re-wrap the
    # already-wrapped model (nested swap) and break swap + LoRA state.
    # Per-inference VRAM cleanup of TE/VAE is handled by the separate
    # UniBlockSwapCacheControl node, which re-runs every inference.
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

        # Re-entry guard: model.clone() shares the SAME underlying model object
        # (no deepcopy). If this node re-runs on an already-swapped model (e.g.
        # num_blocks changed -> input changed -> node re-executed), fully restore
        # the original structure first, otherwise install_block_swap would wrap
        # the SwappableModuleList again (nested) and break swap/LoRA state.
        prev_cleanup = getattr(diffusion_model, "_uniblockswap_cleanup", None)
        if prev_cleanup is not None:
            try:
                # Offload the OLD swap blocks first (point params back at their
                # mmap GGMLTensors) so the re-mount below lands on the mmap
                # objects, not on stale CUDA copies from an interrupted run.
                for m in diffusion_model.modules():
                    if hasattr(m, "offload_swap_blocks"):
                        m.offload_swap_blocks()
            except Exception:
                pass
            try:
                prev_cleanup()
            except Exception:
                logger.warning("UniBlockSwap: failed to restore previous swap before reinstall", exc_info=True)

        # Re-mount quantized LoRA patches (idempotent). After a num_blocks
        # change the node re-runs and clear_comfyui_cache_except() above may
        # have wiped the shared GGMLTensor `.patches` via unpatch_model(); the
        # patch data still lives in patcher._ggml_patches, so re-attach it.
        _remount_ggml_lora(patcher)

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
                for swl in _dit_all_swls:
                    _ensure_lora_functions(p, swl)
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

        # Swap blocks: never write LoRA-patched weights back into parameters
        # here. LoRA is applied at cast time via weight_function
        # (_ensure_lora_functions), otherwise it would double-apply.
        _orig_patch = patcher.patch_weight_to_device
        def _skip_swap_patch(key, *args, **kwargs):
            if _is_dit_swap_key(key):
                return
            return _orig_patch(key, *args, **kwargs)
        patcher.patch_weight_to_device = _skip_swap_patch

        # CRITICAL: filter swap blocks from _load_list so load() never
        # iterates over them (m.to(device_to) would load all swap blocks to
        # GPU at once). Blocks are loaded one-at-a-time by swap + vbar fault.
        _orig_load_list = patcher._load_list
        def _filtered_load_list(*args, **kwargs):
            raw = _orig_load_list(*args, **kwargs)
            return [item for item in raw if not _is_dit_swap_key(item[-3])]
        patcher._load_list = _filtered_load_list

        # Attach LoRA weight_functions to swap blocks (idempotent).
        for swl in _dit_all_swls:
            _ensure_lora_functions(patcher, swl)

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
                    "tooltip": ("前缀常驻块数: 一次性把 block 0..N-1 推送进 CUDA, "
                                "常驻到本次推理结束才释放; 其余块按需逐块懒加载。"
                                "-1 = 单块前缀常驻(最省显存); N = N 块前缀常驻; "
                                ">= 总块数 = 不 swap 全部驻留"),
                }),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "apply_swap"
    CATEGORY = "model/loaders"
    DESCRIPTION = "前缀常驻 swap 文本编码器 block: 前 num_blocks 个 block 常驻 CUDA 到本次推理结束, 其余块逐块懒加载。"

    # NOTE: no IS_CHANGED on purpose (see UniBlockSwap comment). Per-inference
    # VRAM cleanup of the text encoder is handled by UniBlockSwapCacheControl.
    def apply_swap(self, clip, num_blocks=-1):
        if num_blocks == 0:
            return (clip,)

        new_clip = clip.clone()
        cond_stage = _get_cond_stage_model(new_clip)
        if cond_stage is None:
            logger.warning("UniBlockSwapTE: no cond_stage_model found")
            return (new_clip,)

        # MiniMax Music3T dedicated branch: re-implement generate WITHOUT the
        # native vbar / CUDA-graph machinery (see install_minimax_music_swap in
        # block_swap.py). The AR loop is not a single forward, so the normal
        # forward-wrapping swap cannot apply. We manually move each AR layer and
        # the depth decoder into CUDA per step and offload after.
        if _is_minimax_music_te(cond_stage):
            compute = new_clip.patcher.load_device
            offload = new_clip.patcher.offload_device
            mgr_list, cleanup, container_names = install_minimax_music_swap(
                cond_stage, compute, offload, num_blocks=num_blocks)
            if not mgr_list:
                logger.info("UniBlockSwapTE: MiniMax Music3T - nothing to swap")
                return (new_clip,)
            new_clip.patcher.model._uniblockswap_te_cleanup = cleanup

            def _ensure_offloaded():
                try:
                    cleanup()
                except Exception:
                    pass

            new_clip.patcher.add_callback_with_key(
                comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
                "UniBlockSwapTE-MiniMax", lambda p: _ensure_offloaded())

            logger.info("UniBlockSwapTE: MiniMax Music3T swap installed "
                        "(bypass vbar/graph); num_blocks=%s", num_blocks)
            return (new_clip,)

        # Re-entry guard: clip.clone() shares the same underlying model object.
        # Restore any previously installed TE swap structure before reinstalling
        # to avoid nesting SwappableModuleList / double-wrapping forward.
        prev_cleanup = getattr(new_clip.patcher.model, "_uniblockswap_te_cleanup", None)
        if prev_cleanup is not None:
            try:
                for m in cond_stage.modules():
                    if hasattr(m, "offload_swap_blocks"):
                        m.offload_swap_blocks()
            except Exception:
                pass
            try:
                prev_cleanup()
            except Exception:
                logger.warning("UniBlockSwapTE: failed to restore previous swap before reinstall", exc_info=True)

        # Re-mount quantized LoRA patches after a TE reinstall (idempotent);
        # see _remount_ggml_lora docstring. No-op for non-GGUF TE patchers.
        _remount_ggml_lora(new_clip.patcher)

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
            for mgr in mgr_list:
                _ensure_lora_functions(p, mgr)
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

        for mgr in mgr_list:
            _ensure_lora_functions(new_clip.patcher, mgr)

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


class UniBlockSwapCacheControl:
    """Per-inference VRAM cleanup for the other models (TE/VAE), passthrough.

    UniBlockSwap / UniBlockSwapTE no longer force re-execution via IS_CHANGED
    (re-running would re-wrap the shared model object and break swap/LoRA).
    Instead this node re-runs every inference but only performs the cheap
    clear_comfyui_cache_except() side effect, leaving swap installation intact.

    Usage: UniBlockSwap -> UniBlockSwapCacheControl -> KSampler
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "clear_cache"
    CATEGORY = "model/loaders"
    DESCRIPTION = ("每次推理清除除传入 model 外的其他模型(TE/VAE 等)的显存,"
                   "并原样透传 model。放在 UniBlockSwap 之后、KSampler 之前。"
                   "替代原先用 IS_CHANGED 强制 UniBlockSwap 重跑的做法,"
                   "避免重复安装 swap 导致 LoRA 失效。")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-run every inference, but this node only clears cache - it does not
        # reinstall swap, so it has no destructive side effects.
        return uuid.uuid4().hex

    def clear_cache(self, model):
        clear_comfyui_cache_except(model)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "UniBlockSwap": UniBlockSwap,
    "UniBlockSwapTE": UniBlockSwapTE,
    "UniBlockSwapCacheControl": UniBlockSwapCacheControl,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniBlockSwap": "UniBlockSwap",
    "UniBlockSwapTE": "UniBlockSwap TE",
    "UniBlockSwapCacheControl": "UniBlockSwap Cache Control",
}
