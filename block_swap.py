"""
UniBlockSwap - Universal block swap for ComfyUI (fixed-resident prefix).

num_blocks is the inference LOOP mechanism, not a loading mechanism:
  - num_blocks blocks form a PERMANENT-RESIDENT PREFIX: blocks 0..num_blocks-1
    are pushed into CUDA once at install time and stay resident for the whole
    inference (no transfers while the loop runs over them). The remaining
    blocks (num_blocks..total-1) stay on the ORIGINAL lazy path: the GGUF
    plugin dequantizes + transfers each layer inside forward() (overlapped
    with compute; ComfyUI vbar manages VRAM), no swap ops run on the tail.
    Everything (prefix included) is released when inference ends (ON_CLEANUP /
    offload_swap_blocks).
  -1 / <=0 -> prefix of 1 block
  1 <= N < total -> prefix of N blocks (N blocks resident up front)
  N >= total  -> no swap (entire model resident)

Load/unload per block type (the GGUF plugin's loading mechanism is UNCHANGED):
  Safetensor: swap blocks are filtered out of patcher._load_list, so they
  NEVER get a vbar _v alloc - ComfyUI's vbar cast path is not involved at
  all. They run on ComfyUI's PLAIN cast path (resolve_cast_module_with_vbar
  falls through when hasattr(s, "_v") is False), which transfers weights from
  CPU/mmap inside forward() every step. Prefix blocks are therefore preloaded
  the same way as GGUF: module.to(compute_device) moves the whole block onto
  CUDA up front, so the plain cast path sees weight.device == device and
  skips the transfer - zero per-layer stalls over the resident prefix. A
  reference to the original (mmap-backed) param data is kept for release.
  On unload the params are pointed back at the original data (pointer
  assignment, no anonymous RAM), same idea as the GGUF branch.
  GGUF: for every resident block (prefix or current tail) the whole block's
  quantized weights are moved onto CUDA up front (module.to(compute_device),
  GGMLTensor metadata kept), so the blocks are GPU-resident when the loop
  reaches them; the GGUF plugin's own forward-time dequantization then runs
  against resident data (no per-layer mmap->CUDA stall). On release the params
  are pointed back at the original mmap GGMLTensors (pointer assignment, no
  anonymous RAM).
"""

import gc
import logging
import torch
import torch.nn as nn
import comfy

logger = logging.getLogger(__name__)

CONTAINER_NAMES = (
    "blocks", "transformer_blocks", "double_blocks", "single_blocks",
    "input_blocks", "output_blocks", "middle_block", "layers",
    "double_stream_layers", "single_stream_layers",
    "block",
)


def find_blocks(model):
    for name in CONTAINER_NAMES:
        c = getattr(model, name, None)
        if isinstance(c, (nn.ModuleList, list)) and len(c) > 0 and hasattr(c[0], "forward"):
            return name, c
    return None, None


def _is_ggml_tensor(t):
    """True if `t` carries GGMLTensor metadata (quantized or torch-compatible
    GGUF weight). Check both the object itself and its .data - nn.Parameter()
    on a Tensor subclass drops __init__ attrs, and torch.Tensor.data returns
    self for GGMLTensor (detach() -> self), so both paths are covered."""
    if t is None:
        return False
    if getattr(t, "tensor_type", None) is not None:
        return True
    try:
        d = t.data
        if d is not t and getattr(d, "tensor_type", None) is not None:
            return True
    except Exception:
        pass
    return False


def _has_ggml_params(module):
    """Check if module has GGMLTensor parameters (quantized GGUF weights)."""
    for p in module.parameters():
        if _is_ggml_tensor(p):
            return True
    return False


def _backup_ggml_refs(module):
    """Preserve the ORIGINAL mmap-backed GGMLTensor objects for every GGML
    parameter in `module`.

    Why a full-reference backup (not just .data): tensor_type / tensor_shape /
    patches live on the GGMLTensor *object*, and a .to(...) round trip creates
    a fresh tensor that loses the mmap mapping. We must keep the original object
    alive so we can point the parameter back at it later.
    """
    if getattr(module, "_ggml_mmap_backup", None) is not None:
        return
    backup = {}
    for name, param in module.named_parameters(recurse=True):
        t = param if _is_ggml_tensor(param) else param.data
        if _is_ggml_tensor(t):             # a GGMLTensor
            backup[name] = t               # keep the object alive, mmap intact
    module._ggml_mmap_backup = backup


def _restore_ggml_refs(module):
    """Point params back at the original mmap GGMLTensors and drop any GPU
    copies. This is a *pointer assignment* (p.data = orig), so NO anonymous
    heap allocation happens -- unlike module.to(offload_device), which would
    reallocate the dequantized weights as non-reclaimable RAM.

    If a block was never GPU-loaded (no backup), fall back to .to(cpu) which is
    a no-op for an already-mmap'd CPU tensor.
    """
    backup = getattr(module, "_ggml_mmap_backup", None)
    if not backup:
        module.to(module.offload_device if hasattr(module, "offload_device") else "cpu")
        return
    params = dict(module.named_parameters(recurse=True))
    for name, orig in backup.items():
        p = params.get(name)
        if p is not None:
            p.data = orig
    # any plain (non-GGML) params that _load_block's module.to() moved to CUDA
    # must come back to the offload device too
    offload = module.offload_device if hasattr(module, "offload_device") else "cpu"
    offload_t = torch.device(offload)
    for p in module.parameters(recurse=True):
        if p.device.type != offload_t.type:
            p.data = p.data.to(offload)
    # free the GPU copy of the now-unreferenced tensor
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _is_gguf_block(module):
    """A block is GGUF if we captured its original mmap GGMLTensor refs at
    install time. This flag - not _has_ggml_params - is the source of truth:
    a block whose params were .to(cuda)'d still counts as GGUF so load/offload
    take the pointer-assignment path instead of the vbar/meta path.
    """
    return bool(getattr(module, "_ggml_mmap_backup", None))


def _has_meta_params(module):
    """True if the block's params are still meta (model weights not staged
    yet - happens at install time, before ComfyUI loads the safetensor).
    Nothing can be transferred in that state, so preload must be skipped."""
    for p in module.parameters(recurse=True):
        if getattr(p, "device", None) is not None and p.device.type == "meta":
            return True
    return False


def _free_to_meta(module):
    """Free param data to meta tensor - NO CPU copy created.
    The module structure is preserved. next load() restores from backup."""
    for param in module.parameters(recurse=False):
        param.data = torch.empty(0, device='meta')


def _backup_param_refs(module):
    """Preserve the ORIGINAL (CPU/staged) param .data objects for a safetensor
    block before it is moved to CUDA, so unload can point params back without
    reallocation. Generic counterpart of _backup_ggml_refs for non-GGML params.
    """
    if getattr(module, "_param_ref_backup", None) is not None:
        return
    backup = {}
    for name, param in module.named_parameters(recurse=True):
        backup[name] = param.data
    module._param_ref_backup = backup


def _restore_param_refs(module):
    """Point params back at the original (CPU/staged) data and drop any GPU
    copies. Pointer assignment (p.data = orig) - NO anonymous heap allocation.
    No-op if the block was never backed up (still CPU)."""
    backup = getattr(module, "_param_ref_backup", None)
    if not backup:
        module.to(module.offload_device if hasattr(module, "offload_device") else "cpu")
        return
    params = dict(module.named_parameters(recurse=True))
    for name, orig in backup.items():
        p = params.get(name)
        if p is not None:
            p.data = orig
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


class SwappableModuleList(nn.ModuleList):
    """ModuleList with a fixed-resident prefix; tail stays on the lazy path.

    prefix_count (= num_blocks) blocks are pushed into CUDA once and stay
    resident for the whole inference (no sliding, no transfers while the loop
    runs over them). The tail blocks (prefix_count..total-1) are deliberately
    left untouched: eager per-step tail swapping re-transfers the whole tail on
    every loop pass (the loop restarts at block 0) with blocking transfers and
    empty_cache churn - slower than the plugin's lazy per-layer path. The
    plugin dequantizes + transfers tail layers inside forward(), overlapped
    with compute. Everything (prefix included) is released when inference ends
    via offload_swap_blocks() / ON_CLEANUP.
    """

    def __init__(self, modules, compute_device, offload_device, window_size=1):
        super().__init__(modules)
        self.compute_device = compute_device
        self.offload_device = offload_device
        self.window_size = max(1, window_size)
        self.total_count = len(modules)
        # num_blocks -> how many leading blocks are kept resident in CUDA for
        # the whole inference.
        self.prefix_count = min(self.window_size, self.total_count)
        # Kept at 0 for compatibility with the node file (cleanup / LoRA
        # traversal cover ALL blocks, prefix included).
        self.non_swap_count = 0
        self._prefix_loaded = False  # prefix blocks resident in CUDA?
        self.container_name = ''

    def _offload_block(self, idx):
        try:
            blk = self._modules[str(idx)]
            if _is_gguf_block(blk):
                # GGUF: restore the original mmap-backed GGMLTensor by pointer
                # assignment (no anon RAM, no GPU round trip). If the block was
                # never loaded the backup is empty and the helper safely falls
                # back to a no-op .to(cpu).
                _restore_ggml_refs(blk)
            else:
                # Safetensor: point params back at the original (CPU/staged)
                # data. Swap blocks never have a vbar _v (they are filtered out
                # of _load_list), so meta would leave them unrestorable - the
                # pointer assignment keeps the source data alive instead.
                _restore_param_refs(blk)
            # NOTE: no vbar state (_v/_prefetch/_v_signature) exists for swap
            # blocks - they were filtered out of _load_list, so they never got
            # a vbar alloc and run on the plain cast path instead.
        except Exception:
            pass

    def _preload_safetensor_block(self, idx, blk):
        """Explicitly preload a safetensor block into CUDA so the block is
        GPU-resident before the loop reaches it - same effect as the GGUF
        branch.

        NOTE: safetensor swap blocks NEVER get a vbar _v - they are filtered
        out of patcher._load_list by the node, so _v = vbar.alloc() inside
        patcher.load() never runs for them. They execute on ComfyUI's PLAIN
        cast path (resolve_cast_module_with_vbar falls through when
        hasattr(s, "_v") is False), which re-transfers weights from CPU/mmap
        inside forward() on every step. A bare blk.to(compute_device) is
        therefore exactly right here: once weight.device == device, the plain
        cast path skips the transfer and the block stays GPU-resident.

        Blocks whose params are still meta (model weights not staged yet, e.g.
        at install time) are skipped - the lazy cast path takes over and the
        next preload pass (after patcher.load() materializes the weights)
        actually transfers.
        """
        try:
            if _has_meta_params(blk):
                logger.info("UniBlockSwap: load block %d (safetensor, weights not staged - lazy)",
                            idx)
                return
            _backup_param_refs(blk)
            blk.to(self.compute_device)
            logger.info("UniBlockSwap: load block %d (safetensor, resident)", idx)
        except Exception as e:
            logger.warning("UniBlockSwap: safetensor preload block %d failed (%s) - "
                           "falling back to lazy cast", idx, e)

    def _load_block(self, idx):
        blk = self._modules[str(idx)]
        if _is_gguf_block(blk):
            # GGUF: transfer the WHOLE block onto CUDA UP FRONT, keeping every
            # GGMLTensor's metadata (tensor_type/tensor_shape/patches) intact.
            # We do NOT dequantize here - the GGUF plugin dequantizes inside
            # forward() (its shape logic must not be bypassed). Moving the
            # quantized weights ahead of the loop means the plugin's lazy
            # dequantization runs against CUDA-resident data: the per-layer
            # mmap->CUDA transfer inside the inference loop disappears.
            _backup_ggml_refs(blk)
            blk.to(self.compute_device)
            logger.info("UniBlockSwap: load block %d (GGUF, resident)", idx)
        else:
            # Safetensor: preload the whole block with a plain .to(cuda) so
            # the prefix is CUDA-resident like the GGUF branch (the plain cast
            # path then sees weight.device == device and skips the transfer).
            self._preload_safetensor_block(idx, blk)

    def load_prefix(self):
        """Push the permanent-resident prefix blocks (0..prefix_count-1) into
        CUDA in one go. Idempotent. Called at install time and again lazily on
        first access (e.g. after an ON_LOAD offload wiped the prefix)."""
        if self.prefix_count <= 0 or self._prefix_loaded:
            return
        logger.info("UniBlockSwap: preload prefix [0,%d) into CUDA (num_blocks=%d)",
                    self.prefix_count, self.window_size)
        for j in range(self.prefix_count):
            self._load_block(j)
        self._prefix_loaded = True

    def _ensure_window(self, idx):
        """Ensure the resident prefix is in CUDA.

        Prefix blocks (idx < prefix_count) are loaded once and stay resident
        for the whole inference - zero transfers during the loop. Tail blocks
        (idx >= prefix_count) are deliberately NOT touched: eagerly swapping
        them re-transfers the whole tail on every step (the loop restarts at
        block 0, so the previous step's last tail block must be released and
        re-loaded), with blocking transfers + empty_cache churn - slower than
        the plugin's lazy per-layer path. The plugin handles them inside
        forward() (dequant + transfer, overlapped with compute).
        """
        if idx < self.prefix_count:
            if not self._prefix_loaded:
                self.load_prefix()

    def offload_swap_blocks(self):
        """Release ALL blocks (prefix included) and reset swap state. Called
        when inference ends (ON_CLEANUP / TE forward finally / model unload).
        Tail blocks lazily dequantized by the plugin are pointed back at mmap
        too, so VRAM returns to baseline."""
        for i in range(self.total_count):
            self._offload_block(i)
        self.reset_swap_state()

    def reset_swap_state(self):
        """Forget residency state so the next inference re-preloads the prefix.
        Blocks are already released by the caller; this only resets flags."""
        self._prefix_loaded = False

    def _apply(self, fn, recurse=True):
        """Do NOT move any block with model.to(...).

        All blocks are swap-managed: safetensor blocks are meta (no-op) and
        GGUF blocks must stay mmap-backed on CPU (a .to(gpu) here would force
        a full dequantization VRAM spike). nn.ModuleList._apply would apply fn
        to every child - we skip that entirely.
        """
        return self

    def __getattr__(self, name):
        try:
            idx = int(name)
            if 0 <= idx < self.total_count:
                return self.__getitem__(idx)
        except (ValueError, TypeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, idx):
        # Support slicing: blocks[start:end]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_count)
            return [self[i] for i in range(start, stop, step)]

        if 0 <= idx < self.total_count:
            self._ensure_window(idx)

        return super().__getitem__(idx)

    def __iter__(self):
        for idx in range(self.total_count):
            yield self.__getitem__(idx)


def install_block_swap(diffusion_model, compute_device, offload_device,
                       num_blocks=-1):
    all_containers = []
    for name in CONTAINER_NAMES:
        c = getattr(diffusion_model, name, None)
        if isinstance(c, (nn.ModuleList, list)) and len(c) > 0 and hasattr(c[0], "forward"):
            all_containers.append((name, c))

    if not all_containers:
        return None, lambda: None, set(), []

    first_swl = None
    all_names = set()

    for name, orig in all_containers:
        total = len(orig)
        # num_blocks = number of leading blocks kept resident in CUDA for the
        # whole inference (prefix). -1 / <=0 -> 1; 1..total-1 -> N; >= total ->
        # no swap.
        win = num_blocks if num_blocks > 0 else 1
        win = max(1, min(win, total))
        if num_blocks > 0 and win >= total:
            logger.info("UniBlockSwap: '%s' = %d blocks, NO swap (num_blocks=%d >= total)",
                         name, total, num_blocks)
            continue

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            window_size=win,
        )
        swl.container_name = name
        setattr(diffusion_model, name, swl)
        all_names.add(name)
        if first_swl is None:
            first_swl = swl

        # Every block participates: the first `win` blocks form the resident
        # prefix (pushed into CUDA once, kept for the whole inference); the
        # rest stay on the plugin's original lazy path. GGUF blocks keep their
        # mmap refs (dequantized on demand by the GGUF plugin); safetensor
        # blocks are restored by vbar on access.
        n_gguf = 0
        for i in range(total):
            if _has_ggml_params(swl._modules[str(i)]):
                _backup_ggml_refs(swl._modules[str(i)])
                n_gguf += 1
        logger.info("UniBlockSwap: '%s' GGUF blocks: %d/%d", name, n_gguf, total)

        logger.info("UniBlockSwap: '%s' = %d blocks, prefix resident = %d, tail lazy (num_blocks=%d)",
                     name, total, swl.prefix_count, num_blocks)

        # Push the resident prefix into CUDA right away: blocks 0..win-1 are
        # GPU-resident before inference starts and stay there until it ends.
        swl.load_prefix()

    if first_swl is None:
        # num_blocks >= total for every container -> no swap at all.
        return None, lambda: None, set(), []

    orig_fwd = diffusion_model.forward

    def wrapped(*args, **kwargs):
        try:
            return orig_fwd(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize(compute_device)
                gc.collect()
                torch.cuda.empty_cache()

    diffusion_model.forward = wrapped

    def cleanup():
        diffusion_model.forward = orig_fwd
        for name, orig in all_containers:
            setattr(diffusion_model, name, orig)

    all_swls = []
    for name in CONTAINER_NAMES:
        c = getattr(diffusion_model, name, None)
        if hasattr(c, 'offload_swap_blocks'):
            all_swls.append(c)

    return first_swl, cleanup, all_names, all_swls


def install_minimax_music_swap(cond_stage_model, compute_device, offload_device,
                                num_blocks=-1):
    """Install a dedicated MiniMax Music3T block-swap that re-implements
    `MiniMaxMusic3AR.generate` WITHOUT ComfyUI's native vbar / CUDA-graph
    machinery.

    Why a re-implemented generate instead of forward-wrapping:
      - Native swap for other TEs wraps `forward` and extracts patches as weights
        to control per-block CUDA residency. That works because those TEs run a
        single `self.model(...)` forward.
      - MiniMax Music3T is autoregressive: `generate` loops frame-by-frame and
        calls `prefetch_queue_pop(enable_graph=True)` to capture the audio decoder
        inside a CUDA graph. Wrapping forward (and the per-forward offload +
        synchronize) collides with graph capture -> cudaErrorStreamCaptureInvalidated.
      - So we do NOT wrap forward and do NOT touch the native vbar path. We
        monkey-patch `generate` with a version that manually moves each AR layer
        (and the depth decoder) into CUDA right before use and offloads it right
        after, respecting a fixed-resident prefix of `num_blocks` layers.

    The re-implemented generate mirrors the original logic (seed derivation,
    CFG, sampling, KV cache, fixed_kv, freqs_cis, norm) but substitutes plain
    `blk.to(compute_device)` / `_restore_*_refs` for the prefetch queue.

    Returns (mgr_list, cleanup, container_names). mgr_list holds a small
    bookkeeping object (not a SwappableModuleList) used by cleanup.
    """
    model = getattr(cond_stage_model, "model", None)
    if model is None or not hasattr(model, "layers") or not hasattr(model, "audio_decoder"):
        logger.info("UniBlockSwapTE: MiniMax Music3T - nothing to swap (no layers/decoder)")
        return [], lambda: None, set()

    layers = model.layers
    total = len(layers)
    if total == 0:
        return [], lambda: None, set()

    # num_blocks -> leading layers kept resident in CUDA for the whole generate.
    win = num_blocks if num_blocks > 0 else 1
    win = max(1, min(win, total))
    if num_blocks > 0 and win >= total:
        logger.info("UniBlockSwapTE: MiniMax Music3T = %d layers, NO swap (num_blocks=%d >= total)",
                     total, num_blocks)
        return [], lambda: None, set()

    logger.info("UniBlockSwapTE: MiniMax Music3T = %d AR layers, resident prefix = %d, "
                "tail swapped per-layer/per-frame (num_blocks=%d); vbar/graph path bypassed",
                total, win, num_blocks)

    # --- helper closures: bring a block onto CUDA, / release it back to CPU ---
    # Layer swap model (no vbar, no graph):
    #   * non-layer modules (decoder, extra_embedding, KV cache, norm, embed,
    #     lm_head) are RESIDENT on CUDA for the whole generate.
    #   * AR layers: each frame, a layer is H2D-copied from the CPU-resident
    #     model (the single source of truth in host memory) onto CUDA, run, then
    #     RELEASED back to CPU. We do NOT keep a CPU<->CUDA pointer backup; the
    #     host copy is the truth and the next layer simply re-copies. This keeps
    #     peak VRAM = resident parts + sliding window of layers, at the cost of
    #     one H2D copy per layer per frame (the price of low-VRAM swap).
    def _load_block(blk):
        """Copy a block's weights from host memory onto CUDA (H2D)."""
        try:
            if _is_gguf_block(blk):
                _backup_ggml_refs(blk)  # mmap-backed: keep metadata, move onto CUDA
                blk.to(compute_device)
            else:
                if not _has_meta_params(blk):
                    blk.to(compute_device)
        except Exception as e:
            logger.warning("UniBlockSwapTE: MiniMax load block failed (%s)", e)

    def _offload_block(blk):
        """Release a block's CUDA weights back to host memory (frees VRAM).
        For safetensor the host copy is the truth, so a plain .to(offload) is
        enough; for GGUF we must restore the mmap-backed pointers."""
        try:
            if _is_gguf_block(blk):
                _restore_ggml_refs(blk)
            else:
                blk.to(offload_device)
        except Exception:
            pass

    # Resident prefix: blocks 0..win-1 are pushed onto CUDA ONCE and stay for
    # the whole generate (optional optimization: win layers never re-copy).
    # Tail layers (>= win) are NOT loaded here - they are H2D-copied inside the
    # frame loop and released right after, keeping VRAM low.
    # GGUF prefix blocks need their refs backed up; safetensor prefix blocks are
    # plain .to(cuda) (released by pointer/restore on cleanup).
    prefix = []
    for i in range(win):
        blk = layers[i]
        if _is_gguf_block(blk):
            _backup_ggml_refs(blk)
        if not _has_meta_params(blk):
            blk.to(compute_device)
        prefix.append(blk)

    # Non-layer modules: RESIDENT on CUDA for the whole generate (per user spec).
    decoder = model.audio_decoder
    audio_extra_embedding = model.audio_extra_embedding
    _load_block(decoder)
    _load_block(audio_extra_embedding)

    original_generate = cond_stage_model.generate

    mgr = {
        "cond_stage_model": cond_stage_model,
        "model": model,
        "layers": layers,
        "total": total,
        "prefix": prefix,
        "win": win,
        "decoder": decoder,
        "audio_extra_embedding": audio_extra_embedding,
        "compute_device": compute_device,
        "offload_device": offload_device,
        "_load_block": _load_block,
        "_offload_block": _offload_block,
        "_is_gguf_block": _is_gguf_block,
    }

    # --- re-implemented generate (no vbar / no CUDA graph) ---
    from comfy.ldm.minimax_music.ar import (
        derive_seed, sample_topk, CFG_SCALE, CFG_TOP_K, MAX_AUDIO_FRAMES,
        MAX_PROMPT_TOKENS, AUDIO_CODE_OFFSET, C0_VOCAB_SIZE,
    )
    from comfy.ldm.minimax_music.prompt import SPECIAL_TOKEN_IDS
    from comfy.text_encoders.llama import FixedKV  # module-level class

    def _is_fixed_kv(kv):
        return hasattr(kv, "prepare") and hasattr(kv, "advance")

    def _load_layer(i):
        # Prefix blocks (i < win) are RESIDENT: returned as-is, zero transfer.
        # Tail blocks (>= win) are H2D-copied from host memory onto CUDA now,
        # run, and released by _offload_layer right after this frame. The module
        # object itself is layers[i]; _load_block only moves its weights, so we
        # return layers[i] (NOT _load_block's None return).
        if i < win:
            return prefix[i]
        _load_block(layers[i])
        return layers[i]

    def _offload_layer(i, blk):
        # Only tail blocks are released back to host (frees VRAM). Prefix blocks
        # stay resident for the whole generate.
        if i < win:
            return
        _offload_block(blk)

    def _ar_forward(embeds, past_key_values, dtype):
        """Manual re-implementation of Llama2_.forward without prefetch queue /
        CUDA graph.

        Layer swap model (no vbar):
          * Prefix blocks (0..win-1) are resident on CUDA (loaded once at
            install, never re-copied during the loop).
          * Tail blocks (>= win) are H2D-copied from host memory onto CUDA for
            THIS layer, run, then released back to host immediately - so peak
            VRAM only holds the resident prefix plus the layer(s) currently in
            flight. This is the low-VRAM behavior, paid for with one H2D copy
            per tail layer per frame.
          * Non-layer modules (decoder, extra_embedding, KV cache, norm, embed,
            lm_head) are resident on CUDA for the whole generate.

        Returns (x, next_key_values) mirroring original forward (output[0] = last
        hidden, output[2] = next_key_values).
        """
        x = embeds
        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first = past_key_values[0]
            past_len = first.index if _is_fixed_kv(first) else first[2]

        import torch as _torch
        position_ids = _torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)
        freqs_cis = model.compute_freqs_cis(position_ids, x.device)

        mask = None
        if seq_len > 1:
            causal_mask = _torch.empty(past_len + seq_len, past_len + seq_len,
                                       dtype=x.dtype, device=x.device).fill_(
                _torch.finfo(x.dtype).min / 4).triu_(1)
            mask = causal_mask

        from comfy.ldm.modules.attention import optimized_attention_for_device
        optimized_attention = optimized_attention_for_device(
            x.device, mask=mask is not None, small_input=True)

        fixed_kv = past_key_values is not None and len(past_key_values) > 0 and \
            _is_fixed_kv(past_key_values[0])

        next_key_values = list(past_key_values) if past_key_values is not None else []
        loaded = []
        for i in range(total):
            blk = _load_layer(i)
            loaded.append((i, blk))
            past_kv = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None
            if fixed_kv:
                past_kv.prepare(seq_len)
            x, current_kv = blk(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
                past_key_value=past_kv,
            )
            if next_key_values:
                next_key_values[i] = current_kv
            if fixed_kv:
                next_key_values[i].advance(seq_len)
            # release tail layer back to host immediately (prefix stays)
            _offload_layer(i, blk)

        if model.norm is not None:
            x = model.norm(x)
        return x, next_key_values

    def swap_generate(self, input_ids, seed, max_audio_frames, device,
                      cfg_scale=CFG_SCALE, top_k=CFG_TOP_K):
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(f"MiniMax Music3 prompt has {prompt_tokens} tokens; maximum is {MAX_PROMPT_TOKENS}")

        import torch as _torch
        input_ids = input_ids.to(device)
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = _torch.bfloat16
        else:
            execution_dtype = _torch.float32
        unconditioned = input_ids.clone()
        unconditioned[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
        text_ids = _torch.cat((input_ids, unconditioned), dim=0)
        if model.pruned_embedding:
            text_embeds = model.embed_tokens_prefill(text_ids, out_dtype=execution_dtype)
        else:
            text_embeds = model.embed_tokens(text_ids, out_dtype=execution_dtype)
        decode_limit = min(int(max_audio_frames), MAX_AUDIO_FRAMES)
        past = model.init_kv_cache(2, prompt_tokens + decode_limit + 1, device, execution_dtype)

        # prefill: full sequence through AR layers (no graph)
        out = _ar_forward(text_embeds, past, execution_dtype)
        last_hidden = out[0][:, -1]
        past = out[1]

        generator = _torch.Generator(device=device).manual_seed(derive_seed(seed, "ar"))
        depth_io = {
            "hidden": _torch.empty_like(last_hidden),
            "c0": _torch.empty((last_hidden.shape[0],), dtype=_torch.long, device=device),
            "c0_embed": _torch.empty_like(last_hidden),
            "codes": _torch.empty((last_hidden.shape[0], self.num_codebooks), dtype=_torch.long, device=device),
            "depth_hidden": _torch.empty((1, last_hidden.shape[-1] * (self.num_codebooks - 1)),
                                        dtype=execution_dtype, device=device),
        }
        decoder._comfy_cross_step_state = depth_io
        comfy.model_management._register_cross_step(decoder)

        hidden_frames = []
        pending_code = None
        stop_token = None
        pending_event = None
        pending_hidden = None
        progress = comfy.utils.ProgressBar(decode_limit)
        cuda_device = _torch.device(device).type == "cuda"
        vocab_mask = None
        if not model.pruned_lm_head:
            vocab_mask = _torch.ones(model.vocab_size, dtype=_torch.bool, device=device)
            vocab_mask[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + C0_VOCAB_SIZE] = False
            vocab_mask[SPECIAL_TOKEN_IDS["<|audio_end|>"]] = False

        # depth core: decoder + extra embedding are RESIDENT on CUDA (loaded
        # once before the loop), so no per-frame transfer here. No graph.
        def depth_core():
            codes, depth_hidden = self._depth_codes(
                depth_io["hidden"], depth_io["c0"], depth_io["c0_embed"],
                generator, execution_dtype, cfg_scale, top_k)
            depth_io["codes"].copy_(codes)
            depth_io["depth_hidden"].copy_(depth_hidden)

        for frame_index in comfy.utils.model_trange(decode_limit + 1, desc="AR sampling"):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if pending_code is not None:
                if pending_event is not None:
                    pending_event.synchronize()
                if int(pending_code.item()) == stop_token:
                    pending_hidden = None
                    break
                if pending_hidden is not None:
                    hidden_frames.append(pending_hidden)
                    progress.update_absolute(len(hidden_frames))
                    if len(hidden_frames) >= decode_limit:
                        break

            c0, code_or_stop, stop_token = self._sample_c0(last_hidden, cfg_scale, top_k, generator, vocab_mask)
            if pending_code is None:
                pending_code = _torch.empty_like(code_or_stop, device="cpu", pin_memory=cuda_device)
                if cuda_device:
                    pending_event = _torch.cuda.Event()
            pending_code.copy_(code_or_stop, non_blocking=cuda_device)
            if pending_event is not None:
                pending_event.record()

            c0 = c0.repeat(2)
            c0_embed = self._embed_c0(c0, execution_dtype)
            depth_io["hidden"].copy_(last_hidden)
            depth_io["c0"].copy_(c0)
            depth_io["c0_embed"].copy_(c0_embed)

            depth_core()

            feedback_codes = depth_io["codes"]
            depth_hidden = depth_io["depth_hidden"]
            frame_hidden = _torch.cat((last_hidden[:1].detach(), depth_hidden), dim=-1)
            if frame_index > 0:
                pending_hidden = frame_hidden[0].clone()

            feedback = self._embed_audio_frame(feedback_codes, execution_dtype)
            out = _ar_forward(feedback, past, execution_dtype)
            last_hidden = out[0][:, -1]
            past = out[1]

        if pending_hidden is not None and len(hidden_frames) < decode_limit:
            if pending_event is not None:
                pending_event.synchronize()
            if int(pending_code.item()) != stop_token:
                hidden_frames.append(pending_hidden)

        if not hidden_frames:
            raise ValueError("MiniMax Music3 generated zero audio frames")
        return _torch.stack(hidden_frames).to(device="cpu")

    cond_stage_model.generate = swap_generate.__get__(cond_stage_model, type(cond_stage_model))

    def cleanup():
        # Restore original generate and release all GPU-resident blocks.
        try:
            cond_stage_model.generate = original_generate
        except Exception:
            pass
        # Release every layer (resident prefix + any tail left on GPU from an
        # interrupted frame) back to host memory. Decoder + extra_embedding were
        # loaded resident before the loop, so they are released here too.
        for i in range(total):
            _offload_block(layers[i])
        _offload_block(decoder)
        _offload_block(audio_extra_embedding)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    container_names = {"layers"}
    return [mgr], cleanup, container_names


def _is_minimax_music_te(cond_stage_model):
    """Detect the MiniMax Music3T text encoder.

    MiniMax Music3T (MiniMaxMusic3TEModel / MiniMaxMusic3AR) wraps an
    autoregressive Qwen3-style transformer (self.model) plus an RVQ depth
    decoder (self.model.audio_decoder). Its AR sampling loop runs the audio
    decoder inside a CUDA graph capture
    (comfy/model_prefetch.prefetch_queue_pop(..., enable_graph=True)), which
    forbids non-pinned CPU->CUDA copies and stream synchronization during
    capture. Block swap's per-forward offload + synchronize is therefore
    incompatible: it would break the capture and abort the process.

    Returns True only for this architecture. Gemma4Transformer also sets
    graph_dynamic_vbar_blocks/prefetch_dynamic_vbars, but has no
    audio_decoder, so it is NOT matched here.
    """
    try:
        if type(cond_stage_model).__name__ == "MiniMaxMusic3TEModel":
            return True
        model = getattr(cond_stage_model, "model", None)
        if model is not None and hasattr(model, "audio_decoder"):
            return True
    except Exception:
        pass
    return False


def find_te_containers(cond_stage_model):
    results = []
    seen_ids = set()

    def _recurse(module, depth=0):
        if depth > 20:
            return
        for name in CONTAINER_NAMES:
            c = getattr(module, name, None)
            if (isinstance(c, (nn.ModuleList, list)) and
                len(c) > 0 and hasattr(c[0], "forward") and
                id(c) not in seen_ids):
                seen_ids.add(id(c))
                results.append((name, c, module))
        for child_name, child in module.named_children():
            if isinstance(child, (nn.ModuleList, list)):
                continue
            _recurse(child, depth + 1)

    _recurse(cond_stage_model)
    return results


def install_te_block_swap(cond_stage_model, compute_device, offload_device,
                          num_blocks=-1):
    # MiniMax Music3T dedicated branch: its autoregressive AR loop captures the
    # audio decoder in a CUDA graph (model_prefetch.prefetch_queue_pop with
    # enable_graph=True). Block swap is incompatible with graph capture (weight
    # transfers + synchronize are forbidden while capturing), so the whole TE
    # is excluded from swap and keeps ComfyUI's native vbar + CUDA graph path.
    # Other text encoders (Krea2, Gemma4, ...) are unaffected.
    if _is_minimax_music_te(cond_stage_model):
        logger.info("UniBlockSwapTE: MiniMax Music3T detected - CUDA graph "
                    "capture on the audio decoder is incompatible with block "
                    "swap; skipping TE swap (native vbar prefetch kept)")
        return [], lambda: None, set()

    containers = find_te_containers(cond_stage_model)

    if not containers:
        return [], lambda: None, set()

    mgr_list = []
    container_names = set()
    parent_to_mgrs = {}

    for name, orig, parent in containers:
        total = len(orig)
        win = num_blocks if num_blocks > 0 else 1
        win = max(1, min(win, total))
        if num_blocks > 0 and win >= total:
            logger.info("UniBlockSwapTE: '%s' (%s) = %d blocks, NO swap (num_blocks=%d >= total)",
                         name, type(parent).__name__, total, num_blocks)
            continue

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            window_size=win,
        )
        swl.container_name = name
        setattr(parent, name, swl)
        mgr_list.append(swl)
        container_names.add(name)

        parent_id = id(parent)
        if parent_id not in parent_to_mgrs:
            parent_to_mgrs[parent_id] = (parent, parent.forward, [])
        parent_to_mgrs[parent_id][2].append(swl)

        # Every block participates: the first `win` blocks form the resident
        # prefix (pushed into CUDA once, kept for the whole inference); the
        # rest stay on the plugin's original lazy path. GGUF blocks keep their
        # mmap refs (dequantized on demand by the GGUF plugin); safetensor
        # blocks are restored by vbar on access.
        n_gguf = 0
        for i in range(total):
            if _has_ggml_params(swl._modules[str(i)]):
                _backup_ggml_refs(swl._modules[str(i)])
                n_gguf += 1
        logger.info("UniBlockSwap: '%s' GGUF blocks: %d/%d", name, n_gguf, total)

        logger.info("UniBlockSwapTE: '%s' (%s) = %d blocks, prefix resident = %d, tail lazy (num_blocks=%d)",
                     name, type(parent).__name__, total, swl.prefix_count, num_blocks)

        # Push the resident prefix into CUDA right away. The TE forward wrapper
        # offloads everything (prefix included) when the TE run finishes.
        swl.load_prefix()

    wrapped_parents = []
    for parent_id, (parent, orig_fwd, parent_mgrs) in parent_to_mgrs.items():
        def make_wrapped(_orig_fwd=orig_fwd, _mgrs=parent_mgrs, _cdevice=compute_device,
                         _root=cond_stage_model):
            def wrapped(*args, **kwargs):
                try:
                    return _orig_fwd(*args, **kwargs)
                finally:
                    for m in _mgrs:
                        m.offload_swap_blocks()
                    backup_cleaner = getattr(_root, '_uniblockswap_backup_cleanup', None)
                    patcher = getattr(_root, '_patcher_ref', None)
                    if backup_cleaner is not None and patcher is not None:
                        backup_cleaner(patcher)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(_cdevice)
                        gc.collect()
                        torch.cuda.empty_cache()
            return wrapped
        parent.forward = make_wrapped()
        wrapped_parents.append((parent, orig_fwd))

    def cleanup():
        for name, orig, parent in containers:
            current = getattr(parent, name, None)
            if hasattr(current, 'offload_swap_blocks'):
                setattr(parent, name, orig)
        for parent, orig_fwd in wrapped_parents:
            parent.forward = orig_fwd

    return mgr_list, cleanup, container_names
