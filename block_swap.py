"""
UniBlockSwap - Universal single-block swap for ComfyUI.
Safetensor blocks: freed to meta on swap, restored by vbar automatically.
GGUF blocks: freed to CPU on swap, moved to GPU when accessed.
"""

import gc
import logging
import torch
import torch.nn as nn

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


def _has_ggml_params(module):
    """Check if module has GGMLTensor parameters (quantized GGUF weights)."""
    for p in module.parameters():
        if hasattr(p, 'tensor_type'):
            return True
    return False


def _free_to_meta(module):
    """Free param data to meta tensor - NO CPU copy created.
    The module structure is preserved. next load() restores from backup."""
    for param in module.parameters(recurse=False):
        param.data = torch.empty(0, device='meta')


class SwappableModuleList(nn.ModuleList):
    def __init__(self, modules, compute_device, offload_device,
                 non_swap_count=0):
        super().__init__(modules)
        self.compute_device = compute_device
        self.offload_device = offload_device
        self.non_swap_count = non_swap_count
        self.total_count = len(modules)
        self._loaded_swap_idx = -1
        self.container_name = ''

    def _load_swap(self, local_idx):
        idx = local_idx + self.non_swap_count
        if local_idx == self._loaded_swap_idx:
            return
        if self._loaded_swap_idx >= 0:
            prev = self._loaded_swap_idx + self.non_swap_count
            try:
                # FREE previous block GPU memory
                if _has_ggml_params(self._modules[str(prev)]):
                    # GGUF: move quantized data to CPU (preserves GGMLTensor attributes)
                    self._modules[str(prev)].to(self.offload_device)
                else:
                    # Safetensor: set to meta (vbar restores automatically)
                    _free_to_meta(self._modules[str(prev)])
                for m in self._modules[str(prev)].modules():
                    for attr in ('_v', '_prefetch', '_v_signature'):
                        if hasattr(m, attr):
                            try:
                                delattr(m, attr)
                            except Exception:
                                pass
            except Exception:
                pass
        # LOAD current block if GGUF
        if _has_ggml_params(self._modules[str(idx)]):
            self._modules[str(idx)].to(self.compute_device)
        # else: safetensor - vbar handles restoration
        self._loaded_swap_idx = local_idx

    def offload_swap_blocks(self):
        for i in range(self.non_swap_count, self.total_count):
            try:
                if _has_ggml_params(self._modules[str(i)]):
                    self._modules[str(i)].to(self.offload_device)
                else:
                    _free_to_meta(self._modules[str(i)])
                for m in self._modules[str(i)].modules():
                    for attr in ('_v', '_prefetch', '_v_signature'):
                        if hasattr(m, attr):
                            try:
                                delattr(m, attr)
                            except Exception:
                                pass
            except Exception:
                pass
        self._loaded_swap_idx = -1

    def _apply(self, fn, recurse=True):
        """Apply fn to non-swap blocks only.
        
        CRITICAL: Prevents model.to(device_to) from moving swap block
        GGMLTensors to GPU, which would cause a VRAM spike (12GB).
        Safetensor swap blocks are already meta (no-op), so this only
        affects GGUF paths.
        
        nn.ModuleList._apply(recurse=False) applies fn to all _modules
        entries INCLUDING swap blocks. We skip that and handle only
        non_swap_count blocks manually.
        """
        for i in range(self.non_swap_count):
            try:
                child = self._modules.get(str(i))
                if child is not None:
                    child._apply(fn, recurse)
            except Exception:
                pass
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
        if idx >= self.non_swap_count:
            self._load_swap(idx - self.non_swap_count)
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
        return None, lambda: None, set()

    first_swl = None
    all_names = set()

    for name, orig in all_containers:
        total = len(orig)
        n = num_blocks if num_blocks > 0 else total
        n = max(1, min(n, total))

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            non_swap_count=total - n,
        )
        swl.container_name = name
        setattr(diffusion_model, name, swl)
        all_names.add(name)
        if first_swl is None:
            first_swl = swl
        logger.info("UniBlockSwap: '%s' = %d blocks, swapping %d",
                     name, total, n)

        # For GGUF: offload swap blocks to CPU immediately.
        # Safetensor blocks stay on GPU (original behavior).
        for i in range(total - n, total):
            blk = swl._modules[str(i)]
            if _has_ggml_params(blk):
                blk.to(offload_device)

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
    containers = find_te_containers(cond_stage_model)

    if not containers:
        return [], lambda: None, set()

    mgr_list = []
    container_names = set()
    parent_to_mgrs = {}

    for name, orig, parent in containers:
        total = len(orig)
        n = num_blocks if num_blocks > 0 else total
        n = max(1, min(n, total))

        swl = SwappableModuleList(
            orig, compute_device, offload_device,
            non_swap_count=total - n,
        )
        swl.container_name = name
        setattr(parent, name, swl)
        mgr_list.append(swl)
        container_names.add(name)

        parent_id = id(parent)
        if parent_id not in parent_to_mgrs:
            parent_to_mgrs[parent_id] = (parent, parent.forward, [])
        parent_to_mgrs[parent_id][2].append(swl)

        logger.info("UniBlockSwapTE: '%s' (%s) = %d blocks, swapping %d",
                     name, type(parent).__name__, total, n)

        for i in range(total - n, total):
            blk = swl._modules[str(i)]
            if _has_ggml_params(blk):
                blk.to(offload_device)
            else:
                _free_to_meta(blk)

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