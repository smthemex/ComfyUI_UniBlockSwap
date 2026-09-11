"""ComfyUI_UniBlockSwap.

NOTE: the legacy_vram import below MUST stay before the node import - it arms the
DynamicVRAM bypass, which can only be switched before a model object is built.
In the default mode ("auto") it does NOT change anything at import time: it only
installs a /prompt hook, so DynamicVRAM is bypassed solely for workflows that
actually use this node. Use UNIBLOCKSWAP_LEGACY_VRAM=1 to force the old
"bypass the whole process at startup" behaviour. See legacy_vram.py.
"""
from .legacy_vram import apply as _apply_legacy_vram

_apply_legacy_vram()  # ComfyUI >= 0.35 + switch ON -> use the legacy path

from .uniblockswap_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
