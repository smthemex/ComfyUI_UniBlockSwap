"""UniBlockSwap: DynamicVRAM bypass switch for ComfyUI >= 0.35.
"""

import contextlib
import logging
import os

log = logging.getLogger("UniBlockSwap.LegacyVRAM")

__all__ = [
    "SWITCH", "MODE", "ENV", "NO_PIN", "NO_PIN_ENV", "NODE_TYPES",
    "apply", "set_bypass", "restore", "bypass_scope", "mode",
    "status", "runtime_report", "description_suffix",
]

# ---------------------------------------------------------------------------
# 开关
# ---------------------------------------------------------------------------
SWITCH = True                        # 总开关 (False = 完全不做任何事)
MODE = "auto"                        # "auto"   = 只在工作流用到本插件节点时才绕过 (默认)
                                     # "always" = 老行为: 进程启动就绕过 (整机生效)
ENV = "UNIBLOCKSWAP_LEGACY_VRAM"     # 环境变量覆盖 (可选)
MODE_ENV = ENV                       # 同上, 语义化别名

# 本插件的节点 class_type (auto 模式靠它扫图; 新增节点记得加进来)
NODE_TYPES = ("UniBlockSwap", "UniBlockSwapTE", "UniBlockSwapCacheControl")

# 产出模型对象的节点 (auto 模式需要强制它们重新执行, 见 _bust_model_reload)
MODEL_OUTPUT_TYPES = ("MODEL", "CLIP", "VAE")
BUST_KEY = "__uniblockswap_reload__"

# 可选: 同时清零 pinned memory 预算 (Windows 上"共享 GPU 内存"增长的来源)。
# 会让"共享显存"数字变干净, 但 H2D 传输会慢一点。
NO_PIN = False
NO_PIN_ENV = "UNIBLOCKSWAP_LEGACY_VRAM_NO_PIN"

MIN_VERSION = (0, 35)                # 从 0.35 起 DynamicVRAM 默认开启
MIN_VERSION_STR = "0.35"

_ON = ("1", "true", "yes", "on", "enable", "enabled", "open", "always", "force")
_OFF = ("0", "false", "no", "off", "disable", "disabled", "close", "never")
_DRY = ("soft", "dry", "dry-run", "dryrun", "test", "check")
_AUTO = ("auto", "smart", "graph", "on-demand", "ondemand", "prompt", "node")

_state = {"applied": False, "reason": "not evaluated", "version": "unknown", "changes": []}

# /prompt 钩子的状态
_gate = {"installed": False, "last": None, "bannered": False, "seq": 0, "busted": 0}

# 首次 flip 之前的全局量快照 (回退目标 = 启动时的原值)
_original = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _env(name, default=""):
    return str(os.environ.get(name, default)).strip().lower()


def mode():
    """当前模式: "off" / "soft" / "auto" / "always" (环境变量优先于代码里的 SWITCH/MODE)。"""
    env = _env(ENV)
    if env in _OFF:
        return "off"
    if env in _DRY:
        return "soft"
    if env in _AUTO:
        return "auto"
    if env in _ON:
        return "always"
    if not SWITCH:
        return "off"
    text = str(MODE).strip().lower()
    return text if text in ("auto", "always") else "auto"


def switch_enabled():
    """开关是否开启 (环境变量优先于代码里的 SWITCH)。"""
    return mode() != "off"


def dry_run():
    return _env(ENV) in _DRY


def no_pin_enabled():
    env = _env(NO_PIN_ENV)
    if env in _ON:
        return True
    if env in _OFF:
        return False
    return bool(NO_PIN)


def comfyui_version():
    try:
        import comfyui_version
        return str(getattr(comfyui_version, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _parse_version(text):
    parts = []
    for chunk in str(text).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def version_ok():
    """ComfyUI 版本是否 >= 0.35 (拿不到版本号时退化为能力探测)。"""
    parsed = _parse_version(comfyui_version())
    if len(parsed) >= 2:
        return parsed >= MIN_VERSION
    import comfy.model_patcher as model_patcher
    return hasattr(model_patcher, "ModelPatcherDynamic")


def dynamic_active():
    """当前进程是否还处于 DynamicVRAM 模式。"""
    import comfy.memory_management as memory_management
    import comfy.model_patcher as model_patcher
    core = getattr(model_patcher, "CoreModelPatcher", None)
    if core is not None and core is not getattr(model_patcher, "ModelPatcher", None):
        return True
    return bool(getattr(memory_management, "aimdo_enabled", False))


# ---------------------------------------------------------------------------
# auto 模式: /prompt 钩子 (只在工作流真的用到本插件节点时才绕过)
# ---------------------------------------------------------------------------
def _graph_has_node(prompt):
    """递归扫描工作流图, 判断有没有本插件的节点 (含嵌套 subgraph 定义)。"""
    if not isinstance(prompt, dict):
        return False
    stack = [prompt]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            cls = obj.get("class_type")
            if isinstance(cls, str) and cls in NODE_TYPES:
                return True
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)
    return False


def _pending_uses_node():
    """队列里正在执行 / 排队等待的图有没有用到本插件节点。"""
    try:
        import server
        srv = getattr(server.PromptServer, "instance", None)
        queue = getattr(srv, "prompt_queue", None)
        if queue is None:
            return False
        running, queued = queue.get_current_queue_volatile()
        for item in list(running) + list(queued):
            try:
                if _graph_has_node(item[2]):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _loader_class_types():
    """产出 MODEL/CLIP/VAE 的 class_type 集合 = 模型对象的来源节点。"""
    try:
        import nodes
        mapping = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    except Exception:
        return frozenset()
    found = set()
    for cls_name, cls in mapping.items():
        try:
            types = getattr(cls, "RETURN_TYPES", ()) or ()
        except Exception:
            continue
        if isinstance(types, str):
            types = (types,)
        if any(t in MODEL_OUTPUT_TYPES for t in types):
            found.add(cls_name)
    return frozenset(found)


def _bust_model_reload(prompt, token):
    """让"会产出模型"的节点缓存失效, 迫使它们重新执行 (返回命中的节点数)。

    为什么必须这么做 —— ComfyUI 的节点输出缓存是**跨提交保留**的
    (execution.py:670-673, reset() 只在 __init__ 调用), 缓存键 = 输入签名。
    而 CheckpointLoaderSimple 这类 loader **没有 IS_CHANGED** ->
    IsChangedCache.get() 直接返回 False (execution.py:82-84) -> 只要 ckpt_name
    没变, 签名恒定 (caching.py:109-127) => 缓存命中 => **loader 根本不执行**,
    上一轮那个 dynamic patcher 被原样复用, 我们翻转的全局量对它完全无效。

    手段: 给它的 inputs 塞一个多余的常量键 -> 签名变化 -> 缓存不命中 -> 重跑
    -> 重新构造 patcher (此时全局量已是 legacy)。下游节点因为签名包含 ancestors
    (caching.py:125) 会自动一起失效。

    为什么安全 (已对真模块实测):
      - validate_inputs 只遍历节点自己声明的输入 -> 多余键不参与校验 (execution.py:896-898)
      - get_input_data 对未声明的输入 key 直接忽略 -> 不会传给节点函数 (execution.py:174-190)
      - 值必须是**常量**: 长度 2 的 list 会被当成连线 link 去解析上游输出
    """
    loaders = _loader_class_types()
    if not loaders:
        return 0
    hits = 0
    stack = [prompt]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            cls = obj.get("class_type")
            if isinstance(cls, str) and cls in loaders:
                inputs = obj.get("inputs")
                if isinstance(inputs, dict):
                    inputs[BUST_KEY] = token
                    hits += 1
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)
    return hits


def _install_gate():
    """装上 /prompt 钩子 (官方 add_on_prompt_handler)。失败只记日志, 不抛。"""
    if _gate["installed"]:
        return True
    try:
        import server
        srv = getattr(server.PromptServer, "instance", None)
        if srv is None or not hasattr(srv, "add_on_prompt_handler"):
            return False
        handlers = getattr(srv, "on_prompt_handlers", None)
        if isinstance(handlers, list) and _on_prompt in handlers:
            _gate["installed"] = True
            return True
        srv.add_on_prompt_handler(_on_prompt)
        _gate["installed"] = True
        return True
    except Exception:
        log.warning("[UniBlockSwap] could not install the /prompt gate", exc_info=True)
        return False


def _on_prompt(json_data):
    """每次提交工作流时决定要不要绕过。

    ⚠️ server 的 trigger_on_prompt 会把返回值当成新的 json_data, 必须原样返回。
    """
    try:
        if mode() != "auto":
            return json_data
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        used = _graph_has_node(prompt) or _pending_uses_node()
        _gate["last"] = bool(used)

        if used:
            changed = set_bypass(True, quiet=True,
                                 reason="auto: workflow uses %s" % "/".join(NODE_TYPES))
            if changed:
                # 刚翻过去 -> 换一个 token, 让上一轮的模型缓存彻底失效
                _gate["seq"] += 1
            if _graph_has_node(prompt):
                # 本轮图真的用到本插件节点: 必须保证模型**重新构造**。模型通常在上游
                # loader 里早就建好了(而且很可能命中跨提交的输出缓存), 光翻全局量对它
                # 没用。token 只在翻转时递增 -> 未翻转的后续提交用同一个 token, 缓存键
                # 稳定, 可以正常命中"已经按 legacy 构造好"的那份。
                forced = _bust_model_reload(prompt, str(_gate["seq"]))
                _gate["busted"] = forced
                if changed and forced:
                    log.info("[UniBlockSwap] forced %d model loader(s) to re-run so the model "
                             "is rebuilt under the legacy path", forced)
            if changed:
                if not _gate["bannered"]:
                    _gate["bannered"] = True
                    _banner(
                        [_rule()]
                        + [" UniBlockSwap: DynamicVRAM BYPASSED (mode=auto)"]
                        + ["   " + c for c in changed]
                        + ["   why  : this workflow (or one still in the queue) uses "
                           + "/".join(NODE_TYPES),
                           "   scope: models built from now on. To undo WITHOUT restarting a",
                           "          workflow: submit a workflow that does not use this node",
                           "          (the bypass is restored automatically before loading).",
                           "          Already-built models cannot be converted - but any loader",
                           "          that still has to run will rebuild them the right way."]
                        + [_rule()]
                    )
        elif _state["applied"]:
            changed = set_bypass(False, quiet=True,
                                 reason="auto: no UniBlockSwap node in this workflow")
            if changed:
                _gate["busted"] = 0
                log.info("[UniBlockSwap] workflow does not use this node -> DynamicVRAM "
                         "restored for models loaded from now on")
    except Exception:
        log.warning("[UniBlockSwap] /prompt gate failed", exc_info=True)
    return json_data


# ---------------------------------------------------------------------------
# 回退机制: 快照 / 应用 / 还原
# ---------------------------------------------------------------------------
def _snapshot():
    """记下当前(通常是启动时)的全局量, 作为 restore() 的目标。"""
    import comfy.memory_management as memory_management
    import comfy.model_patcher as model_patcher
    from comfy.cli_args import args
    snap = {
        "CoreModelPatcher": model_patcher.CoreModelPatcher,
        "aimdo_enabled": bool(getattr(memory_management, "aimdo_enabled", False)),
        "disable_dynamic_vram": bool(getattr(args, "disable_dynamic_vram", False)),
        "disable_pinned_memory": bool(getattr(args, "disable_pinned_memory", False)),
        "MAX_PINNED_MEMORY": None,
    }
    try:
        import comfy.model_management as model_management
        snap["MAX_PINNED_MEMORY"] = int(getattr(model_management, "MAX_PINNED_MEMORY", 0))
    except Exception:
        pass
    return snap


def _apply_legacy(no_pin):
    """把全局量翻成 legacy 口径, 返回实际改动项。"""
    import comfy.memory_management as memory_management
    import comfy.model_patcher as model_patcher
    from comfy.cli_args import args

    changed = []
    if getattr(memory_management, "aimdo_enabled", False):
        memory_management.aimdo_enabled = False
        changed.append("comfy.memory_management.aimdo_enabled = False")
    if model_patcher.CoreModelPatcher is not model_patcher.ModelPatcher:
        model_patcher.CoreModelPatcher = model_patcher.ModelPatcher
        changed.append("comfy.model_patcher.CoreModelPatcher = ModelPatcher")
    if not getattr(args, "disable_dynamic_vram", False):
        args.disable_dynamic_vram = True
        changed.append("args.disable_dynamic_vram = True")

    if no_pin:
        try:
            import comfy.model_management as model_management
            if getattr(model_management, "MAX_PINNED_MEMORY", 0) > 0:
                model_management.MAX_PINNED_MEMORY = 0
                changed.append("comfy.model_management.MAX_PINNED_MEMORY = 0")
        except Exception:
            pass
        if not getattr(args, "disable_pinned_memory", False):
            args.disable_pinned_memory = True
            changed.append("args.disable_pinned_memory = True")
    return changed


def _restore_state(snap):
    """按快照还原全局量, 返回实际改动项。"""
    import comfy.memory_management as memory_management
    import comfy.model_patcher as model_patcher
    from comfy.cli_args import args

    changed = []
    if bool(getattr(memory_management, "aimdo_enabled", False)) != snap["aimdo_enabled"]:
        memory_management.aimdo_enabled = snap["aimdo_enabled"]
        changed.append("comfy.memory_management.aimdo_enabled = %s" % snap["aimdo_enabled"])
    if model_patcher.CoreModelPatcher is not snap["CoreModelPatcher"]:
        model_patcher.CoreModelPatcher = snap["CoreModelPatcher"]
        changed.append("comfy.model_patcher.CoreModelPatcher = %s"
                       % getattr(snap["CoreModelPatcher"], "__name__", "?"))
    if bool(getattr(args, "disable_dynamic_vram", False)) != snap["disable_dynamic_vram"]:
        args.disable_dynamic_vram = snap["disable_dynamic_vram"]
        changed.append("args.disable_dynamic_vram = %s" % snap["disable_dynamic_vram"])
    if bool(getattr(args, "disable_pinned_memory", False)) != snap["disable_pinned_memory"]:
        args.disable_pinned_memory = snap["disable_pinned_memory"]
        changed.append("args.disable_pinned_memory = %s" % snap["disable_pinned_memory"])
    if snap["MAX_PINNED_MEMORY"] is not None:
        try:
            import comfy.model_management as model_management
            if int(getattr(model_management, "MAX_PINNED_MEMORY", 0)) != snap["MAX_PINNED_MEMORY"]:
                model_management.MAX_PINNED_MEMORY = snap["MAX_PINNED_MEMORY"]
                changed.append("comfy.model_management.MAX_PINNED_MEMORY = %d"
                               % snap["MAX_PINNED_MEMORY"])
        except Exception:
            pass
    return changed


def set_bypass(enabled=True, no_pin=None, quiet=False, reason=""):
    """运行时切换绕过态 (幂等)。返回实际改动项 list, 空 list = 已经是目标态。

    对**之后构造**的模型立即生效; 已构造的模型不受影响 (想让它们也变,
    需要让它们重新加载: comfy 的 /free 卸载模型、换 checkpoint、或重启)。
    """
    global _original
    if _original is None:
        _original = _snapshot()

    if enabled:
        if no_pin is None:
            no_pin = no_pin_enabled()
        changed = _apply_legacy(no_pin)
    else:
        changed = _restore_state(_original)

    _state["applied"] = not dynamic_active()
    _state["changes"] = changed
    _state["reason"] = reason or ("runtime: bypass on" if enabled else "runtime: bypass off")

    if changed and not quiet:
        title = ("DynamicVRAM BYPASSED (runtime)" if enabled
                 else "DynamicVRAM bypass DISABLED (runtime revert)")
        extra = (["   note  : this only affects models constructed from now on;",
                  "           already-built models stay on the legacy path - unload them",
                  "           (comfy /free, or switch checkpoint) or restart to get back."]
                 if not enabled else
                 ["   note  : only models constructed from now on are affected;"])
        _banner([_rule()]
                + [" UniBlockSwap: " + title]
                + ["   " + c for c in changed]
                + extra
                + [_rule()])
    return changed


def restore():
    """运行时回退: 回到启动时的原值 (== set_bypass(False))。"""
    return set_bypass(False)


@contextlib.contextmanager
def bypass_scope(enabled=True, no_pin=None, quiet=True):
    """上下文管理器: 域内**新构造**的模型走 (或不走) legacy, 退出时还原。

        with legacy_vram.bypass_scope():        # eval / 脚本化加载时用
            model = comfy.sd.load_diffusion_model(path)

    退出时按"进入前"的快照还原 (内部 try/finally, 体内抛异常也照样还原),
    嵌套安全。注意它只能影响域内新建的模型。
    """
    snap = _snapshot()
    applied_before = _state["applied"]
    reason_before = _state["reason"]
    try:
        set_bypass(enabled, no_pin=no_pin, quiet=quiet,
                   reason="bypass_scope(%s)" % bool(enabled))
        yield
    finally:
        try:
            _restore_state(snap)
            _state["applied"] = applied_before
            _state["reason"] = reason_before
        except Exception:
            log.warning("[UniBlockSwap] bypass_scope restore failed", exc_info=True)


def status():
    return {
        "switch": switch_enabled(),
        "mode": mode(),
        "dry_run": dry_run(),
        "version": comfyui_version(),
        "version_ok": version_ok(),
        "dynamic_active": dynamic_active(),
        "applied": _state["applied"],
        "reason": _state["reason"],
        "changes": list(_state["changes"]),
        "revertable": _original is not None,
        "gate_installed": bool(_gate["installed"]),
        "last_prompt_used_node": _gate["last"],
        "last_forced_reload_nodes": _gate["busted"],
    }


def _banner(lines):
    """醒目的控制台提示 (绝不抛异常, 免得 custom node 导入失败)。"""
    for line in lines:
        try:
            log.warning("%s", line)
        except Exception:
            pass
    try:
        print("\n".join(lines), flush=True)
    except Exception:
        pass


def _rule(char="="):
    return char * 74


def description_suffix():
    """给节点 DESCRIPTION 用的状态后缀 (UI 上可见)。"""
    try:
        if _state["applied"]:
            return " [DynamicVRAM bypass: ON]"
        current = mode()
        if current == "auto" and dynamic_active():
            return " [DynamicVRAM bypass: AUTO - 只在本节点被用到时才生效]"
        if current == "always" and dynamic_active():
            return " [DynamicVRAM bypass: 未生效, 需重启]"
        return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 生效
# ---------------------------------------------------------------------------
def _apply_impl():
    version = comfyui_version()
    _state["version"] = version
    current = mode()

    if current == "off":
        _state["reason"] = "switch off (env %s=%s)" % (ENV, _env(ENV) or "unset")
        log.info("[UniBlockSwap] DynamicVRAM bypass switch is OFF; keeping DynamicVRAM")
        return False

    if not version_ok():
        _state["reason"] = "ComfyUI %s < %s, bypass not needed" % (version, MIN_VERSION_STR)
        log.info("[UniBlockSwap] ComfyUI %s: no DynamicVRAM to bypass", version)
        return False

    if not dynamic_active():
        _state["reason"] = "DynamicVRAM already off (startup args)"
        log.info("[UniBlockSwap] DynamicVRAM is already off at startup; nothing to bypass")
        return False

    from comfy.cli_args import args

    if getattr(args, "enable_dynamic_vram", False):
        _state["reason"] = "--enable-dynamic-vram given explicitly, bypass skipped"
        _banner([
            _rule(),
            " UniBlockSwap: bypass switch is ON but NOT applied",
            "   reason: ComfyUI was started with --enable-dynamic-vram (explicit CLI flag wins)",
            "   fix   : drop that flag, or set UNIBLOCKSWAP_LEGACY_VRAM=0 to silence this",
            _rule(),
        ])
        return False

    if current == "soft":
        _state["reason"] = "dry run"
        _banner([
            _rule(),
            " UniBlockSwap: DRY RUN (UNIBLOCKSWAP_LEGACY_VRAM=%s)" % _env(ENV),
            "   would set: comfy.memory_management.aimdo_enabled = False",
            "   would set: comfy.model_patcher.CoreModelPatcher = ModelPatcher",
            "   nothing was changed, DynamicVRAM stays active",
            _rule(),
        ])
        return False

    if current == "auto":
        armed = _install_gate()
        _state["reason"] = ("auto: armed, engages only when a workflow uses %s"
                            % "/".join(NODE_TYPES))
        _banner(
            [_rule()]
            + [" UniBlockSwap: DynamicVRAM bypass ARMED (mode=auto)"]
            + ["   DynamicVRAM is NOT touched right now - it stays as configured at startup."
               if armed else
               "   ⚠️ could not install the /prompt gate; bypass stays OFF (see warnings above)."]
            + ["   rule  : 只有当提交的工作流真的用到 %s 时," % "/".join(NODE_TYPES),
               "           才会在模型构造之前把 DynamicVRAM 翻回 legacy 口径;",
               "           队列里没有这个节点时自动保持/还原 dynamic。",
               "   cost  : while it is engaged: no aimdo demand paging / prefetch /",
               "           cuda-graph weight load + conservative legacy VRAM estimates.",
               "   manual: legacy_vram.set_bypass(True/False) / restore() /",
               "           with legacy_vram.bypass_scope(): (脚本化加载用)",
               "   force : UNIBLOCKSWAP_LEGACY_VRAM=1 -> 老行为, 进程启动就整机绕过",
               _rule()]
        )
        return armed

    changed = set_bypass(True, quiet=True, reason="import-time bypass (mode=always)")
    _state["reason"] = "bypassed DynamicVRAM (mode=always)"

    _banner(
        [_rule()]
        + [" UniBlockSwap: DynamicVRAM BYPASSED  (mode=always, ComfyUI %s >= %s)"
           % (version, MIN_VERSION_STR)]
        + ["   " + c for c in changed]
        + [
            "   effect: this whole process uses the pre-0.35 legacy ModelPatcher path",
            "           (block swap keeps its _load_list filtering + per-block sync copy)",
            "   cost  : no aimdo demand paging / prefetch / cuda-graph weight load,",
            "           legacy conservative VRAM estimates (big models may OOM on small cards)",
            "   note  : EVERY workflow is affected, whether or not it uses this node.",
            "           If you only want it for workflows using this node, use mode=auto",
            "           (unset UNIBLOCKSWAP_LEGACY_VRAM / set it to 'auto').",
            "   revert: legacy_vram.set_bypass(False) (or restore()) - runtime, applies to",
            "           models loaded from then on; or UNIBLOCKSWAP_LEGACY_VRAM=0 + restart",
            _rule(),
        ]
    )
    return True


def apply():
    """在 __init__.py 里 import 时调用一次 (auto 模式只装钩子, 不翻转全局量)。

    任何异常都不会阻断 custom node 加载。
    """
    try:
        return _apply_impl()
    except Exception:
        _state["reason"] = "apply() failed"
        log.warning("[UniBlockSwap] DynamicVRAM bypass failed, keeping defaults", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# 节点执行时的提示
# ---------------------------------------------------------------------------
_HINTED = set()


def runtime_report(node="UniBlockSwap", patcher=None):
    """节点执行时调用: 把当前生效状态提示给用户, 返回状态字符串。"""
    try:
        dynamic = dynamic_active()
        patcher_dynamic = False
        try:
            patcher_dynamic = bool(patcher is not None and patcher.is_dynamic())
        except Exception:
            pass

        if _state["applied"] and not patcher_dynamic:
            if "ok" not in _HINTED:
                _HINTED.add("ok")
                log.info("[%s] DynamicVRAM bypass active (legacy ModelPatcher), "
                         "block swap runs on its native code path", node)
            return "bypass"

        if patcher_dynamic or dynamic:
            if "warn" not in _HINTED:
                _HINTED.add("warn")
                lines = [
                    _rule("!"),
                    " UniBlockSwap: still running under DynamicVRAM",
                    "   reason: %s" % _state["reason"],
                    "   hint  : block swap degrades into the old synchronous cast path and",
                    "           saves much less VRAM.",
                ]
                if mode() == "auto":
                    lines += [
                        "   note  : mode=auto only helps models that get CONSTRUCTED after the",
                        "           prompt is submitted, and it forces every MODEL/CLIP/VAE",
                        "           producing node to re-run exactly so that happens.",
                        "           Seeing this warning means the model STILL came from a cache:",
                        "           its loader did not re-run (custom loader with its own cache,",
                        "           e.g. GGUF) or it was built outside the /prompt path.",
                        "   fix   : restart ComfyUI, or pick another checkpoint and switch back,",
                        "           or start with --cache-none.",
                    ]
                else:
                    lines += [
                        "   hint  : to bypass: legacy_vram.set_bypass(True) then RELOAD the",
                        "           model (already-built models cannot be converted);",
                        "           UNIBLOCKSWAP_LEGACY_VRAM=1 + restart also works.",
                    ]
                lines.append(_rule("!"))
                _banner(lines)
            return "dynamic"

        if "off" not in _HINTED:
            _HINTED.add("off")
            log.info("[%s] DynamicVRAM bypass switch off; legacy path used anyway (%s)",
                     node, _state["reason"])
        return "off"
    except Exception:
        return "unknown"
