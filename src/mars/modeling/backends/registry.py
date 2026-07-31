"""训练与预测分发共用的后端注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from mars.modeling.backends.base import MarsBaseModelStrategy


@dataclass(frozen=True)
class BackendSpec:
    """注册后端的元数据描述。"""

    name: str
    strategy_cls: type[MarsBaseModelStrategy]
    aliases: tuple[str, ...]


BACKEND_REGISTRY: Dict[str, BackendSpec] = {}
_BACKEND_ALIASES: Dict[str, str] = {}
_BUILTINS_LOADED = False


def register_backend(name: str, *aliases: str):
    """按规范名称与别名注册后端策略类。"""

    def decorator(strategy_cls: type[MarsBaseModelStrategy]) -> type[MarsBaseModelStrategy]:
        canonical = str(name).lower()
        normalized_aliases = tuple(str(alias).lower() for alias in aliases)
        spec = BackendSpec(
            name=canonical,
            strategy_cls=strategy_cls,
            aliases=normalized_aliases,
        )
        BACKEND_REGISTRY[canonical] = spec
        _BACKEND_ALIASES[canonical] = canonical
        for alias in normalized_aliases:
            _BACKEND_ALIASES[alias] = canonical
        return strategy_cls

    return decorator


def ensure_builtin_backends_registered() -> None:
    """仅加载一次内置后端，确保装饰器副作用完成注册。"""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    import mars.modeling.backends as _builtin_backends  # noqa: F401

    _BUILTINS_LOADED = True


def resolve_backend_name(model_type: str) -> str:
    """将别名解析为规范后端名称。"""
    ensure_builtin_backends_registered()
    key = str(model_type).lower()
    if key not in _BACKEND_ALIASES:
        raise KeyError(key)
    return _BACKEND_ALIASES[key]


def has_backend(model_type: str) -> bool:
    """判断后端名称或别名是否已注册。"""
    ensure_builtin_backends_registered()
    return str(model_type).lower() in _BACKEND_ALIASES


def get_backend_spec(model_type: str) -> BackendSpec:
    """返回指定后端的注册元数据。"""
    canonical = resolve_backend_name(model_type)
    return BACKEND_REGISTRY[canonical]


def get_backend_strategy(model_type: str) -> type[MarsBaseModelStrategy]:
    """返回指定后端对应的策略类。"""
    return get_backend_spec(model_type).strategy_cls


def registered_backend_names() -> list[str]:
    """返回用于校验报错的已注册名称与别名列表。"""
    ensure_builtin_backends_registered()
    return sorted(_BACKEND_ALIASES)


def backend_map() -> dict[str, type[MarsBaseModelStrategy]]:
    """返回兼容旧心智的 alias 到策略类映射。"""
    ensure_builtin_backends_registered()
    return {alias: BACKEND_REGISTRY[canonical].strategy_cls for alias, canonical in _BACKEND_ALIASES.items()}


def iter_backend_specs() -> Iterable[BackendSpec]:
    """遍历已注册的规范后端元数据。"""
    ensure_builtin_backends_registered()
    return BACKEND_REGISTRY.values()
