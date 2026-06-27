"""建模切分 mixin 的宿主协议。"""

from __future__ import annotations

from typing import Protocol


class _HybridKeyValidator(Protocol):
    """声明 hybrid 切分依赖的宿主校验能力。"""

    def _validate_hybrid_keys(
        self,
        split_ratios: dict[str, float],
        train_key: str,
        val_key: str,
    ) -> tuple[float, float, float]:
        """校验 hybrid 切分键并返回训练、验证和建模占比。"""
