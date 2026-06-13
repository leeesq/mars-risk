"""建模会话的样本切分辅助函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from mars.compute import FrameLike
from mars.modeling.contracts.specs import SplitSpec
from mars.modeling.workflows.splitter import MarsModelDataSplitter

if TYPE_CHECKING:
    from mars.modeling.workflows.session import MarsModelingSession


def session_slice(
    session: MarsModelingSession,
    df: FrameLike,
    *,
    time_col: str,
    split_ratios: Mapping[str, float],
    target: str | None = None,
    mode: str = "strict",
    train_key: str = "train",
    val_key: str = "val",
    random_seed: int = 42,
) -> FrameLike:
    """执行会话级样本切分。"""
    split_spec = SplitSpec(
        time_col=time_col,
        label_col=target or session.tuner.spec.target,
        mode=mode.lower(),
        train_key=train_key,
        val_key=val_key,
        random_seed=random_seed,
    )
    splitter = MarsModelDataSplitter()
    if split_spec.mode == "strict":
        return splitter.split_by_time_strictly(
            df,
            time_col=split_spec.time_col,
            target=split_spec.label_col,
            split_ratios=dict(split_ratios),
            dataset_flag_col=session.tuner.spec.dataset_flag_col,
        )
    if split_spec.mode == "hybrid":
        return splitter.split_hybrid_random_val(
            df,
            time_col=split_spec.time_col,
            target=split_spec.label_col,
            split_ratios=dict(split_ratios),
            dataset_flag_col=session.tuner.spec.dataset_flag_col,
            train_key=split_spec.train_key,
            val_key=split_spec.val_key,
            random_seed=split_spec.random_seed,
        )
    raise ValueError(f"Unsupported slice mode: {mode!r}. Expected 'strict' or 'hybrid'.")
