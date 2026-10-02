"""从原 Logo 生成慢色流 SVG／GIF，不修改字形与几何结构。

生成依赖仅用于品牌资产：python -m pip install cairosvg pillow
在仓库根目录运行：python scripts/generate_logo_animation.py
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
PALETTES = (
    ("#14b8a6", "#8b5cf6", "#38bdf8"),
    ("#38bdf8", "#6366f1", "#a78bfa"),
    ("#8b5cf6", "#38bdf8", "#14b8a6"),
)
FRAMES = 64
DURATION_MS = 125


def _color(start: str, end: str, fraction: float) -> str:
    """平滑插值颜色；不改变亮度透明度造成闪烁。"""
    channels = [
        round(int(start[index:index + 2], 16) * (1 - fraction)
              + int(end[index:index + 2], 16) * fraction)
        for index in (1, 3, 5)
    ]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _frame(source: str, index: int, *, dark: bool) -> str:
    """只替换原渐变色，深色渲染使用原 Logo 的主题颜色。"""
    phase = index / FRAMES * len(PALETTES)
    segment = int(phase)
    fraction = (1 - math.cos(math.pi * (phase - segment))) / 2
    palette = [
        _color(start, end, fraction)
        for start, end in zip(PALETTES[segment], PALETTES[(segment + 1) % len(PALETTES)])
    ]
    values = iter(palette)
    frame = re.sub(r'(<stop\b[^>]*stop-color=")[^"]+("/>)',
                   lambda match: match[1] + next(values) + match[2], source)
    theme = (
        ".panel { fill: #111827; stroke: #312e81; } "
        ".letter:not(.accent) { stroke: #f8fafc; } "
        if dark else ""
    )
    # 原暗色规则在 .accent 之后；明确保留渐变笔画，不改其他字形。
    return frame.replace("</style>", theme + ".accent { stroke: url(#logo-gradient); }</style>")


def _animated_svg(source: str) -> str:
    """追加独立 SVG 内的 CSS 渐变，减少动态偏好时保留静态色。"""
    rules = [".accent { stroke: url(#logo-gradient); }"]
    for index in range(3):
        colors = [palette[index] for palette in PALETTES]
        rules.append(
            f"@keyframes mars-color-{index} {{ "
            f"0%, 100% {{ stop-color: {colors[0]}; }} "
            f"33.333% {{ stop-color: {colors[1]}; }} "
            f"66.667% {{ stop-color: {colors[2]}; }} }}"
        )
        rules.append(
            f"#logo-gradient stop:nth-child({index + 1}) "
            f"{{ animation: mars-color-{index} 8s ease-in-out infinite; }}"
        )
    rules.append(
        "@media (prefers-reduced-motion: reduce) { "
        "#logo-gradient stop { animation: none; } }"
    )
    return source.replace("</style>", "\n      " + "\n      ".join(rules) + "\n    </style>")


def main() -> None:
    """导出一个 SVG 增强资产与两份主题 GIF，原静态 SVG 不变。"""
    import cairosvg
    from PIL import Image

    source = (ASSETS / "mars-logo.svg").read_text(encoding="utf-8")
    (ASSETS / "mars-logo-animated.svg").write_text(_animated_svg(source), encoding="utf-8")
    for theme in ("light", "dark"):
        images = []
        for index in range(FRAMES):
            png = cairosvg.svg2png(
                bytestring=_frame(source, index, dark=theme == "dark").encode("utf-8"),
                output_width=920, output_height=178,
                background_color="#111827" if theme == "dark" else "#ffffff",
            )
            with Image.open(io.BytesIO(png)) as image:
                images.append(image.convert("RGB"))
        images[0].save(
            ASSETS / f"mars-logo-{theme}.gif", save_all=True, append_images=images[1:],
            duration=DURATION_MS, loop=0, optimize=True,
        )
        for image in images:
            image.close()


if __name__ == "__main__":
    main()
