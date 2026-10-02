"""将案例包中的 Markdown 提示材料同时保留为可下载原文。"""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig


def on_post_build(config: MkDocsConfig) -> None:
    """复制已生成原文；不重新计算统计或修改报告。

    Parameters
    ----------
    config : MkDocsConfig
        MkDocs 当前构建目录与源码目录配置。

    Notes
    -----
    MkDocs 默认将 Markdown 渲染为页面，不能用渲染页冒充 ZIP 中的原文下载。
    本 hook 只保留案例静态目录中的 Markdown 字节，HTML 页面仍由原构建器生成。
    """
    source = Path(config.docs_dir) / "assets/cases"
    destination = Path(config.site_dir) / "assets/cases"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.md"):
        (destination / path.name).write_bytes(path.read_bytes())
