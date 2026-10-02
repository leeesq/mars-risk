[中文](README.md) / **English**

<div align="center">
<img src="docs/assets/mars-logo.svg" alt="MARS" width="480">
<br>
<img src="docs/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="480">
<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>
</div>

# A risk analysis toolkit for humans and AI agents

**Data profiling · Binning evaluation · Feature selection · Correlation analysis · Score cross analysis · Rule mining**

Turn a Pandas or Polars wide table into analysis reports that people can read, query, deliver and reuse. Human views and external Agent evidence come from the same Report.

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/cases/readme-preview-mobile.png">
  <img src="docs/assets/cases/readme-preview.png" alt="Actual Score Cross matrix and public Agent query from the same synthetic report" width="1040">
</picture>

Actual browser capture of a synthetic Score Cross report; the current report UI is in Chinese. The matrix and the JSON excerpt share the same report identity and query scope. [Full evidence and provenance](docs/assets/cases/case-4.json) · [Runnable case](docs/demos/score-cross.md).

[Start analyzing](https://leeesq.github.io/mars-risk/getting-started/quickstart/) · [Seven practical cases](docs/demos/index.md) · [External Agent guide](https://leeesq.github.io/mars-risk/user-guide/external-agents/) · [Full documentation (Chinese)](https://leeesq.github.io/mars-risk/)

## Choose a task

| Your question | Capability | Result and runnable case |
| --- | --- | --- |
| Can this data be used as it is? | Data profiling | [Quality, labels, schema and distribution evidence](docs/demos/data-quality.md) |
| Which features discriminate risk and remain stable? | Binning evaluation | [Fixed reference bins, multi-target IV / KS / PSI](docs/demos/binning-stability.md) |
| Why keep one feature and remove another? | Feature selection + correlation analysis | [Decisions, signed redundancy, separate raw / WOE representations](docs/demos/selection-correlation.md) |
| Does an auxiliary score separate risk within a main-score tier? | Score cross analysis | [Interactive matrix, row-relative Δ, cell evidence and policy replay](docs/demos/score-cross.md) |
| Which candidate rules warrant review? | Rule mining — Experimental | [Discovery audit, independent validation and coverage](docs/demos/rule-evidence.md) |
| How can I save and query this analysis later? | Public reports | [Fresh-process loading, pagination and character budgets](docs/demos/saved-reports.md) |
| How do I deliver one analysis to different users? | Report exports | [HTML / Excel / .marsreport / bounded JSON / Agent prompt materials](docs/demos/report-delivery.md) |

The seven cases share 18,000 synthetic rows with a fixed seed. Discovery and validation are separate; observation has no `late60` labels, while `bad30` remains observed where available. Synthetic statistics do not establish business profit, causal effects, approval decisions or production readiness. Detailed parameters have one authority: [API Reference (Chinese)](https://leeesq.github.io/mars-risk/reference/).

## Installation and version boundaries

The source version is **0.0.28**. The examples use current public report capabilities. Install the verified source commit:

```bash
pip install "git+https://github.com/leeesq/mars-risk.git@746b8fa76439e6841b466bd590c35249a975ab56"
```

On **2026-10-02**, the live [PyPI package](https://pypi.org/project/mars-risk/) was **0.0.27**. It does not contain all the capabilities shown here. Source 0.0.28 has not been published by this task. The six badges retain their actual dynamic sources.

```bash
pip install mars-risk==0.0.27
```

The base package supports Python **3.8–3.12**, with a frozen dependency stack for 3.8. Public cases were generated with Python 3.11 and are checked with Python 3.12; use **Python 3.10+** for optional modeling, tuning, notebooks, documentation development and the built-in Agent SDK. [Installation guide (Chinese)](https://leeesq.github.io/mars-risk/getting-started/installation/) explains extras and constraints.

To reproduce the new case scripts, use this task branch until it is merged:

```bash
git clone --branch codex/task-cases-bilingual-readme https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --output-dir docs/assets/cases
```

Case sources and committed downloads are available on this branch. New interactive URLs on the documentation site become available when the branch is merged and the existing Pages workflow deploys it. This task does not merge or request an additional deployment.

## A useful first report

Run in an empty working directory. This self-contained example creates a risk summary, an HTML report and a snapshot:

```python
import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame({
    "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
    "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
    "target": [1, 1, 0, 0, 1, 1, 0, 0],
})
report = profile_risk(
    df, target="target", features=["income", "utilization"],
    method="quantile", n_bins=4,
).report
print(report.get_table("summary", sort_by="iv", descending=True, limit=2))
report.write_html("risk_report.html", include_charts=False)
report.save("risk_report.marsreport")
```

[Full tested script](docs/snippets/readme_quickstart.py) · [Quickstart (Chinese)](https://leeesq.github.io/mars-risk/getting-started/quickstart/). The eight-row fixture demonstrates the interface; its IV values are not evidence of real model quality. HTML here omits charts; provide valid date context when calculating time trends.

## One Report for people and external Agents

People can read the HTML. In a **new process**, an external consumer can load the snapshot without the original wide table, analyzer, LLM, API key or `MarsAgentSession`:

```python
from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=1)
next_page = report.query_page("summary", offset=page["next_offset"], limit=1)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)
```

The loaded `report` is a ReportSnapshot; `report.write_excel("risk_report.xlsx")` exports current static tables. The original binning report's Excel method uses the legacy pivot template and may require native Excel refresh; use the loaded snapshot for this static delivery.

`describe()` discovers the actual tables, schema, semantics and supported capabilities. `query_page()` returns a bounded page, counts, a continuation offset and a replayable evidence reference. `to_ai_context()` serializes bounded JSON: **`max_chars` measures final JSON characters, not tokens**. Check clipping and status before answering; an empty result, an unavailable value and a valid zero mean different things.

Loading returns **ReportSnapshot**. It restores report identity, metadata and saved tables; use `show_table()` for generic display. Dedicated score-policy replay uses saved aggregates. Loading does not restore the analyzer, original records, arbitrary methods, a trained model or a deployable RuleSet. New dimensions and individual-record questions require new analysis.

[Persistence and query case](docs/demos/saved-reports.md) demonstrates next pages, empty results, invalid requests and actual budget clipping. [Delivery case](docs/demos/report-delivery.md) lists supported formats and limitations. Excel is a static delivery; HTML interactivity and offline behavior are verified for each actual export. TXT files are prompt materials, and the local deterministic consumer is explicitly identified as a script. Example answers are manually written and checked against the evidence; no LLM experiment is claimed.

## Maturity and development focus

**Analysis, Feature and Reporting are Stable. Rule, Monitoring, Modeling, Pipeline, Scoring and Agent are Experimental.** Maturity and investment are separate.

Current work prioritizes core analysis performance, memory efficiency, human use and public report consumption by external Agents. **Monitoring, Modeling (including Pipeline) and Scoring are paused for feature development.** Existing functionality, documentation and historical cases remain, with correctness fixes, run fixes and direct adaptation to upstream changes. Those downstream modules do not cause upstream compatibility aliases or duplicate computation. This exception does not permit arbitrary changes to core public APIs or saved reports.

[Stability policy (Chinese)](https://leeesq.github.io/mars-risk/project/stability/) · [Retained LightGBM case](docs/demos/history.md) · [Detailed rule and Agent guide](docs/user-guide/rule-reports-and-agents.md).

## Reproduce, verify and contribute

Generation, evidence verification and site building are explicit, separate steps. MkDocs renders preverified resources; `mkdocs-jupyter` uses `execute: false`, which does not execute notebooks. See the [case index](docs/demos/index.md) for the artifact capability matrix, shared downloads and reproducibility details.

```bash
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py -m "not docs_ml"
python -m mkdocs build --strict
```

Development checks, dependencies and actual browser acceptance are documented in [CONTRIBUTING.md](CONTRIBUTING.md). [Case provenance](docs/assets/cases/manifest.json) records seed, scale, source commit, dependency versions and file hashes. [MIT License](LICENSE).
