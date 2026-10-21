[中文](https://github.com/leeesq/mars-risk/blob/main/README.md) / **English**

<p align="center">
<picture>
<source media="(prefers-reduced-motion: reduce)" srcset="https://leeesq.github.io/mars-risk/assets/mars-logo.svg">
<source media="(prefers-color-scheme: dark)" srcset="https://leeesq.github.io/mars-risk/assets/mars-logo-dark.gif">
<img src="https://leeesq.github.io/mars-risk/assets/mars-logo-light.gif" alt="MARS" width="480">
</picture>
<br>
<img src="https://leeesq.github.io/mars-risk/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="480">
</p>
<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8--3.12-364f6b?style=flat-square"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="https://github.com/leeesq/mars-risk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>

<h1 align="center">A high-performance credit risk toolkit for humans and AI agents</h1>
<p align="center"><strong>Data profiling · Binning evaluation · Feature selection · Correlation analysis · Score cross analysis · Rule mining</strong></p>
<p align="center">Analyze credit risk from Pandas or Polars tables, inspect charts, query statistics and export reports.<br>Analysts use native charts to assess risk. AI agents query evidence from the same reports.</p>
<p align="center"><a href="https://leeesq.github.io/mars-risk/getting-started/quickstart/">Start analyzing</a> · <a href="https://leeesq.github.io/mars-risk/demos/">Practical cases</a> · <a href="https://leeesq.github.io/mars-risk/user-guide/external-agents/">Build with AI agents</a> · <a href="https://leeesq.github.io/mars-risk/">Documentation · 中文</a></p>

<p align="center">
<a href="https://leeesq.github.io/mars-risk/assets/cases/binning-native-home-credit.svg">
<img src="https://leeesq.github.io/mars-risk/assets/cases/binning-native-home-credit.png" alt="MARS native binning trends: fixed bins, distributions, bad rate, amount risk and stability" width="1040">
</a>
</p>

<p align="center"><sub>Home Credit EXT_SOURCE_2, exported by MARS from fixed Train bins. APPLY_DATE is simulated. Click for the full-resolution SVG.</sub><br>
<a href="https://leeesq.github.io/mars-risk/demos/binning-stability/">Binning and stability case</a> · <a href="https://leeesq.github.io/mars-risk/assets/cases/binning.html">Open the interactive report</a></p>

## What MARS brings to your workflow

MARS uses Polars for computation and accepts Pandas or Polars wide tables. Check data quality, evaluate risk across bins, select features, analyze correlations and mine rules. Reports retain the results so you can compare distributions and risk across populations and periods, inspect selection decisions, and export charts, HTML or Excel.

External AI agents query the same results through public report APIs. Reports preserve the table catalog, metric definitions, feature metadata and calculation states. Queries return replayable evidence references. Your agent can use these results to continue an analysis and call the model libraries and experiment tools you choose.

| Capability | What you can inspect | Practical example |
| --- | --- | --- |
| Data profiling | Missing and special values, sample and label coverage, distributions | [Check whether the data is ready](https://leeesq.github.io/mars-risk/demos/data-quality/) |
| Binning and risk evaluation | Native trend charts, fixed bins, multi-target IV / KS / PSI, count and amount risk | [Compare discrimination and stability](https://leeesq.github.io/mars-risk/demos/binning-stability/) |
| Feature selection | Candidate decisions and reasons across screening stages | [Understand retained and removed features](https://leeesq.github.io/mars-risk/demos/selection-correlation/) |
| Correlation analysis | Positive and negative correlations, with separate raw-value and WOE results | [Inspect related features](https://leeesq.github.io/mars-risk/demos/selection-correlation/#correlation) |
| Score cross analysis | Risk across joint segments of two external-source scores and their combination conditions | [Compare external-source scores](https://leeesq.github.io/mars-risk/demos/score-cross/) |
| Rule mining · Experimental | Candidate origins, screening, coverage and independent validation | [Review rules with evidence](https://leeesq.github.io/mars-risk/demos/rule-evidence/) |

[All seven cases](https://leeesq.github.io/mars-risk/demos/) share all 356,255 Home Credit application records. Original features and TARGET come from the competition; APPLY_DATE is simulated. Four labeled partitions precede the 48,744 unlabeled competition-test records. [Inputs and five partitions](https://leeesq.github.io/mars-risk/user-guide/home-credit-data/) explain their uses.

## Build your agents on MARS

MARS performs calculations and provides analysis evidence. You define the agent's steps, choose training tools and set the business decision process.

| Your external agent | MARS building blocks | Useful output |
| --- | --- | --- |
| Modeling agent | Data quality, binning, feature selection, correlation and risk evaluation | Select candidate features, inspect removal reasons and pass evaluation results to your chosen training tools |
| Monitoring and diagnosis agent | Grouped profiles, missing and distribution changes, fixed-bin risk trends and stability | Locate changing populations or features, gather evidence and propose hypotheses to validate |
| Rule analysis agent | Candidate rules, hit evaluation, discovery audit and independent validation | Review each candidate's risk, coverage and independent validation results |
| Report analysis agent | Report catalog, business metadata, paginated queries, snapshots and evidence references | Continue querying in a new session and check answers against evidence references |

You implement these agent workflows. MARS also offers an optional <code>mars.agent</code> integration that uses the same public report interface. [External Agent guide · 中文](https://leeesq.github.io/mars-risk/user-guide/external-agents/) · [Rule evidence guide · 中文](https://leeesq.github.io/mars-risk/user-guide/rule-reports-and-agents/).

## Install

**MARS 0.0.28 is published on PyPI**. Install the fixed version:

```bash
pip install mars-risk==0.0.28
```

For the current development branch, install from source:

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

The base package supports Python **3.8-3.12**, with pinned dependencies for Python 3.8. Use **Python 3.10+** for optional modeling, tuning, notebooks, documentation development and the built-in Agent SDK. [Installation and extras · 中文](https://leeesq.github.io/mars-risk/getting-started/installation/).

## Start with one report

Follow the [data preparation guide](https://leeesq.github.io/mars-risk/user-guide/home-credit-data/) to download the three official CSV files into `examples/home-credit-2018/data/raw_data/`. From the repository root, run:

```bash
python docs/snippets/readme_quickstart.py
```

The script runs the code below. If copying it elsewhere, place [home_credit_data.py](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/home_credit_data.py) beside your script and prepare the data paths as described in the guide.

```python
from home_credit_data import prepare_home_credit_data

from mars.analysis import profile_risk

inputs = prepare_home_credit_data()
report = profile_risk(
    inputs.df, target="TARGET", features=["EXT_SOURCE_2", "EXT_SOURCE_3"],
    benchmark_df=inputs.benchmark_df, group_col="dataset_flag", time_col="APPLY_DATE",
    amount_col="AMT_CREDIT", method="quantile", n_bins=4,
    feature_metadata=inputs.feature_metadata, business_context=inputs.business_context,
).report
report.save_risk_trend_images(
    "risk_charts", features="EXT_SOURCE_2", target="TARGET", image_format="png", dpi=300,
)
report.write_html("risk_report.html", chart_embed_mode="inline")
report.save("risk_report.marsreport")
print(report.query_page("summary", features="EXT_SOURCE_2", limit=1)["data"])
```

The script creates `risk_charts/`, `risk_report.html` and `risk_report.marsreport` in the current directory, then prints one summary row. The preparation function reads both application files and adds field descriptions, simulated dates and partitions. You can also use <code>report.save_risk_trend_images()</code> to save native figures as SVG or high-resolution PNG.

[Shared runnable source](https://github.com/leeesq/mars-risk/blob/main/docs/snippets/readme_quickstart.py) · [Full quickstart · 中文](https://leeesq.github.io/mars-risk/getting-started/quickstart/) · [Full trend charts and downloads](https://leeesq.github.io/mars-risk/demos/binning-stability/).

## Give the same result to an AI agent

In a new process, load the saved report and query only the evidence you need:

```python
from mars.reporting import load_report

report = load_report("risk_report.marsreport")
description = report.describe()
page = report.query_page("summary", limit=1)
next_page = report.query_page("summary", offset=page["next_offset"], limit=1)
context_json = report.to_ai_context(tables=["summary"], max_chars=16000)
```

Start with <code>describe()</code> to inspect the table catalog, metric definitions, parameters and states, then use <code>query_page()</code> to read pages of results. Each page includes data and a replayable evidence reference. <code>to_ai_context()</code> produces JSON from the selected tables with a length limit; **<code>max_chars</code> counts characters, not tokens**. Querying saved results requires no LLM, API key, <code>MarsAgentSession</code>, original table or analyzer.

Loading returns a ReportSnapshot containing saved statistics and metadata. Use <code>report.write_excel("risk_report.xlsx")</code> to export static tables or <code>report.show_table("summary", limit=10)</code> to display the summary. A snapshot can answer only questions supported by its saved statistics. New analysis dimensions or questions about original records require the source data and a new calculation. [Save and query results](https://leeesq.github.io/mars-risk/demos/saved-reports/) · [Deliver HTML, Excel, snapshots and Agent materials](https://leeesq.github.io/mars-risk/demos/report-delivery/).

## Development focus

**Analysis, Feature and Reporting are Stable. Rule, Monitoring, Modeling, Pipeline, Scoring and Agent are Experimental.** These labels describe interface maturity.

Current development focuses on faster core analysis, lower memory use, and better report viewing and querying for people and external agents. **Monitoring, Modeling (including Pipeline) and Scoring are paused for feature development.** We retain existing functionality and documentation, make necessary fixes, and update these downstream modules directly when core interfaces change. Core code does not retain compatibility aliases or redundant computation for them.

Combine AI, MARS analysis reports and your chosen model tools to build workflows for your business.

[Stability policy · 中文](https://leeesq.github.io/mars-risk/project/stability/) · [Retained LightGBM example · 中文](https://leeesq.github.io/mars-risk/demos/history/).

## Reproduce and contribute

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[docs]"
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/task-cases
```

[Case downloads and reproducibility](https://leeesq.github.io/mars-risk/demos/#run) · [Contribution guide](https://github.com/leeesq/mars-risk/blob/main/CONTRIBUTING.md) · [Engineering Skill](https://github.com/leeesq/mars-risk/blob/main/.codex/skills/mars-risk-engineering/SKILL.md) · [MIT License](https://github.com/leeesq/mars-risk/blob/main/LICENSE).
