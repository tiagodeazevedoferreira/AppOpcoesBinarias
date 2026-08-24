# Radical 99 — Current Project State

> Operational handoff document. Update this file whenever the research state changes.

## Snapshot

**Date:** 2026-08-24

**Repository:** `tiagodeazevedoferreira/AppOpcoesBinarias`

**Primary symbol:** `frxEURUSD`

**Primary horizon:** 60 seconds

**Current research branch:** `research/radical-v2-multiscale`

**Current PR:** #7 — `research: discover the radical precision frontier before changing the model`

**PR status at last update:** open; GitHub reported the branch as having merge conflicts at the time of observation, so conflict status must be rechecked before merge.

## Last decisive experiment

Command:

```bash
python -m app_opcoes_binarias.scripts.evaluate_radical_edge
```

Result:

- target 99.0%: 0 decisions;
- target 99.5%: 0 decisions;
- target 99.9%: 0 decisions;
- holdout rows: 115,015;
- walk-forward rows: 315,329;
- five walk-forward folds;
- minimum evidence: 100;
- no-bet rate effectively 100% at the requested frontier.

Interpretation:

The current `state + trend + motif` representation does not produce any state with enough historical evidence and a 99% one-sided Wilson lower-bound gate to execute. This is a **valid abstention result**, not a successful predictive model.

Do not relax the 99% promotion target merely to increase coverage.

## Baseline context

Latest model evaluation showed approximately:

| Model / strategy | OOS accuracy |
|---|---:|
| Majority | 44.84% |
| Softmax | 44.90% |
| Nearest centroid | 37.07% |
| Persistence | 90.96% |
| Regime persistence | 46.18% |

Persistence is unusually strong in this dataset, including approximately:

- 15s: 81.51%;
- 30s: 87.33%;
- 60s: 90.96%;
- 120s: 94.07%;
- 300s: 96.46%.

This makes beating persistence a critical requirement for any new directional model.

## Current implementation state

The Radical Selective Edge currently uses a multi-view state concept:

- coarse regime/state;
- trend state;
- exact short motif;
- consensus across views;
- historical state purity;
- minimum evidence;
- one-sided Wilson lower-bound gate;
- expanding walk-forward validation.

The latest linter issue in `radical_edge.py` was corrected and the relevant validation became green.

A later Ruff failure involving import formatting in:

- `src/app_opcoes_binarias/scripts/evaluate_models.py`;
- `src/app_opcoes_binarias/scripts/evaluate_radical_edge.py`

was also corrected in the development flow.

## Current next experiment

### Radical Precision Frontier

Purpose: discover the **precision frontier** of the existing representation before changing the representation again.

The frontier explores:

- Wilson lower-bound thresholds: 80%, 85%, 90%, 93%, 95%, 97%, 98%, 98.5%, 99%;
- minimum evidence: 20, 50, 100;
- one-view and two-view selection;
- `state`, `trend`, `motif`, and consensus views;
- holdout evaluation;
- five expanding walk-forward folds.

Required outputs:

- best observed OOS accuracy;
- best OOS Wilson lower bound;
- decision count;
- decision rate / coverage;
- maximum evidence;
- best configurations;
- per-fold stability;
- whether any configuration approaches or exceeds 99% without collapsing into statistical insignificance.

## Immediate next action

**Run/inspect the Radical Precision Frontier workflow and return its complete JSON result.**

Do not design the next architecture until this result is analyzed.

## Decision rules after frontier

### Case A — >=99% found with adequate evidence

Create a dedicated specialist detector around the discovered state and validate it on untouched periods.

### Case B — 97–99%

Investigate the state that produces the frontier. Add causal multiscale context and disagreement features specifically around that state.

### Case C — 90–97%

The representation has useful conditional structure but is insufficient for the target. Move to multiscale state representation and specialist models.

### Case D — no stable edge

Stop threshold tuning. Redesign the feature/state space.

## Current workflow state

A dedicated workflow was added:

`.github/workflows/research-radical-frontier.yml`

It is intended to run:

1. unit tests;
2. Ruff;
3. Radical Precision Frontier evaluation.

At the moment the workflow command references:

```bash
python -m app_opcoes_binarias.scripts.evaluate_radical_frontier
```

Before relying on it, verify that the corresponding script exists on the current branch and that the workflow has actually executed successfully. If it does not exist or fails for an implementation reason, fix the research tooling rather than interpreting the absence of a result as scientific evidence.

## PR / Git state

Current PR created for the frontier research:

- PR: #7
- branch: `research/radical-v2-multiscale`
- base: `main`
- title: `research: discover the radical precision frontier before changing the model`

Before merging:

- check merge conflicts;
- ensure unit tests are green;
- ensure Ruff is green;
- ensure the frontier experiment result is available;
- review the scientific interpretation.

## What must NOT happen next

- Do not lower the promotion target just because coverage is zero.
- Do not train on test labels.
- Do not tune a threshold after inspecting the same holdout result and call it OOS.
- Do not promote a model because of a single aggregate accuracy number.
- Do not compare a new strategy only against majority; persistence is the critical baseline.
- Do not interpret zero decisions as zero market edge.
- Do not implement a large new architecture before analyzing the precision frontier.

## Update protocol

Whenever a meaningful event occurs, update this file:

- new workflow result;
- new PR/branch;
- validation failure;
- validation success;
- architecture change;
- new scientific conclusion;
- new blocker;
- next experiment.

For each update preserve the previous conclusion when it remains valid and append the new evidence to the appropriate section.

## Handoff instruction

A new ChatGPT conversation should read:

1. `docs/RADICAL_99_MASTER_PLAN.md`
2. `docs/RADICAL_99_STATE.md`
3. current GitHub PR/branch/workflow state

Then continue from **Immediate next action**. Never restart completed experiments unless there is a scientific reason to replicate them.
