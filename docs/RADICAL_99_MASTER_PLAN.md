# Radical 99 — Master Development Plan

## 1. North Star

The objective of this research program is not to improve an existing classifier from 60% to 65%. The objective is to discover whether the market contains **conditionally predictable states** that can support extremely high precision when the system is allowed to abstain most of the time.

Target frontier:

- primary target: >= 99.0% accuracy on executed out-of-sample decisions;
- stretch targets: >= 99.5% and >= 99.9%;
- coverage is secondary to precision;
- no promotion based on random holdout alone;
- no promotion unless the edge survives expanding walk-forward validation;
- no leakage from future labels into training state tables;
- if the current representation cannot support the target, change the representation rather than relaxing the target.

A 99% target is a **research objective**, not a claim that financial markets can reliably deliver 99% predictive accuracy. Every result must remain empirical and out-of-sample.

## 2. Scientific principle

The project follows a radical-selectivity principle:

> Do not ask how to make every row predictable. Ask whether there is a small, objectively identifiable subset of rows that is extraordinarily predictable.

Therefore the system may abstain. The fundamental optimization is:

**maximize precision subject to statistical validity, while treating coverage as a constrained secondary objective.**

The system must never manufacture 99% accuracy by evaluating only convenient rows after observing their labels.

## 3. Current evidence

Current symbol: `frxEURUSD`.

Current primary horizon: 60 seconds.

Dataset observed in the latest evaluation: 420,224 raw rows.

The conventional models are not competitive:

- softmax walk-forward accuracy: ~44.9%;
- nearest-centroid: ~37.1%;
- majority: ~44.8%.

Persistence is exceptionally strong for this dataset:

- 60s persistence: ~90.96%;
- 120s persistence: ~94.07%;
- 300s persistence: ~96.46%.

The existing Radical Selective Edge attempted a multi-view state consensus (`coarse regime + trend state + exact short motif`) with a one-sided 99% Wilson lower-bound gate and minimum evidence of 100.

Its latest result was:

- target 99.0%: 0 decisions;
- target 99.5%: 0 decisions;
- target 99.9%: 0 decisions.

This is **not evidence that the market has no edge**. It is evidence that the current representation plus statistical gate did not identify any sufficiently evidenced state that could safely be promoted.

## 4. Phase A — Precision Frontier Discovery

### Status

Current next experiment: **Radical Precision Frontier**.

The frontier diagnostic must determine where useful conditional signal lives before another major architecture is invented.

Evaluate:

- training Wilson lower-bound thresholds: 80%, 85%, 90%, 93%, 95%, 97%, 98%, 98.5%, 99%;
- minimum evidence: 20, 50, 100;
- one-view and two-view selection;
- existing views: `state`, `trend`, `motif`, and their consensus;
- holdout and five expanding walk-forward folds.

For every frontier point report:

- observed OOS accuracy;
- OOS Wilson lower bound;
- decision count;
- decision rate;
- maximum historical evidence;
- per-fold behavior;
- stability of the selected state across time.

### Decision tree

If the frontier finds >=99% OOS precision with meaningful evidence:

1. isolate the corresponding state definition;
2. verify it independently on untouched periods;
3. estimate confidence intervals and failure rate;
4. test regime dependence;
5. build a dedicated selective detector.

If it finds 97–99%:

1. identify which views produce the edge;
2. analyze disagreement states;
3. introduce multiscale temporal context;
4. test whether the edge becomes >=99% after conditioning on additional *pre-decision* information.

If it finds 90–97%:

1. treat the representation as useful but insufficient;
2. move to multiscale feature/state discovery;
3. do not promote the current edge as a 99% system.

If it finds no stable conditional edge:

1. preserve the negative result;
2. stop tuning thresholds on the same representation;
3. redesign the feature space.

## 5. Phase B — Multiscale State Representation

If Phase A does not reach the target, build a state representation across multiple causal time scales.

Candidate scales:

- micro: seconds / recent ticks;
- short: 15–30s;
- primary: 60s;
- medium: 2–5m;
- context: 10–30m.

The representation should encode *state*, not simply raw price direction.

Candidate causal features:

- normalized return across multiple windows;
- return acceleration/deceleration;
- realized volatility;
- volatility expansion/compression;
- directional persistence;
- reversal pressure;
- range position;
- tick-arrival intensity;
- spread/liquidity proxies when available;
- temporal/session context;
- agreement/disagreement across scales.

All features must be computed using information available at decision time.

## 6. Phase C — Conditional Edge Discovery

Replace the question “which class does the model predict?” with:

> “Under which causal market state is the conditional probability of the next outcome sufficiently concentrated?”

Research methods may include:

- hierarchical state partitioning;
- conditional probability tables;
- supervised trees with strict temporal validation;
- calibrated probabilistic models;
- mixture-of-experts;
- regime-specific predictors;
- rare-event / selective classification;
- conformal-style abstention where statistically appropriate.

The output must remain a decision policy, not merely a classifier.

## 7. Phase D — Specialist / Expert Architecture

If a high-purity subpopulation exists, create specialists rather than one universal model.

Conceptual architecture:

```text
raw ticks
   |
   v
causal feature engine
   |
   v
market-state detector
   |
   +---- unstable / ambiguous ------> NO BET
   |
   +---- state A --------------------> specialist A
   |
   +---- state B --------------------> specialist B
   |
   +---- state C --------------------> specialist C
                     |
                     v
              confidence gate
                     |
              +------+------+
              |             |
           execute        NO BET
```

The gate must be learned without using future test labels.

## 8. Phase E — Statistical Validation

Every promoted detector must satisfy all of the following:

1. training data ends strictly before validation data;
2. state dictionaries/tables contain no future labels relative to each fold boundary;
3. evaluation is performed on untouched observations;
4. expanding walk-forward validation is used;
5. every required fold satisfies the promotion threshold;
6. confidence intervals are reported;
7. sample size is sufficient for the statistical claim;
8. results are reproducible from a clean checkout.

For the Radical 99 target, the one-sided Wilson lower bound is a primary statistical gate for state purity. A high observed accuracy with insufficient evidence is not promotion-worthy.

## 9. Phase F — Robustness / Adversarial Validation

A candidate edge must survive:

- different time periods;
- different market regimes;
- non-overlapping samples;
- alternative horizons;
- perturbation of thresholds;
- small feature perturbations;
- removal of individual features/views;
- bootstrap or repeated temporal samples where appropriate;
- outlier and stale-data handling;
- realistic execution assumptions.

The goal is to distinguish a structural conditional edge from a dataset artifact.

## 10. Phase G — Economic Validation

Accuracy alone is insufficient for deployment.

For any surviving detector evaluate:

- payout assumptions;
- transaction/operational costs where applicable;
- expected value;
- maximum losing streak;
- drawdown;
- calibration;
- decision frequency;
- sensitivity to execution delay;
- sensitivity to threshold drift.

A 99% accuracy detector with negligible or economically useless coverage is a research curiosity, not necessarily a viable strategy.

## 11. Phase H — Production Promotion Gate

Promotion requires:

- >=99% target accuracy on the defined executed decision set;
- every walk-forward fold meets the required gate;
- statistically defensible evidence;
- no leakage findings;
- robustness tests pass;
- economic validation passes;
- deterministic/reproducible evaluation;
- unit tests and Ruff green;
- explicit human review of the research report.

No model should be promoted merely because it produces an impressive aggregate number.

## 12. Phase I — Continuous Research Loop

After a candidate reaches production quality, continue monitoring:

- rolling precision;
- coverage;
- calibration drift;
- state-frequency drift;
- regime distribution drift;
- unexpected losing clusters;
- data-quality anomalies.

If the statistical gate degrades, the system should abstain more aggressively before attempting retraining.

## 13. Engineering rules

Every implementation must preserve:

- temporal causality;
- deterministic feature generation;
- explicit typing;
- Ruff cleanliness;
- unit-test coverage for new logic;
- small, reviewable commits;
- research workflows separate from production workflows;
- no silent fallback that turns missing data into a valid prediction;
- no promotion code path that bypasses the statistical gate.

## 14. Workflow protocol for future chats

When continuing this project in a new ChatGPT conversation:

1. Read this file first.
2. Read `docs/RADICAL_99_STATE.md` second.
3. Inspect current GitHub branches, PRs, commits and workflow status.
4. Do not repeat completed work.
5. Run or inspect the experiment specified in `RADICAL_99_STATE.md`.
6. Analyze the result before designing the next experiment.
7. Implement only the next justified step.
8. Run unit tests and Ruff.
9. Run the appropriate research workflow.
10. Update `RADICAL_99_STATE.md` with the new evidence and next action.
11. Update this master plan only when the roadmap or scientific strategy changes.

## 15. Non-negotiable research philosophy

The target is radical, but the methodology must be conservative.

**Aggressive objective + conservative validation.**

We are allowed to redesign the entire representation to pursue 99% precision. We are not allowed to redesign the validation rules merely to make 99% appear.
