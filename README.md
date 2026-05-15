# comp5400-stigmergic-sorting

Before the structure, one reframe worth doing now: the research question as your teammate phrased it ("Can evolved neural controllers outperform classical hand-designed stigmergic sorting?") has a binary trap. It pushes toward a single answer where the more defensible — and more publishable — finding is conditional: *under which regimes does each approach dominate, and what does the boundary tell us*. Suggested introduction framing:

> We characterise the design tradeoff between local hand-designed rules and evolved neural controllers for collective sorting, identifying regimes in which each approach dominates and the failure modes that delineate the boundary.

This makes every outcome reportable — including "baseline matches or beats evolved," which is the most interesting case and the one most likely under tight evolution budgets. Reframing now saves a rewrite at the end.

---

## Repo skeleton — lock Day 0

```
stigmergy-evo/
├── docs/
│   ├── report.tex
│   ├── refs.bib
│   └── figures/
├── src/
│   ├── arena.py        # world, pellets, physics
│   ├── agents.py       # base agent, sensors, actuators
│   ├── baseline.py     # Deneubourg rule set
│   ├── controller.py   # NN forward pass
│   ├── evolution.py    # GA loop
│   ├── metrics.py      # cluster purity, count, stability
│   ├── experiment.py   # runner, CSV logging
│   └── viz.py          # matplotlib/pygame
├── experiments/
│   ├── configs/        # one YAML per run
│   └── results/        # seeded CSV outputs
├── tests/
└── README.md
```

Non-negotiables from day one: **seeded RNG everywhere** (numpy, python `random`, GA init), **CSV logging on every run** (not just final number — full time-series), **one config file per experiment**. Without these the results are not reproducible and the marker can tell.

---

## Milestone structure

Each milestone has four parts: **Gate** (what defines done), **Parallel tracks** (who works on what concurrently), **Failure-yield** (what insight you extract if it doesn't work), **Report.tex commit** (which section gets written, what data).

### M0 — Shared mental model (½ day)

**Gate.** All three members can independently describe, on a whiteboard, the environment, the Deneubourg rule, and the evolved controller's input/output spec without referring to notes.

**Parallel tracks.** None — this is a synchronisation point.

**Output artefact.** A 1-page `docs/env_spec.md` covering: arena dimensions, pellet count and colour count, agent count, sensor radius, action space, tick rate, episode length, RNG seeding policy.

**Failure-yield.** If you can't converge on the spec in half a day, you've surfaced ambiguity that would have caused integration pain at M3. Document the disagreement and its resolution — that becomes a Methods note about design rationale.

**Report.tex commit.** Section skeleton: Abstract, Introduction, Background, Methods (Environment / Baseline / Metrics / Evolved Controller), Results, Discussion, Limitations, Conclusion, AI declaration. Drop the agreed env spec into `Methods/Environment` as bullet-point placeholder — to be turned into prose at M1.

### M1 — Arena + pellets + headless run (1 day)

**Gate.** `python -m experiment --config configs/sanity.yaml` runs N agents on pure random walk in the arena for T ticks with K static pellets. Viz renders. CSV log written with `(tick, agent_id, x, y, carrying, action)` columns.

**Parallel tracks.**
- A: `arena.py`, `viz.py`, `experiment.py` runner skeleton
- B: starts `metrics.py` — define cluster purity and cluster count using DBSCAN or connected components over pellet positions; write tests against three hand-built synthetic configs
- C: starts `controller.py` (NN forward pass, no learning yet) and `evolution.py` skeleton (population, tournament selection, mutation — fitness function deferred)

**Failure-yield.** If movement looks wrong (agents stuck at walls, tunneling, jitter), you've found physics bugs at the cheapest possible moment. The resolution — discretised vs continuous space, collision handling, boundary policy — is a paragraph in `Methods/Environment`.

**Report.tex commit.** First figure: snapshot of initial arena state. Caption with parameters. Lives in `Methods/Environment`. The Environment subsection prose can now be written — concrete, easy, anchors everything that follows.

### M2 — Pickup/drop + Deneubourg baseline (2 days)

**Gate.** Agents running Deneubourg produce visible clustering within ~10 000 ticks for a 2-colour, 200-pellet, 20-agent config. "Visible" = number of clusters drops by ≥50% from random initial placement.

Deneubourg rule (so it's not hunted down later):
- P(pickup) = (k₁ / (k₁ + f))² when not carrying
- P(drop) = (f / (k₂ + f))² when carrying
- f = fraction of similar pellets in local window of radius r

Starting points: k₁ ≈ 0.1, k₂ ≈ 0.3, r = 3–5 cells. Calibrate.

**Parallel tracks.**
- A: pickup/drop mechanics in arena, local sensing window
- B: `baseline.py` finalised, metrics validated and unit-tested
- C: GA loop infrastructure, NN architecture lock — recommended: input = local pellet density per colour over radius r + one-hot carrying state; hidden = 1 layer of 8–16 units, tanh; output = turn-angle (continuous) + pickup-or-drop logit

**Failure-yield.** If clustering doesn't emerge after a serious (k₁, k₂) sweep, that *is* a result — you've mapped the parameter sensitivity boundary of Deneubourg's model. Sweep, log, and report the regime where clustering works. This is your first real finding regardless of which side it lands.

**Report.tex commit.** Two sections become writable now:
- `Background — Stigmergic Sorting`: write the model description with equations, cite Deneubourg (1990) as the model source, Franks & Sendova-Franks (1992) for biological grounding in ant brood sorting, Grassé (1959) for the original stigmergy concept (define the term), Dorigo et al. (1996) to bridge to swarm-algorithmic framing.
- `Methods — Baseline`: your specific implementation choices and parameter values.
- Figure: baseline cluster formation at t=0, t=mid, t=end. Three-panel.

**Hard branch point.** If the baseline never clusters even after parameter exploration, *stop and write* a "Calibration of Deneubourg's Model" subsection with the parameter sweep heatmap. You have a complete piece of work even if the comparison study never happens.

### M3 — Metrics validation (½ day, overlaps with end of M2)

**Gate.** Each metric returns expected values within ±5% on three synthetic configurations: (i) perfectly sorted, (ii) fully random, (iii) hand-placed intermediate state.

**Parallel tracks.** B owns this; A and C continue on their threads.

**Failure-yield.** Metrics disagreeing with intuition is the most common silent killer of comparison studies. Catching this here means headline numbers are trustworthy when reviewers prod them.

**Report.tex commit.** `Methods — Evaluation Metrics` with formal definitions of each. Include a small validation table — markers reward visible methodology rigour.

### M4 — GA closed-loop, first evolved population (1.5 days)

**Gate.** GA runs end-to-end: random population of ~30 weight vectors, each evaluated for one episode, fitness = `cluster_purity − λ·cluster_count` or similar composite, tournament selection, Gaussian mutation (σ ≈ 0.1), no crossover initially. Best individual at generation 50 exceeds best initial fitness by ≥20%.

**Parallel tracks.**
- C: GA core, fitness function tuning
- A: viz upgrade — overlay live metric values, save short videos of best individuals per generation for the presentation
- B: experiment runner with multi-seed support (≥5 seeds per condition), CSV aggregator, plotting utility

**Failure-yield.** Three distinct failure modes, each yielding a Discussion paragraph:
- *Flat fitness landscape* → the task needs fitness shaping (reward intermediate clustering states, not only end-state purity). The shaping itself is a methodological contribution.
- *High inter-seed variance* → single-episode evaluation is too noisy; switch to averaging over k episodes. Reportable as evaluation methodology.
- *Premature convergence* → selection pressure / mutation rate are imbalanced. Reportable as GA hyperparameter sensitivity.

You can't really lose this milestone — every failure mode is a finding.

**Report.tex commit.** `Methods — Evolved Controller` becomes writable: NN architecture, GA hyperparameters, fitness function with justification. Cite Stanley & Miikkulainen (2002) for NEAT context — and explicitly justify the simpler weight-evolution choice if you don't use NEAT (likely time-driven). Cite Trianni / Groß / Dorigo as evidence that neuroevolution works in collective tasks. First fitness-over-generations plot lands here.

M4 isn't enough for the most failure-prone part of the project. Fitness design is where most neuroevolution attempts on collective tasks die quietly. Proper treatment:
The core problem. Terminal cluster purity as the only signal is too sparse. An agent has to chain three behaviours — wander → pick up isolated pellet → drop near similar pellet — before any fitness signal exists. Random networks almost never produce that chain in generation 0. Most of the initial population scores at random-floor and selection has nothing to grip. This is the most common reason neuroevolution on sorting tasks looks "broken" — it's not broken, it's flat.
Candidate components (each normalised to [0,1] before weighting):

Cluster purity at episode end — what you actually care about. Terminal target.
Cluster count reduction = 1 − (final_clusters / initial_clusters) — rewards consolidation.
Time-integrated purity = (1/T) Σₜ purity(t) — denser signal than terminal-only. Cheap to compute, often the single biggest fix.
Pickup–drop activity = number of pickup+drop events normalised by ceiling — bootstrap reward. Critical early; decay across generations.
Locality bonus = average colour-similarity of pellets within radius r of each drop event — rewards good drops even before global clusters consolidate.

Composite formulation. Use linear weighting because it's transparent and ablatable:
F = α · purity_terminal + β · purity_integrated + γ · consolidation + δ · activity_decayed
Starting weights: α=0.4, β=0.3, γ=0.2, δ=0.1, with δ decaying linearly to 0 by generation 25. Avoid multiplicative composites — they zero out when any component is zero, which is exactly the failure mode you're trying to escape.
Pickup/drop bootstrap. Two-pronged:

Initialise the pickup-output bias slightly positive (~+0.5) so generation 0 actually picks things up. Document this in Methods — it's a legitimate warm-start, not a cheat.
Keep δ (activity term) active for the first ~25 generations, then switch it off. Gets the population off the floor; once agents are interacting with pellets, the purity-shaped terms take over.

Per-episode noise. Single-episode fitness is too noisy. Evaluate each genome on k=3 episodes with different seeds, take the mean. Cost: 3× compute. Benefit: usable selection signal. Non-negotiable — if you skip this, fitness curves will be jagged and inter-seed variance will swamp any real learning.
Reference-frame fitness. Compute Deneubourg's performance distribution on the same seed set once, store as baseline_purity_distribution. Then report evolved performance as Δ over baseline rather than absolute. "Evolved exceeds baseline by X" is a sharper sentence than "evolved reaches Y purity" and matches the comparative framing of the research question.
Where this lives in the milestones:

M4 implementation. The composite above as default. Hyperparameters in configs/fitness_default.yaml — version-controlled so the report can cite the exact values.
M4 failure-yield (expanded). If fitness plateaus or oscillates, run a fitness ablation — turn off one component at a time, re-evolve. The ablation itself is a result. A figure showing "contribution of each fitness component to final performance" is publishable content even if the final controller doesn't beat the baseline.
Report.tex commit. Methods/Evolved Controller gets a Fitness Function subsection with: the formula, normalisation scheme, weights, one-sentence justification per component, and the bootstrap rationale. The ablation, if run, goes in Results.

One thing to resist. The urge to keep tuning α, β, γ, δ until evolution "works." That's reward hacking by proxy — you tune until the curve looks publishable, then you can't honestly defend the design choices. Pick the weights a priori with the reasoning above, run with them, report what happens. If results are poor, that's the finding and it's defensible. 

### M5 — Head-to-head experiment (1 day) — **the critical gate**

**Gate.** Experiment 1 runs with ≥5 seeds per condition on the 2-colour baseline config. CSV results exist for both baseline and evolved. Statistical comparison computed (Mann–Whitney U for purity with effect size; Welch's t if you prefer parametric — justify either way).

**Parallel tracks.**
- A: orchestrates runs, manages compute time
- B: aggregation, plotting (boxplots over seeds, time-series of metrics during episodes)
- C: continues controller tuning if needed; if M4 closed cleanly, begins on extension experiments

**Failure-yield.** All three outcomes are publishable:
- *Evolved wins* — expected; discuss what the learned representation captures beyond Deneubourg's local rule (likely: state-dependent action selection, e.g. modulating drop behaviour based on what's already in hand-vs-environment).
- *Baseline wins* — the most interesting outcome. Argue: minimal hand-designed local rules with good parameter choice are highly efficient, and evolution under tight budgets struggles to discover them. This is a direct line back to Brooks ("Intelligence without Representation") — cite if you take this angle.
- *Tie* — the task lies in the regime where simple rules suffice; you've identified the regime. Frame as a finding about task complexity vs controller complexity.

**Report.tex commit.** `Results — Experiment 1: Baseline vs Evolved` with the headline figure (boxplots of cluster purity by condition across seeds) and a results table. **This is the report's centerpiece. Until this section exists in prose, nothing else matters.** If M5 closes, the project is a complete deliverable regardless of what M6 produces.

### M6 — Extensions (time-budget gated, variable)

Three candidates, pick by remaining time:

| Experiment | Cost | Yield | Notes |
|---|---|---|---|
| **E3: environment complexity** (2 vs 3 colours, pellet density, arena size) | Low (config sweep) | Medium–high | Scaling laws are always quotable. Do this first. |
| **E2 (truncated): MLP-small vs MLP-medium** | Medium | Medium | Skip CTRNN under time pressure — it's a separate calibration project. |
| **Deneubourg parameter ablation** (k₁, k₂ heatmap) | Low | Medium | Strengthens baseline characterisation. |

Recommendation: E3 first (cheapest, broadest finding), then the Deneubourg ablation if time remains. The architecture comparison is genuinely risky under time pressure because it requires proper per-architecture hyperparameter control — without that, a "Medium MLP underperforms" result means nothing (was it the architecture or the hyperparameters?).

**Parallel tracks.** Each member runs a different experiment config; results converge in `experiments/results/`.

**Failure-yield.** Whichever experiment completes becomes content. Partial completion is fine — frame as "preliminary investigation."

**Report.tex commit.** One `Results` subsection per completed extension experiment.

### M7 — Discussion + polish (1 day, hard deadline)

**Gate.** Report reads end-to-end without gaps. All figures captioned. Every citation resolves. Limitations section is honest about what wasn't done — markers reward honest limitations, they penalise omissions presented as completions.

**Parallel tracks.**
- A: code cleanup, README with reproduction instructions, makefile, zip
- B: report prose polish, figure consistency pass, abstract written last
- C: presentation slides — note that the slides cannot use GenAI per the coursework rules, only the report skeleton can have used it

**Report.tex commit.** Discussion (connect back to biological inspiration: ants don't do gradient descent — what does the comparison say about the relationship between evolved and rule-based collective behaviour?), Limitations, Conclusion, AI declaration paragraph.

---

## Report.tex section build order

Writing order ≠ reading order. Write in this sequence:

1. `Methods/Environment` — M1
2. `Methods/Baseline` — M2
3. `Methods/Metrics` — M3
4. `Methods/Evolved Controller` — M4
5. `Results/Experiment 1` — M5 (centerpiece)
6. `Results/Extensions` — M6
7. `Background` — written *after* Methods, so your literature review is shaped by what you actually used. No orphan citations.
8. `Introduction` — written second-to-last, to match the report you actually wrote rather than the one you intended.
9. `Discussion / Limitations / Conclusion` — M7
10. `Abstract` — last.

This order also means your most important section (M5 Results) is written when you're freshest on the data, not at 2am the night before submission.

---

## AI declaration

Under Amber: name the tool, list specifically what it was used for (e.g. "project structuring, literature search assistance, code scaffolding for the arena module"), and state your verification steps — running the code, inspecting outputs against expected behaviours, manual review of any AI-suggested implementation choices. One paragraph. Write it last but don't forget it — missing declarations are a flat penalty.

---

## Operational notes

- **One person owns merging.** Pick them now. PR turnaround ≤12 hours.
- **CSV-first, plots-second.** Raw data on disk from every run. Plots are derived and re-derivable. A broken plot at midnight is fixable in the morning if the CSV exists.
- **Daily 15-minute standup for the first three days post-M0**, then async by default — the parallel structure is designed so nobody blocks anyone.
- **Drop a placeholder sentence into every report.tex section now.** "To be written after Mn." This makes the document feel non-empty and gives every working session a visible place for progress.

---

## What to do right now

1. 90-minute call: lock the M0 environment spec with all three of you present. Don't start any code before this is signed off.
2. Push the repo skeleton above.
3. Drop the section placeholders into `report.tex`.
4. Decide who owns merging.

If you want, I can draft the `report.tex` skeleton with section commands, placeholder bullets, and a starter `refs.bib` populated with the references you listed — that would give the team a more substantial-feeling starting state than the blank file you have now.
