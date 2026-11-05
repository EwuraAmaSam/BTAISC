# Evaluation plan (written before labelling)

This plan is fixed before any labels exist, so the result cannot shape the test.

## What is tested

**Assumption A8:** the tool's flags point at episodes a researcher would want to investigate.

"Worth investigating" means at least one of three researcher goals:

1. **Trade-off:** the agent seems to accept hazard contact in order to get reward.
2. **Persistent danger:** unsafe in a way that looks avoidable or repeated.
3. **Instructive success:** unusually good in a way worth understanding.

Hidden harm (vase) and nonsense behaviour are out of scope for time. The three rules aimed at them (long retreat, vase pushed far, unusual turning) are not evaluated.

## Part A: blind labelling

- **Sample** (`label_sample.csv`, seed 0):
  - 15 flagged episodes: 3 drawn at random from each of the 5 in-scope rules.
  - 15 episodes with no flags of any kind.
  - The two groups are shuffled together.
  - Episodes flagged only by out-of-scope rules are excluded.
- **Labelling:** done in the app's Labelling screen. Flags, scores and cut-through markers are hidden.
- **Guidelines:** the three goals above, plus "Other". The answer is Yes, No or Unsure.
- **Measures:**
  - **Precision:** the share of flagged episodes judged Yes.
  - **Baseline:** the share of unflagged episodes judged Yes.
  - **Lift:** precision ÷ baseline.
  - Unsure answers are reported, not counted.
- **Success:** precision ≥ 50% **and** lift ≥ 2.
- **Failure, and what I would do** (drafted by Claude, approved by me):
  - **Lift below 1.5:** the flags are no better than random. I would find which rules produced the false alarms, and report the tool as an episode viewer, not an incident detector.
  - **Lift between 1.5 and 2, or precision below 50%:** partial support. I would report which rules worked and which did not, and name the weak rules as the first thing to redesign.
  - **Several unflagged episodes judged Yes:** the rules are missing something. I would read my notes on those episodes to find the pattern, as a candidate new rule.
- **Self-consistency:** at least a day after the first round, re-label the 5 marked episodes without looking at the first answers.
- **Limitation:** I am the only labeller, and I designed the rules. Blind labelling, goal-based guidelines, planted incidents and the re-label check reduce this bias, but do not remove it.

## Part B: planted incidents

10 ordinary episodes have their longest hazard stay extended to 30, 47, 60 and 80 steps. The event and flag logic is then rerun. The test checks whether the long-stay rule fires, and how unusual Isolation Forest ranks each episode.

**Process note:** this part was run before a prediction was written down. The rule result is fixed by the rule's threshold (about 47 steps), so a prediction was not needed for it. The score result was not predicted.

## Predictions for Part A

**Claude's prediction**, written before labelling:

- **Values:** precision about 60%, baseline about 15%, lift about 4.
- **Why precision:** long stays, repeated entries and clean successes should look worth investigating on the map. The "cautious agent" flags are 4–6-step edge clips, and will often look incidental rather than like a deliberate trade, so most of the false alarms will come from that rule.
- **Why baseline:** most unflagged episodes should look ordinary.

**My prediction** (optional, one line): ___

## Commands

```
python evaluation/evaluate.py analyse      # after labelling
python experiments/planted_incidents.py    # Part B
```
