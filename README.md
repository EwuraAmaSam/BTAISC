# Safe RL Experiment Investigator
**Technical Project Brief | BlueDot Impact — Engineering Track**

## 1. Project Overview

Safe RL Experiment Investigator is an interactive tool for analyzing Reinforcement Learning (RL) experiments and flagging potentially interesting or concerning agent behaviors for human investigation. It focuses on identifying unusual patterns in the relationship between task reward and safety cost within individual episodes and across multiple episodes.

The project will build on existing Safe RL environments and experiment data rather than implementing a new RL algorithm from scratch.

## 2. Problem Statement

RL experiments can produce hundreds or thousands of episodes, each containing a sequence of observations, actions, rewards, and safety costs. Aggregate metrics, such as total reward and cumulative safety cost, summarize an agent's performance but may conceal the specific events or behavioral patterns that produced those outcomes.

For example, an agent may achieve high reward by taking unsafe actions, or it may achieve unexpectedly high reward while maintaining low safety costs. Both cases may warrant investigation: the former could reveal reward hacking or specification gaming, while the latter could reveal an effective strategy worth understanding.

Manually inspecting large numbers of trajectories can be time-consuming. The proposed tool aims to help researchers prioritize their attention by automatically surfacing potentially interesting episodes and trajectory segments.

## 3. Proposed Approach

The tool will process existing Safe RL experiment outputs and analyze reward and safety-cost patterns at two levels:

- **Episode level:** Calculate or inspect cumulative reward, cumulative safety cost, episode length, and other available metrics to identify unusual episodes.
- **Trajectory level:** Examine changes in reward and safety cost over time to identify potentially interesting events or segments within an episode.

Initial detection signals may include:
- Unusually high safety costs relative to other episodes.
- Sudden changes or spikes in reward or safety cost.
- High-reward episodes accompanied by high safety costs.
- High-reward episodes with unusually low safety costs.
- Changes in the relationship between reward and safety cost across training runs or episodes.

These signals will be treated as investigation heuristics, not definitive evidence of unsafe behavior or reward hacking.

## 4. Proposed System Components

1. **Data ingestion:** Load experiment logs or trajectory data from a supported Safe RL environment.
2. **Analysis engine:** Calculate episode-level metrics and apply initial anomaly-detection heuristics.
3. **Investigation interface:** Display experiment summaries, flagged episodes, relevant metrics, and the reasons for each flag.
4. **Trajectory inspector:** Allow researchers to examine the sequence of actions, rewards, safety costs, and relevant environment events. Where supported by the available data and environment, provide a visual replay.
5. **Evaluation:** Test whether the detection heuristics identify deliberately selected or otherwise known unusual patterns in the experiment data.

## 5. Scope and Feasibility

The project is intended for approximately 30 hours of engineering work. The initial implementation will prioritize one supported environment and a limited set of detection signals. Training a new RL agent, implementing RL algorithms, and supporting multiple environments are not initial requirements.

A suitable starting point may be Safety-Gymnasium, subject to confirming that its available experiment outputs provide the information needed for the proposed analysis.

## 6. AI Safety Relevance

The project supports the investigation of reward–safety trade-offs and potentially problematic agent behavior. It may help researchers discover cases where optimizing a reward objective appears to conflict with intended safety constraints.

The tool will not claim to determine whether an agent is reward hacking. Instead, it will surface evidence that helps researchers decide what to investigate further.

## 7. Potential Extensions

Possible future directions include comparing multiple agents or training runs, detecting additional behavioral patterns, improving anomaly-detection methods, and supporting more environments.

**Intended outcome:** A working, evaluated prototype that helps users move from aggregate Safe RL metrics to targeted investigation of individual agent trajectories.

## 8. Progress

### Data source

Safety-Gymnasium does not publish episode logs. The first dataset is an existing offline collection from [DSRL](https://github.com/liuzuxin/DSRL), the Safety-Gymnasium PointGoal1 trajectories:

`data/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5`

Source: [YYY-45/DSRL on Hugging Face](https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5). The file is about 698 MB and is gitignored. Re-run the inspection with `python scripts/inspect_pointgoal1.py`.

### What the file contains

The file is one sequence of 2,022,000 transitions, split into **2,022 episodes** by the `timeouts` flags. Every episode is exactly 1,000 steps. There are no early terminations. Each step has a 60-dimensional observation, a 2-dimensional action, a reward, and a safety cost.

| Field | What it looks like |
| --- | --- |
| Episode reward | 0.01 to 30.07. Median 12.71. |
| Episode cost | 0 to 100. Median 24. This is the count of unsafe steps. |
| Step cost | Only 0 or 1. About 3.1% of steps are costly. |
| Zero-cost episodes | 433 of 2,022. |

### Signals already visible

These are investigation heuristics, not evidence of reward hacking.

- **High reward with high cost:** 224 episodes are in the top quarter of reward (at least 19.04) and the top quarter of cost (at least 49). Several of those reach the maximum recorded cost of 100 while still scoring above 20 reward.
- **High reward with low cost:** 54 episodes are in the top quarter of reward and the bottom quarter of cost (6 or below). Some of those finish near reward 28 with cost 0 or 1.
- **Cost inside an episode:** cost does not vary in size. It arrives in runs of consecutive steps. There are 3,956 such runs. The median run is 15 steps, and the longest is 87.
- **Reward inside an episode:** most steps pay a small amount, positive or negative, for progress toward the goal. 13,568 steps pay about 1.0. Those are goals reached. No step pays between 0.05 and 0.9.

### What the data measures

- **Observations:** dims 0–2 are the accelerometer, 3–5 velocity, 6–8 gyro (dim 8 is turn rate), 9–11 compass. Dims 12–27 are goal lidar, 28–43 hazard lidar, 44–59 vase lidar. Each lidar has 16 directions, and a reading of 1 − distance / 3 for the nearest object in that direction.
- **Cost** means the robot is inside a hazard. It matches hazard lidar ≥ 0.933 (within 0.2 units) on 99.97% of steps. Touching or pushing the vase costs nothing.
- **Reward** is progress toward the goal, plus about 1.0 when a goal is reached. A new goal then appears.

### Data caveats

- **The data is curated.** According to the DSRL paper, episodes come from several expert agents trained with different cost limits, and a density filter removed redundant episodes. "Unusual" therefore means unusual within this curated set.
- **No episode has cost above 100**, and the counts stop sharply there. The paper does not explain this cut-off. The worst episodes may be missing.
- **Episode order is not training order.** Episodes run roughly from high-cost to low-cost agents. There are no run, agent or checkpoint IDs, so comparing training runs is out of scope.
- **The first observation of an episode is often stale.** In 474 episodes it is an exact copy of the previous episode's final state. Every episode still has its own fresh layout. The pipeline only uses `next_observations`, the state after each action.
- **No cost occurs in the first 31 steps of any episode.** Episodes start away from hazards. This is a property of the starts, not of the agents.

### Pipeline

Run in this order. Outputs go to `data/processed/` (gitignored).

| Script | Output | What it does |
| --- | --- | --- |
| `scripts/build_episode_index.py` | `episodes.parquet` | Where each episode starts and stops, and whether its first observation is stale. |
| `scripts/build_paths.py` | `paths.parquet`, `map.parquet` | Rebuilds the robot's path from compass and velocity (0.02 s per step), then places the hazards, goals and vase from lidar. |
| `scripts/build_events.py` | `steps.parquet`, `events.parquet` | Readable per-step columns, and events: hazard stays, near misses, goals reached, moving away from the goal, near the vase, vase pushed. |
| `scripts/build_flags.py` | `features.parquet`, `flags.parquet` | Episode features, rule-based episode flags with reasons, cut-through moment markers, and an Isolation Forest unusualness score with a reason. |

### The app

Install the libraries with `pip install -r requirements.txt`, run the four pipeline scripts in the order above, then start the app with `streamlit run app/app.py`.

- **Overview:** a reward-vs-cost scatter of every episode, coloured by flag category, and the flag list with a reason for each flag. A row or dot opens the episode.
- **Inspector:** the rebuilt map, a timeline of closeness, cost and reward with events marked, and a radar of what the robot sensed at the chosen step. A flag button jumps to the flagged moment, with 20 steps of context either side.
- **Labelling:** blind labelling for evaluation. Flags, scores and cut-through markers are hidden. Labels are saved to `evaluation/labels.csv`, for a fixed sample in `evaluation/label_sample.csv`.

**Path and map check:** goal position estimates agree to within 0.004 units. During 96% of costed steps the robot is within 0.2 units of a located hazard, compared with 0.08% of other steps. This replaces visual replay, which is not possible because the layouts were not saved.

### Findings so far

- **"Goal soon after a hazard" is not a useful flag.** Within 30 steps, a goal follows 76% of hazard stays, 60% of near misses, and about 25% of random moments. The robot is almost always heading to a goal, so hazards are met on the way. Entering instead of avoiding gives a modest 1.22× within the same episodes, rising to 1.40× in low-cost episodes. See `experiments/a10_goal_after_hazard.py`.
- **The vase gets pushed in about a quarter of episodes**, up to 1.9 units, at no cost. This is a side effect the cost signal does not capture.

### Related work

A short search found no existing tool that flags reward–cost incidents in recorded safe RL episodes and explains why. The nearest work: visual tools for RL behaviour (DRLViz, VISITOR, MARLViz), policy summaries that show important moments (HIGHLIGHTS), trajectory anomaly detection with learned models, and reward-hacking detectors, mostly for LLM agents.
