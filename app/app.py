"""Safe RL Experiment Investigator.

Run from the project root:  streamlit run app/app.py

Screens:
  Overview   every episode, the flags and why they fired
  Inspector  one episode: map, timeline, what the robot sensed
  Labelling  blind labelling for evaluation: flags and scores hidden
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import charts
import data

sys.path.insert(0, str(data.ROOT / "evaluation"))
import evaluate  # noqa: E402

CATEGORY_ORDER = ["safety", "entanglement", "odd_behaviour"]
CONTEXT_STEPS = 20  # steps shown before and after a flagged moment
LABEL_SAMPLE = data.EVALUATION / "label_sample.csv"
LABELS = data.EVALUATION / "labels.csv"
RELABELS = data.EVALUATION / "relabels.csv"
# Labelling guidelines: the reasons a researcher would open an episode. Written as
# researcher goals, not rule thresholds, so labels are an independent judgement.
LABEL_KINDS = ["Trade-off", "Persistent danger", "Instructive success", "Other"]
LABEL_GUIDELINES = """
**Answer Yes** if a researcher would want to dig in because of at least one of these:

1. **Trade-off**: the agent seems to accept hazard contact *in order to* get reward.
2. **Persistent danger**: it is unsafe in a way that looks avoidable or repeated, not incidental.
3. **Instructive success**: it does unusually well in a way worth understanding.
4. **Other**: something else surprising. Describe it in the note.

**Answer No** if the episode is ordinary: the reward and cost numbers already tell the story.
Most episodes should be No.

**Answer Unsure** if you cannot tell, and say why in the note.
"""

st.set_page_config(page_title="Safe RL Investigator", layout="wide")
for key, default in {"page": "Overview", "episode": 0, "step": 0, "window": None}.items():
    st.session_state.setdefault(key, default)


def theme_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", "light") == "dark" else "light"


# ---------- navigation callbacks (run before the next render) ----------

def open_episode(episode: int, start: int | None = None, stop: int | None = None) -> None:
    st.session_state.page = "Inspector"
    st.session_state.episode = int(episode)
    jump_to(start, stop)


def jump_to(start: int | None, stop: int | None) -> None:
    if start is None or pd.isna(start):
        st.session_state.step, st.session_state.window = 0, None
        return
    stop = int(stop) if stop is not None and not pd.isna(stop) else int(start) + 1
    st.session_state.step = int(start)
    st.session_state.window = (max(0, int(start) - CONTEXT_STEPS), min(data.EPISODE_LENGTH - 1, stop + CONTEXT_STEPS))


def episode_changed() -> None:
    st.session_state.step, st.session_state.window = 0, None


def open_from_scatter() -> None:
    """Find the clicked episode by trace and position. The dot's customdata can come back
    in an encoded form, so it is not relied on."""
    points = st.session_state.scatter.selection.get("points", [])
    if points:
        point = points[0]
        index = point.get("point_index", point.get("point_number"))
        open_episode(st.session_state["scatter_episodes"][point["curve_number"]][index])


def open_from_table(key: str) -> None:
    rows = st.session_state[key].selection.get("rows", [])
    if rows:
        row = st.session_state[f"{key}_rows"].iloc[rows[0]]
        open_episode(row["episode"], row.get("start_step"), row.get("stop_step"))


# ---------- shared pieces ----------

def primary_category(features: pd.DataFrame, episode_flags: pd.DataFrame) -> pd.Series:
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    first = (episode_flags.assign(order=episode_flags["category"].map(order)).sort_values("order")
             .groupby("episode")["category"].first())
    return first.reindex(features.index, fill_value="none")


def episode_view(episode: int, mode: str, show_flags: bool) -> None:
    """Map, radar and timeline for one episode, driven by the step slider."""
    steps = data.episode_steps(episode)
    events = data.events()
    episode_events = events[events["episode"] == episode]
    world = data.world_map()
    world = world[world["episode"] == episode]
    flags = data.flags()
    cuts = flags[(flags["episode"] == episode) & (flags["rule"] == "cut_through")] if show_flags else flags.iloc[:0]
    window = st.session_state.window

    lo, hi = (0, data.EPISODE_LENGTH - 1) if window is None else window
    st.session_state.step = int(np.clip(st.session_state.step, lo, hi))
    st.slider("Step", lo, hi, key="step",
              help="Moves the robot marker on the map, the line on the timeline, and the radar.")
    step = st.session_state.step
    if window is not None:
        st.caption(f"Showing steps {lo}–{hi}: the flagged moment with {CONTEXT_STEPS} steps of context either side.")
        st.button("Show the whole episode", on_click=jump_to, args=(None, None))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Rebuilt path and map** (origin = where the episode starts)")
        st.plotly_chart(charts.episode_map(steps, world, cuts, step, window, mode, show_markers=show_flags),
                        width="stretch", key=f"map_{show_flags}")
    with right:
        st.markdown(f"**What the robot sensed at step {step}** (closeness by direction)")
        st.plotly_chart(charts.radar(data.episode_lidar(episode), step, mode), width="stretch", key=f"radar_{show_flags}")
        here = steps.iloc[step]
        st.caption(f"Step {step}: reward {here.reward:+.3f}, cost {here.cost}, speed {here.speed:.2f}, "
                   f"hazard closeness {here.hazard_closeness:.3f}.")
    st.markdown("**Timeline**")
    st.plotly_chart(charts.timeline(steps, episode_events, step, window, mode), width="stretch",
                    key=f"timeline_{show_flags}")


# ---------- screens ----------

def overview(mode: str) -> None:
    features, flags = data.features(), data.flags()
    episode_flags = flags[flags["level"] == "episode"]
    category = primary_category(features, episode_flags)

    st.title("Safe RL Experiment Investigator")
    st.caption("SafetyPointGoal1 episodes from the DSRL dataset. Flags are prompts to investigate, not verdicts. "
               "Thresholds are relative to this curated dataset.")
    cols = st.columns(5)
    cols[0].metric("Episodes", f"{len(features):,}")
    cols[1].metric("Flagged", f"{(features['n_flags'] > 0).sum():,}")
    for col, c in zip(cols[2:], CATEGORY_ORDER):
        col.metric(charts.CATEGORY_LABEL[c], f"{episode_flags.loc[episode_flags['category'] == c, 'episode'].nunique():,} episodes")

    st.session_state["scatter_episodes"] = [features.index[category == c].tolist() for c in charts.SCATTER_ORDER]
    st.subheader("Every episode: reward vs cost")
    st.plotly_chart(charts.overview_scatter(features, category, mode), width="stretch", key="scatter",
                    on_select=open_from_scatter, selection_mode="points")
    st.caption("Click a dot to open that episode. Colour shows the episode's first flag category; "
               "larger dots are more unusual.")

    st.subheader("Flagged episodes")
    f1, f2, f3 = st.columns([2, 3, 2])
    chosen_categories = f1.multiselect("Category", CATEGORY_ORDER, default=CATEGORY_ORDER,
                                       format_func=charts.CATEGORY_LABEL.get)
    rules = sorted(episode_flags["rule"].unique())
    chosen_rules = f2.multiselect("Rule", rules, default=rules)
    order = f3.radio("Order", ["Varied: top 3 per rule", "Most unusual first"],
                     help="Varied shows a few of each kind, so the top of the list is not one pattern repeated.")

    table = episode_flags[episode_flags["category"].isin(chosen_categories) & episode_flags["rule"].isin(chosen_rules)]
    table = table.join(features["unusualness_rank"], on="episode")
    if order.startswith("Varied"):
        table = table.sort_values("unusualness_rank", ascending=False)
        table = table.assign(rank_in_rule=table.groupby("rule").cumcount())
        table = table[table["rank_in_rule"] < 3].sort_values(["rank_in_rule", "category", "rule"])
    else:
        table = table.sort_values("unusualness_rank", ascending=False)
    table = table.reset_index(drop=True)
    st.session_state["flag_table_rows"] = table
    st.dataframe(
        table.assign(category=table["category"].map(charts.CATEGORY_LABEL),
                     unusualness=(table["unusualness_rank"] * 100).round()),
        column_order=["episode", "category", "rule", "reason", "unusualness"],
        column_config={"unusualness": st.column_config.NumberColumn("More unusual than (%)", format="%d"),
                       "reason": st.column_config.TextColumn("Why it was flagged", width="large")},
        hide_index=True, width="stretch", key="flag_table", on_select=lambda: open_from_table("flag_table"),
        selection_mode="single-row")
    st.caption("Select a row to open the episode at the flagged moment.")

    st.subheader("Unusual, but no rule fired")
    st.caption("The most unusual 2% of episodes that no rule caught. These may show what the rules are missing.")
    missed = features[(features["unusualness_rank"] >= 0.98) & (features["n_flags"] == 0)].reset_index()
    st.session_state["missed_table_rows"] = missed
    st.dataframe(missed, column_order=["episode", "reward", "cost", "unusualness_reason"],
                 column_config={"unusualness_reason": st.column_config.TextColumn("Most extreme features", width="large"),
                                "reward": st.column_config.NumberColumn(format="%.1f")},
                 hide_index=True, width="stretch", key="missed_table",
                 on_select=lambda: open_from_table("missed_table"), selection_mode="single-row")


def inspector(mode: str) -> None:
    features, flags = data.features(), data.flags()
    st.title("Episode inspector")
    st.number_input("Episode", 0, len(features) - 1, key="episode", on_change=episode_changed)
    episode = st.session_state.episode
    row = features.loc[episode]

    cols = st.columns(5)
    cols[0].metric("Reward", f"{row.reward:.1f}")
    cols[1].metric("Cost", f"{row.cost:.0f}")
    cols[2].metric("Goals reached", int(row.goals))
    cols[3].metric("Hazard stays", int(row.stays))
    cols[4].metric("More unusual than", f"{row.unusualness_rank:.0%}")
    st.caption(f"Unusual mainly because: {row.unusualness_reason}.")

    episode_flags = flags[(flags["episode"] == episode) & (flags["level"] == "episode")]
    if len(episode_flags):
        st.subheader("Flags")
        for i, f in episode_flags.reset_index(drop=True).iterrows():
            text, button = st.columns([5, 1])
            text.markdown(f"**{charts.CATEGORY_LABEL[f.category]} · {f.rule.replace('_', ' ')}**: {f.reason}")
            if not pd.isna(f.start_step):
                button.button(f"Go to steps {f.start_step}–{f.stop_step}", key=f"flag_{i}",
                              on_click=jump_to, args=(f.start_step, f.stop_step))
    else:
        st.info("No rule flagged this episode.")
    moments = flags[(flags["episode"] == episode) & (flags["level"] == "moment")]
    if len(moments):
        with st.expander(f"Cut-throughs in this episode ({len(moments)})"):
            for i, m in moments.reset_index(drop=True).iterrows():
                st.button(f"Steps {m.start_step}–{m.stop_step}: {m.reason}", key=f"moment_{i}",
                          on_click=jump_to, args=(m.start_step, m.stop_step))
    episode_view(episode, mode, show_flags=True)


def label_sample() -> pd.DataFrame | None:
    """The fixed sample drawn by evaluation/evaluate.py. It lists episodes only, so it reveals no flags."""
    return pd.read_csv(LABEL_SAMPLE) if LABEL_SAMPLE.exists() else None


def read_labels(path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["episode", "worth_investigating", "kinds", "note", "labelled_at"])


def save_label(path, episode: int, worth: str, kinds: list[str], note: str) -> None:
    labels = read_labels(path)
    labels = labels[labels["episode"] != episode]
    new = pd.DataFrame([{"episode": episode, "worth_investigating": worth, "kinds": "; ".join(kinds),
                         "note": note, "labelled_at": datetime.now().isoformat(timespec="seconds")}])
    pd.concat([labels, new], ignore_index=True).to_csv(path, index=False)


def show_results() -> None:
    """Only offered once every first-round episode is labelled, so results cannot bias labelling."""
    with st.container(border=True):
        st.markdown("**All episodes labelled.** The analysis compares your labels with the tool's flags.")
        if st.button("Show results", type="primary"):
            st.session_state.results_text = evaluate.analysis_text()
        if "results_text" not in st.session_state and evaluate.RESULTS.exists():
            st.session_state.results_text = evaluate.RESULTS.read_text(encoding="utf-8")
        if "results_text" in st.session_state:
            st.code(st.session_state.results_text, language=None)
            st.caption(f"Saved to evaluation/{evaluate.RESULTS.name}. Running it again overwrites that file "
                       "with up-to-date results.")


def labelling(mode: str) -> None:
    features = data.features()
    st.title("Blind labelling")
    sample_table = label_sample()
    if sample_table is None:
        st.error("No labelling sample yet. Run `python evaluation/evaluate.py sample` from the project folder.")
        return

    first_round = sample_table["episode"].tolist()
    first_labels = read_labels(LABELS)
    first_done = set(first_labels["episode"]) & set(first_round)
    relabel_ready = len(first_done) == len(first_round)
    rounds = ["First round"] + (["Re-label check"] if relabel_ready else [])
    chosen_round = st.radio("Round", rounds, horizontal=True,
                            help="The re-label check unlocks once every first-round episode is labelled.")
    if chosen_round == "First round":
        sample, path = first_round, LABELS
    else:
        sample, path = sample_table.loc[sample_table["relabel"], "episode"].tolist(), RELABELS
        last = pd.to_datetime(first_labels["labelled_at"]).max()
        hours = (datetime.now() - last).total_seconds() / 3600
        st.info(f"Re-label these {len(sample)} episodes without looking at your first answers. "
                f"Your last first-round label was {hours:.0f} hours ago"
                + (". Waiting at least a day makes the check more honest." if hours < 24 else "."))
    labels = read_labels(path)
    done = set(labels["episode"]) & set(sample)

    st.caption("Flags, scores and cut-through markers are hidden, so your judgement is independent of the rules. "
               "Label each episode: would a researcher want to investigate it?")
    with st.expander("Labelling guidelines", expanded=not len(first_labels)):
        st.markdown(LABEL_GUIDELINES)
    st.progress(len(done) / len(sample), text=f"{len(done)} of {len(sample)} labelled")
    if relabel_ready:
        show_results()

    remaining = [e for e in sample if e not in done]
    position = sample.index(remaining[0]) if remaining else 0
    episode = st.selectbox("Episode to label", sample, index=position,
                           format_func=lambda e: f"Episode {e}" + (" ✓ labelled" if e in done else ""),
                           key=f"label_episode_{chosen_round}", on_change=episode_changed)
    st.session_state.episode = episode
    row = features.loc[episode]
    cols = st.columns(3)
    cols[0].metric("Reward", f"{row.reward:.1f}")
    cols[1].metric("Cost", f"{row.cost:.0f}")
    cols[2].metric("Goals reached", int(row.goals))
    episode_view(episode, mode, show_flags=False)

    # Prefill only from this round's own file, so the re-label check never shows first-round answers.
    previous = labels[labels["episode"] == episode]
    with st.form(f"label_{chosen_round}_{episode}"):
        options = ["Yes", "No", "Unsure"]
        worth = st.radio("Worth investigating?", options, horizontal=True,
                         index=options.index(previous.iloc[0].worth_investigating) if len(previous) else None)
        kinds = st.multiselect("Why? (pick any that apply)", LABEL_KINDS,
                               default=[k for k in str(previous.iloc[0].kinds).split("; ") if k in LABEL_KINDS]
                               if len(previous) else [])
        note = st.text_area("What did you see?", value=str(previous.iloc[0].note) if len(previous) and
                            not pd.isna(previous.iloc[0].note) else "")
        if st.form_submit_button("Save label"):
            if worth is None:
                st.error("Choose Yes, No or Unsure first.")
            else:
                save_label(path, episode, worth, kinds, note)
                st.success(f"Saved episode {episode}.")
                st.rerun()


mode = theme_mode()
with st.sidebar:
    st.radio("Screen", ["Overview", "Inspector", "Labelling"], key="page")
    st.markdown("**Colours**: blue = hazards and safety, orange = goals and entanglement, "
                "aqua = vase and odd behaviour.")
    st.caption("Positions are rebuilt from the robot's own sensors (origin = episode start). "
               "The layout was not recorded, so there is no video replay.")

{"Overview": overview, "Inspector": inspector, "Labelling": labelling}[st.session_state.page](mode)
