"""Plotly figures for the app.

Colour follows the entity everywhere, so a reader learns it once:
hazards and safety flags are slot 1 (blue), goals and entanglement flags
slot 2 (orange), the vase and odd-behaviour flags slot 3 (aqua). These three
pass the colour-blindness checks as a set, in light and dark mode.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTES = {
    "light": {"hazard": "#2a78d6", "goal": "#eb6834", "vase": "#1baf7a", "hazard_light": "#86b6ef",
              "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "none": "#c3c2b7", "surface": "#fcfcfb"},
    "dark": {"hazard": "#3987e5", "goal": "#d95926", "vase": "#199e70", "hazard_light": "#1c5cab",
             "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "none": "#52514e", "surface": "#1a1a19"},
}
CATEGORY_ENTITY = {"safety": "hazard", "entanglement": "goal", "odd_behaviour": "vase"}
CATEGORY_LABEL = {"safety": "Safety", "entanglement": "Reward–cost entanglement",
                  "odd_behaviour": "Odd behaviour", "none": "No flag"}
# Trace order of the overview scatter. The app maps a clicked dot back to its
# episode by trace number and position, so the two must use the same order.
SCATTER_ORDER = ["none", "safety", "entanglement", "odd_behaviour"]
HAZARD_RADIUS = 0.2
GOAL_RADIUS = 0.3
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def _layout(fig: go.Figure, c: dict, height: int, **extra) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=64, r=16, t=48, b=56), font=dict(family=FONT, color=c["ink2"], size=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=c["surface"],
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=c["ink2"])),
                      hoverlabel=dict(font=dict(family=FONT)), **extra)
    fig.update_xaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, tickfont=dict(color=c["muted"]))
    fig.update_yaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, tickfont=dict(color=c["muted"]))
    return fig


def overview_scatter(features: pd.DataFrame, primary_category: pd.Series, mode: str) -> go.Figure:
    """Reward vs cost for every episode. Colour = first flag category; size = unusualness."""
    c = PALETTES[mode]
    fig = go.Figure()
    for category in SCATTER_ORDER:
        rows = features[primary_category == category]
        colour = c["none"] if category == "none" else c[CATEGORY_ENTITY[category]]
        fig.add_trace(go.Scatter(
            x=rows["cost"], y=rows["reward"], mode="markers", name=CATEGORY_LABEL[category],
            customdata=np.column_stack((rows.index, rows["unusualness_rank"] * 100)),
            marker=dict(color=colour, size=6 + 10 * rows["unusualness_rank"] ** 4,
                        opacity=0.45 if category == "none" else 0.9,
                        line=dict(width=1, color=c["surface"])),
            hovertemplate="Episode %{customdata[0]}<br>Reward %{y:.1f} · Cost %{x:.0f}"
                          "<br>More unusual than %{customdata[1]:.0f}% of episodes<extra></extra>"))
    fig.update_xaxes(title="Total cost (steps inside a hazard)")
    fig.update_yaxes(title="Total reward")
    return _layout(fig, c, 460)


def episode_map(steps: pd.DataFrame, world: pd.DataFrame, cut_throughs: pd.DataFrame, step: int,
                window: tuple[int, int] | None, mode: str, show_markers: bool = True) -> go.Figure:
    """Rebuilt path and map. Origin is where the episode starts."""
    c = PALETTES[mode]
    fig = go.Figure()
    for _, h in world[world["object"] == "hazard"].iterrows():
        fig.add_shape(type="circle", x0=h.x - HAZARD_RADIUS, x1=h.x + HAZARD_RADIUS, y0=h.y - HAZARD_RADIUS,
                      y1=h.y + HAZARD_RADIUS, line=dict(color=c["hazard"], width=1.5), fillcolor=c["hazard"],
                      opacity=0.25, layer="below")
    hazards = world[world["object"] == "hazard"]
    fig.add_trace(go.Scatter(x=hazards["x"], y=hazards["y"], mode="markers", name="Hazard",
                             marker=dict(color=c["hazard"], size=10, opacity=0.35, symbol="circle"),
                             hovertemplate="Hazard (radius 0.2)<extra></extra>"))

    shown = steps if window is None else steps.iloc[window[0]:window[1] + 1]
    fig.add_trace(go.Scatter(x=steps["x"], y=steps["y"], mode="lines", name="Path (whole episode)",
                             line=dict(color=c["axis"], width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=shown["x"], y=shown["y"], mode="markers",
                             name="Path, by time" if window is None else "Path, selected window",
                             marker=dict(color=shown.index, colorscale=[[0, c["axis"]], [1, c["ink"]]], size=3),
                             customdata=shown.index, hovertemplate="Step %{customdata}<extra></extra>"))
    costly = shown[shown["cost"] > 0]
    fig.add_trace(go.Scatter(x=costly["x"], y=costly["y"], mode="markers", name="Inside a hazard (cost)",
                             marker=dict(color=c["hazard"], size=6, line=dict(width=1, color=c["surface"])),
                             customdata=costly.index, hovertemplate="Step %{customdata}: inside a hazard<extra></extra>"))

    goals = world[world["object"] == "goal"]
    fig.add_trace(go.Scatter(x=goals["x"], y=goals["y"], mode="markers+text", name="Goal (numbered)",
                             text=(goals["goal_number"] + 1).astype(int).astype(str), textposition="middle center",
                             textfont=dict(color=c["surface"], size=9),
                             marker=dict(color=c["goal"], size=16, symbol="circle"),
                             customdata=goals["reached_step"].map(lambda v: "not reached" if pd.isna(v) else f"step {int(v)}"),
                             hovertemplate="Goal %{text}<br>Reached: %{customdata}<extra></extra>"))

    vase = shown.dropna(subset=["vase_x"])
    if len(vase):
        fig.add_trace(go.Scatter(x=vase["vase_x"], y=vase["vase_y"], mode="markers", name="Vase position",
                                 marker=dict(color=c["vase"], size=4, symbol="square"),
                                 customdata=vase.index, hovertemplate="Vase at step %{customdata}<extra></extra>"))

    if show_markers and len(cut_throughs):
        cuts = cut_throughs
        mid = ((cuts["start_step"] + cuts["stop_step"]) // 2).clip(upper=len(steps) - 1)
        fig.add_trace(go.Scatter(x=steps["x"].to_numpy()[mid], y=steps["y"].to_numpy()[mid], mode="markers",
                                 name="Cut-through", marker=dict(symbol="x", size=11, color=c["ink"]),
                                 customdata=np.column_stack((cuts["start_step"], cuts["stop_step"])),
                                 hovertemplate="Cut-through, steps %{customdata[0]}–%{customdata[1]}<extra></extra>"))

    here = steps.iloc[step]
    fig.add_trace(go.Scatter(x=[here.x], y=[here.y], mode="markers", name=f"Robot at step {step}",
                             marker=dict(symbol="triangle-up", angle=90 - np.degrees(here.heading), size=16,
                                         color=c["ink"], line=dict(width=2, color=c["surface"])),
                             hovertemplate=f"Robot at step {step}<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (units)")
    fig.update_xaxes(title="x (units)")
    return _layout(fig, c, 520)


def timeline(steps: pd.DataFrame, events: pd.DataFrame, step: int, window: tuple[int, int] | None,
             mode: str) -> go.Figure:
    """Three stacked panels sharing the step axis; each panel has its own single y-scale."""
    c = PALETTES[mode]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.45, 0.2, 0.35],
                        subplot_titles=("Closeness (1 = touching)", "Cost", "Cumulative reward"))
    x = steps.index
    fig.add_trace(go.Scatter(x=x, y=steps["hazard_closeness"], name="Nearest hazard",
                             line=dict(color=c["hazard"], width=2),
                             hovertemplate="Step %{x}: hazard closeness %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=steps["goal_closeness"], name="Goal",
                             line=dict(color=c["goal"], width=2),
                             hovertemplate="Step %{x}: goal closeness %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0.933, line=dict(color=c["muted"], width=1), row=1, col=1,
                  annotation_text="inside hazard", annotation_font_color=c["muted"], annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=x, y=steps["cost"], name="Cost", line=dict(color=c["hazard"], width=2, shape="hv"),
                             fill="tozeroy", showlegend=False,
                             hovertemplate="Step %{x}: cost %{y}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=steps["reward"].cumsum(), name="Cumulative reward",
                             line=dict(color=c["ink2"], width=2), showlegend=False,
                             hovertemplate="Step %{x}: reward so far %{y:.2f}<extra></extra>"), row=3, col=1)

    bands = {"hazard_stay": c["hazard"], "vase_pushed": c["vase"]}
    for _, e in events[events["event_type"].isin(bands)].iterrows():
        fig.add_vrect(x0=e.start_step, x1=e.stop_step, fillcolor=bands[e.event_type], opacity=0.18,
                      line_width=0, row=1, col=1)
    for _, e in events[events["event_type"] == "near_miss"].iterrows():
        fig.add_vrect(x0=e.start_step, x1=e.stop_step, fillcolor="rgba(0,0,0,0)",
                      line=dict(color=c["hazard"], width=1), row=1, col=1)
    goals = events[events["event_type"] == "goal_reached"]
    fig.add_trace(go.Scatter(x=goals["start_step"], y=np.ones(len(goals)), mode="markers", name="Goal reached",
                             marker=dict(symbol="star", size=11, color=c["goal"], line=dict(width=1, color=c["surface"])),
                             hovertemplate="Goal reached at step %{x}<extra></extra>"), row=1, col=1)
    # Legend entries for the shaded bands.
    for label, colour, symbol in [("Hazard stay", c["hazard"], "square"), ("Near miss (outlined)", c["hazard"], "square-open"),
                                  ("Vase pushed", c["vase"], "square")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(symbol=symbol, size=10, color=colour, opacity=0.5 if symbol == "square" else 1)),
                      row=1, col=1)

    fig.add_vline(x=step, line=dict(color=c["ink"], width=1))
    if window is not None:
        fig.update_xaxes(range=[window[0] - 0.5, window[1] + 0.5])
    fig.update_xaxes(title="Step", row=3, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.1], tickvals=[0, 1], row=2, col=1)
    fig.update_annotations(font=dict(color=c["ink2"], size=12))
    return _layout(fig, c, 520, hovermode="x unified")


def radar(lidar: dict[str, np.ndarray], step: int, mode: str) -> go.Figure:
    """What the robot sensed at one step: 16 directions per object type. Forward is up."""
    c = PALETTES[mode]
    angles = (np.arange(16) + 0.5) * 22.5
    fig = go.Figure()
    for name, entity in [("hazard", "hazard"), ("goal", "goal"), ("vase", "vase")]:
        fig.add_trace(go.Barpolar(r=lidar[name][step], theta=angles, width=[20] * 16, name=name.capitalize(),
                                  marker=dict(color=c[entity], line=dict(color=c["surface"], width=2)), opacity=0.8,
                                  hovertemplate=f"{name.capitalize()}: closeness %{{r:.2f}} at %{{theta:.0f}}°<extra></extra>"))
    # Polar bars stack by default, which would add hazard and goal readings together.
    fig.update_layout(polar=dict(barmode="overlay", bgcolor=c["surface"],
                                 angularaxis=dict(rotation=90, direction="counterclockwise", gridcolor=c["grid"],
                                                  tickvals=[0, 90, 180, 270], ticktext=["front", "left", "back", "right"],
                                                  tickfont=dict(color=c["ink2"])),
                                 radialaxis=dict(range=[0, 1], angle=45, tickangle=45, tickvals=[0.5, 1], showline=False,
                                                 gridcolor=c["grid"], tickfont=dict(color=c["muted"], size=10))))
    fig = _layout(fig, c, 420)
    return fig.update_layout(legend=dict(y=-0.08, yanchor="top"))
