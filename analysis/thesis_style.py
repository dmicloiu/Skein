#!/usr/bin/env python3
"""Shared plot style for the thesis figures.

One module every plot script CAN import so the figures read as one system. It
captures the *settled* house conventions extracted from the approved scripts
(plot_vllm_perf_study.py, plot_sem_filter_ab.py, plot_cross_system_frontier.py),
not a new invented look:

  * palette   -- blue = baseline / reference, green = behaviour under test.
                 Pure blue/green (plus neutral grays). NO red/orange.
  * rcParams  -- the two settled font scales (12 pt default, 10 pt dense).
  * labels    -- symbol-first axis labels, e.g. "K (distinct prefixes) [log]".
  * titles    -- HOUSE CONVENTION: descriptive per-panel titles are ALLOWED
                 (e.g. "Throughput", "Tail latency") to name what each panel
                 shows. NO argumentative suptitle -- the figure never states its
                 own conclusion, and run context (model / GPU / cap / batch /
                 scale) still belongs in the LaTeX caption. Neutral "(a)/(b)"
                 letter markers are OPTIONAL and OFF BY DEFAULT (panels are named
                 left/right in the caption); a configurable panel-marker policy
                 switches them on for figures with 3+ panels that need them.
  * sizing    -- a fit_to_text_width() helper (below) can normalise a figsize to
                 the text width so \\includegraphics[width=\\textwidth] does not
                 shrink it. It is DEFINED BUT CURRENTLY UNUSED: the router/async
                 figures are authored at their natural wide figsizes (the roomy
                 layout the author preferred) and accept the on-page shrink.

House vocabulary also collected here as named constants/helpers so it stays
consistent across figures: neutral grays for cross-figure reference lines and
gray dashed prediction bands, a white-bbox knockout for text over busy artwork,
and shaded-span alphas.

The rcParams presets reproduce the approved scripts' current look exactly, so a
later migration of those scripts to this module is behaviour-preserving.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette -- blue = baseline / reference, green = behaviour under test.
# When a figure needs a second series in a family, use the *_ALT shade.
# NEVER red/orange in the comparison figures.
# ---------------------------------------------------------------------------
BLUE = "blue"          # primary baseline / reference (matplotlib pure blue)
GREEN = "green"        # primary behaviour under test (matplotlib green)
BLUE_ALT = "#08519c"   # second blue-family series (dark blue)
GREEN_ALT = "#009E73"  # second green-family series (Okabe-Ito bluish green)

# Okabe-Ito colour-blind-safe anchors used by the cross-system frontier figure
# (flock = bluish green, LOTUS = blue). Exposed so that approved script can
# migrate without its two blue/green series changing hue.
OKABE_BLUE = "#0072B2"
OKABE_GREEN = "#009E73"  # == GREEN_ALT

# Sole documented exception to "pure blue/green": the cross-system frontier is a
# THREE-system comparison and needs a third distinguisher for Palimpzest. Do NOT
# use this in new comparison figures.
CROSS_SYSTEM_THIRD = "#E69F00"  # Okabe-Ito orange, cross_system_frontier only

# Neutral grays -- reference lines, prediction bands, callout text/edges.
GRAY = "#7f7f7f"       # cross-figure reference lines, secondary annotations
GRAY_DARK = "0.35"     # ceiling lines, muted callouts
GRAY_DARKER = "0.2"    # strong neutral emphasis (linear-ideal line, callout text)

# Named series roles (semantic aliases, so scripts read as intent not hue).
COLOR_BASELINE = BLUE
COLOR_UNDER_TEST = GREEN
COLOR_BASELINE_ALT = BLUE_ALT
COLOR_UNDER_TEST_ALT = GREEN_ALT

# Shaded-span alphas (house-standard translucency for axhspan/axvspan bands).
SPAN_ALPHA = 0.06
BAND_ALPHA = 0.18      # per-series min/max fill_between band

# Gray dashed prediction-band line style (e.g. a min(morsels, threads) law).
PREDICTION_LINE = dict(color=GRAY, ls="--", lw=2.6, alpha=0.55)


# ---------------------------------------------------------------------------
# rcParams -- the two settled font scales.
# ---------------------------------------------------------------------------
def house_rcparams(base_font: int = 12) -> dict:
    """Return the settled house rcParams for one of the two house font scales.

    base_font=12 -> the router / cross-system / sem-filter scale (default).
    base_font=10 -> the denser vllm-perf-study / asyncllmclient scale.

    Both scales are reproduced exactly (behaviour-preserving for the approved
    scripts). Any other value scales the 12 pt preset proportionally.
    """
    common = {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "legend.loc": "best",
    }
    if base_font == 10:
        # Exact reproduction of plot_vllm_perf_study.py / asyncllmclient_vs_vllm.py.
        common.update({
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        })
        return common
    # 12 pt house scale (router / cross-system / sem-filter). hatch.linewidth is
    # harmless where there are no hatches, so it lives in the default preset.
    title = 13 if base_font == 12 else base_font + 1
    tick = 11 if base_font == 12 else base_font - 1
    legend = 11 if base_font == 12 else base_font - 1
    common.update({
        "font.size": base_font,
        "axes.titlesize": title,
        "axes.labelsize": base_font,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "hatch.linewidth": 1.1,
    })
    return common


def setup_style(base_font: int = 12) -> None:
    """Apply the house rcParams (see house_rcparams). Call once before plotting."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(house_rcparams(base_font))


# ---------------------------------------------------------------------------
# Figure sizing -- OPTIONAL text-width normaliser (currently unused).
# ---------------------------------------------------------------------------
# The thesis text block is ~453 pt ~= 6.3 in wide. A figure authored WIDER than
# this is shrunk to the text block by \includegraphics[width=\textwidth], and
# that shrink divides every point-sized label with it: a 12 pt axis label on a
# 13 in figure lands at ~6 pt on the page. Authoring at the text width instead
# would avoid the shrink. This helper is a single module-level lever for that,
# but is NOT currently applied: the router/async figures are authored at their
# natural wide figsizes (the roomy, non-colliding layout) and accept the on-page
# shrink. Kept for reference / future use.
TEXT_WIDTH_IN = 6.3


def fit_to_text_width(figsize):
    """Scale a natural ``(w, h)`` figsize to TEXT_WIDTH_IN, preserving aspect.

    The saved figure is then exactly the text width, so
    ``\\includegraphics[width=\\textwidth]`` neither shrinks nor enlarges it and
    the house font sizes render at their true point size on the page. The on-page
    figure dimensions are unchanged from before (width is forced to \\textwidth
    either way) -- only the labels render bigger."""
    w, h = figsize
    s = TEXT_WIDTH_IN / w
    return (w * s, h * s)


# ---------------------------------------------------------------------------
# Axis labels -- symbol-first.
# ---------------------------------------------------------------------------
def axis_label(symbol: str, detail: str | None = None, *, log: bool = False) -> str:
    """House axis label: symbol first, optional parenthetical gloss/unit, then
    an optional " [log]" scale tag.

        axis_label("K", "distinct recurring prefixes", log=True)
            -> "K (distinct recurring prefixes) [log]"
        axis_label("throughput", "rows/s")   -> "throughput (rows/s)"
        axis_label("R", "rows per prompt", log=True) -> "R (rows per prompt) [log]"
    """
    out = symbol
    if detail:
        out += f" ({detail})"
    if log:
        out += " [log]"
    return out


# ---------------------------------------------------------------------------
# Titles -- house convention: descriptive per-panel titles ALLOWED; NO
# argumentative suptitle; "(a)/(b)" letter markers OPTIONAL and OFF BY DEFAULT.
# ---------------------------------------------------------------------------
# Scripts name each panel with ax.set_title("Throughput") -- what the panel
# shows, never its conclusion (that stays in the caption). The "(a)/(b)" letter
# markers below are a SEPARATE, optional overlay that composes with those
# descriptive titles; it is off by default:
# "none"    -> no in-figure "(a)/(b)" letter marker (default; panels are named
#              in the caption as left/right).
# "letters" -> neutral "(a)", "(b)", ... markers, for figures with 3+ panels
#              that need in-figure identification.
# Flip this one module-level value for a thesis-wide switch; scripts should call
# set_panel_marker() once per panel so the switch touches nothing else.
PANEL_MARKER_POLICY = "none"


def set_panel_marker(ax, index: int, policy: str | None = None) -> None:
    """Apply the house panel-marker policy to one axes.

    index is 0-based (0 -> 'a'). Under the default "none" policy this is a no-op,
    so a figure shows only its data + descriptive per-panel title + symbol-first
    axis labels + legend. Setting PANEL_MARKER_POLICY (or passing
    policy="letters") switches every panel to a neutral "(a)/(b)" left-aligned
    marker in one place; the letter is PREPENDED to any descriptive title already
    set (e.g. "(a) Throughput"), so the two conventions compose without clobber.
    """
    pol = policy or PANEL_MARKER_POLICY
    if pol == "none":
        return
    if pol == "letters":
        letter = f"({chr(ord('a') + index)})"
        existing = ax.get_title()
        ax.set_title(f"{letter} {existing}" if existing else letter)
        return
    raise ValueError(f"unknown panel-marker policy: {pol!r}")


def no_suptitle(fig) -> None:
    """Enforce the no-suptitle house policy: blank any figure suptitle that slipped
    in. Descriptive text and run context belong in the LaTeX caption."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")


# ---------------------------------------------------------------------------
# House vocabulary helpers.
# ---------------------------------------------------------------------------
def white_bbox(pad: float = 0.8) -> dict:
    """White knockout box for text placed over busy artwork (bars, curves)."""
    return dict(facecolor="white", edgecolor="none", pad=pad)


def reference_line(ax, *, x=None, y=None, color: str = GRAY, ls: str = ":",
                   lw: float = 1.0, **kw):
    """A neutral cross-figure reference line (vertical if x given, else horizontal)."""
    if x is not None:
        return ax.axvline(x, color=color, ls=ls, lw=lw, **kw)
    return ax.axhline(y, color=color, ls=ls, lw=lw, **kw)
