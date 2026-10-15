"""One visual system for every figure (PNG for slides + interactive HTML to explore).

Palette: the validated reference palette from the dataviz method (categorical hues assigned in fixed
order, one-hue sequential ramp, blue↔red diverging with a neutral midpoint). Every figure is saved by
`save_fig(fig, name, caption)` which writes reports/figures/<name>.png, <name>.html and records the
caption in reports/figures/captions.json → CAPTIONS.md so the report and slides can reuse them."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import ROOT

FIG_DIR = ROOT / "reports" / "figures"

# --- palette (light surface) ---------------------------------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", "#f0efec", "#f2a09f", "#e34948", "#a32626", "#6b1515"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb", "page": "#f9f9f7"}
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

_registered = False


def setup() -> None:
    """Register the plotly template + matplotlib rcParams (idempotent)."""
    global _registered
    if _registered:
        return
    import matplotlib as mpl
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["smc"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, size=13, color=INK["primary"]),
            title=dict(font=dict(size=16, color=INK["primary"]), x=0.01, xanchor="left", y=0.97, yanchor="top"),
            paper_bgcolor=INK["surface"],
            plot_bgcolor=INK["surface"],
            colorway=PALETTE,
            margin=dict(l=70, r=28, t=96, b=60),
            hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, size=12, color=INK["primary"]), bordercolor=INK["axis"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=12, color=INK["secondary"]), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=INK["grid"], gridwidth=1, zeroline=False, linecolor=INK["axis"], tickfont=dict(color=INK["muted"], size=11), title=dict(font=dict(color=INK["secondary"], size=12)), showline=True, automargin=True),
            yaxis=dict(gridcolor=INK["grid"], gridwidth=1, zeroline=False, linecolor=INK["axis"], tickfont=dict(color=INK["muted"], size=11), title=dict(font=dict(color=INK["secondary"], size=12)), showline=False, automargin=True),
            bargap=0.28,
            bargroupgap=0.08,
            coloraxis=dict(colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)], colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=INK["muted"], size=11))),
        ),
        data=dict(
            bar=[go.Bar(marker=dict(line=dict(width=0)), textfont=dict(color=INK["secondary"]))],
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=1.5, color=INK["surface"])))],
            histogram=[go.Histogram(marker=dict(line=dict(width=0)))],
        ),
    )
    pio.templates.default = "smc"

    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"], "font.size": 11,
        "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"], "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": INK["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10, "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2, "patch.linewidth": 0, "figure.dpi": 110,
    })
    _registered = True


def _captions_file() -> Path:
    return FIG_DIR / "captions.json"


def _write_captions_md(caps: dict[str, Any]) -> None:
    lines = ["# Figure captions", "", "Every figure exists as `<name>.png` (for slides/report) and `<name>.html` (interactive).", "",
             "| # | File | Caption | Source |", "|--:|------|---------|--------|"]
    for i, (name, c) in enumerate(sorted(caps.items(), key=lambda kv: kv[1].get("order", 0)), start=1):
        lines.append(f"| {i} | `{name}.png` | {c['caption']} | {c.get('source','')} |")
    (FIG_DIR / "CAPTIONS.md").write_text("\n".join(lines) + "\n")


def record_caption(name: str, caption: str, source: str = "") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f = _captions_file()
    caps = json.loads(f.read_text()) if f.exists() else {}
    order = caps.get(name, {}).get("order") or (max([c.get("order", 0) for c in caps.values()] + [0]) + 1)
    caps[name] = {"caption": caption, "source": source or os.environ.get("SMC_SOURCE", ""), "order": order, "updated": time.strftime("%Y-%m-%d")}
    f.write_text(json.dumps(caps, indent=2, ensure_ascii=False))
    _write_captions_md(caps)


def _png_via_kaleido(fig, out: Path, width: int, height: int, scale: float) -> bool:
    try:
        fig.write_image(str(out), width=width, height=height, scale=scale)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[viz] kaleido PNG export failed for {out.name}: {type(e).__name__}: {str(e)[:160]}")
        return False


def _png_via_playwright(html_path: Path, out: Path, width: int, height: int, scale: float) -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
            pg.goto(html_path.resolve().as_uri())
            pg.wait_for_timeout(600)
            pg.screenshot(path=str(out), full_page=False)
            b.close()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[viz] playwright PNG export failed for {out.name}: {type(e).__name__}: {str(e)[:160]}")
        return False


def save_fig(fig, name: str, caption: str, width: int = 960, height: int = 540, scale: float = 2.0, source: str = "") -> dict[str, str]:
    """Save a plotly or matplotlib figure as PNG + HTML and record its caption. Returns the paths."""
    setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png, html = FIG_DIR / f"{name}.png", FIG_DIR / f"{name}.html"
    is_plotly = hasattr(fig, "write_html")
    if is_plotly:
        fig.update_layout(width=None, height=None, autosize=True)
        fig.write_html(str(html), include_plotlyjs="directory", full_html=True, config={"displaylogo": False, "responsive": True})
        fig.update_layout(width=width, height=height)
        ok = _png_via_kaleido(fig, png, width, height, scale) or _png_via_playwright(html, png, width, height, scale)
        fig.update_layout(width=None, height=None)
        if not ok:
            print(f"[viz] no PNG written for {name}")
    else:
        fig.savefig(str(png), dpi=int(110 * scale), bbox_inches="tight")
        html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{name}</title></head><body style='margin:0;background:{INK['surface']}'><img src='{name}.png' style='max-width:100%'></body></html>")
    record_caption(name, caption, source)
    return {"png": str(png), "html": str(html)}


def hover_fmt(cols: list[str]) -> str:
    """Plotly hovertemplate listing `cols` from customdata."""
    return "<br>".join(f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(cols)) + "<extra></extra>"
