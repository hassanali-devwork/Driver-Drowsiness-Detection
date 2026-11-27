"""Shared design system, dual-theme styling, and inference helpers for the Eye-State Streamlit app."""
import io
import json
import wave
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR, RESULTS_DIR, IMG_DIR = ROOT / "models", ROOT / "results", ROOT / "images"
IMG_SIZE = 64  # must match Phase 1 / Phase 2 preprocessing
GREEN, RED = (52, 168, 112), (214, 72, 72)

# ---------------------------------------------------------------------------
# Dual-Theme Palette (Dark = Midnight Horizon Blue, Light = Warm Skin / Sand)
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "name": "Midnight Blue",
        "switch_label": "Light Skin",
        "bg": "#070b14",
        "bg_elevated": "#0b1120",
        "surface": "#0e1628",
        "surface_soft": "rgba(18, 29, 52, 0.72)",
        "surface_hover": "#142038",
        "nav_bg": "rgba(11, 17, 32, 0.78)",
        "border": "rgba(148, 179, 232, 0.13)",
        "border_strong": "rgba(148, 179, 232, 0.26)",
        "text": "#eef3fb",
        "text_secondary": "#a4b5d0",
        "text_muted": "#6e82a3",
        "accent": "#6b9cff",
        "accent_deep": "#284b8a",
        "accent_soft": "rgba(107, 156, 255, 0.14)",
        "cta_bg": "#eef3fb",
        "cta_text": "#081020",
        "good": "#34d399",
        "good_soft": "rgba(52, 211, 153, 0.13)",
        "warning": "#eab308",
        "warning_soft": "rgba(234, 179, 8, 0.13)",
        "danger": "#f87171",
        "danger_soft": "rgba(248, 113, 113, 0.14)",
        "neutral": "#94a3b8",
        "hero_bg": (
            "radial-gradient(ellipse 85% 70% at 50% 106%, rgba(196, 134, 102, 0.22) 0%, "
            "rgba(49, 98, 180, 0.44) 36%, rgba(15, 34, 72, 0.88) 70%, #091020 100%), "
            "linear-gradient(180deg, #0b162e 0%, #122852 52%, #091020 100%)"
        ),
        "hero_text": "#f4f7fd",
        "hero_sub": "rgba(218, 229, 248, 0.82)",
        "hero_chip_bg": "rgba(255, 255, 255, 0.08)",
        "hero_chip_border": "rgba(195, 216, 252, 0.20)",
        "horizon_ridge": "#070b14",
        "horizon_ridge_mid": "rgba(16, 33, 66, 0.72)",
        "ambient_glow": (
            "radial-gradient(ellipse 80% 40% at 50% -10%, rgba(44, 88, 166, 0.24), transparent 70%)"
        ),
        "shadow": "0 20px 52px rgba(2, 6, 18, 0.55)",
        "shadow_soft": "0 8px 24px rgba(2, 6, 18, 0.34)",
        "chart_grid": "rgba(148, 179, 232, 0.12)",
        "chart_palette": ["#6b9cff", "#34d399", "#93c5fd", "#c4b5fd"],
        "table_cmap": "PuBu",
    },
    "light": {
        "name": "Warm Skin",
        "switch_label": "Midnight Blue",
        "bg": "#faf3ec",
        "bg_elevated": "#f4e9de",
        "surface": "#fffaf5",
        "surface_soft": "rgba(244, 231, 218, 0.72)",
        "surface_hover": "#f7ede2",
        "nav_bg": "rgba(255, 250, 245, 0.84)",
        "border": "rgba(148, 108, 84, 0.17)",
        "border_strong": "rgba(148, 108, 84, 0.31)",
        "text": "#241914",
        "text_secondary": "#5d483d",
        "text_muted": "#8c7265",
        "accent": "#9b5d44",
        "accent_deep": "#78442f",
        "accent_soft": "rgba(155, 93, 68, 0.11)",
        "cta_bg": "#2b1e18",
        "cta_text": "#faf3ec",
        "good": "#2b7a58",
        "good_soft": "rgba(43, 122, 88, 0.11)",
        "warning": "#9f6620",
        "warning_soft": "rgba(159, 102, 32, 0.12)",
        "danger": "#b8443a",
        "danger_soft": "rgba(184, 68, 58, 0.11)",
        "neutral": "#7c685e",
        "hero_bg": (
            "radial-gradient(ellipse 85% 70% at 50% 106%, rgba(191, 131, 98, 0.34) 0%, "
            "rgba(222, 184, 158, 0.56) 38%, rgba(242, 224, 208, 0.90) 74%, #f4e6d8 100%), "
            "linear-gradient(180deg, #edd8c5 0%, #f5e9dc 55%, #f2e2d2 100%)"
        ),
        "hero_text": "#241914",
        "hero_sub": "rgba(66, 46, 36, 0.80)",
        "hero_chip_bg": "rgba(255, 250, 245, 0.62)",
        "hero_chip_border": "rgba(148, 108, 84, 0.24)",
        "horizon_ridge": "#faf3ec",
        "horizon_ridge_mid": "rgba(214, 178, 152, 0.55)",
        "ambient_glow": (
            "radial-gradient(ellipse 80% 40% at 50% -10%, rgba(212, 166, 134, 0.25), transparent 70%)"
        ),
        "shadow": "0 18px 44px rgba(84, 54, 38, 0.08)",
        "shadow_soft": "0 6px 20px rgba(84, 54, 38, 0.05)",
        "chart_grid": "rgba(148, 108, 84, 0.14)",
        "chart_palette": ["#9b5d44", "#2b7a58", "#c2876b", "#6e5144"],
        "table_cmap": "YlOrBr",
    },
}


def init_theme() -> str:
    """Initialize and return the active theme ('dark' or 'light')."""
    if "theme" not in st.session_state:
        qp = st.query_params.get("theme", "dark")
        st.session_state["theme"] = qp if qp in THEMES else "dark"
    return st.session_state["theme"]


def get_theme() -> dict:
    """Return the token dictionary for the active theme."""
    return THEMES[init_theme()]


def toggle_theme() -> None:
    """Switch between dark midnight-blue and light warm-skin themes."""
    nxt = "light" if init_theme() == "dark" else "dark"
    st.session_state["theme"] = nxt
    st.query_params["theme"] = nxt


def svg_icon(name: str, size: int = 18, stroke: float = 1.65) -> str:
    """Return a clean minimal inline SVG icon (no emojis)."""
    paths = {
        "eye": (
            '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/>'
            '<circle cx="12" cy="12" r="3"/>'
        ),
        "eye-off": (
            '<path d="M17.94 17.94A10.9 10.9 0 0 1 12 19c-6.4 0-10-7-10-7a18.5 18.5 0 0 1 5.06-5.94"/>'
            '<path d="M9.9 5.24A10.5 10.5 0 0 1 12 5c6.4 0 10 7 10 7a18.5 18.5 0 0 1-2.16 3.19"/>'
            '<line x1="2" y1="2" x2="22" y2="22"/>'
        ),
        "camera": (
            '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>'
            '<circle cx="12" cy="13" r="4"/>'
        ),
        "upload": (
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            '<polyline points="17 8 12 3 7 8"/>'
            '<line x1="12" y1="3" x2="12" y2="15"/>'
        ),
        "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "cpu": (
            '<rect x="4" y="4" width="16" height="16" rx="2"/>'
            '<rect x="9" y="9" width="6" height="6"/>'
            '<line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>'
            '<line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>'
            '<line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>'
            '<line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
        ),
        "sliders": (
            '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
            '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
            '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
            '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
            '<line x1="17" y1="16" x2="23" y2="16"/>'
        ),
        "alert": (
            '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
        ),
        "check": (
            '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
            '<polyline points="22 4 12 14.01 9 11.01"/>'
        ),
        "layers": (
            '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
            '<polyline points="2 17 12 22 22 17"/>'
            '<polyline points="2 12 12 17 22 12"/>'
        ),
        "sun": (
            '<circle cx="12" cy="12" r="5"/>'
            '<line x1="12" y1="1" x2="12" y2="3"/>'
            '<line x1="12" y1="21" x2="12" y2="23"/>'
            '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>'
            '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>'
            '<line x1="1" y1="12" x2="3" y2="12"/>'
            '<line x1="21" y1="12" x2="23" y2="12"/>'
            '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>'
            '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
        ),
        "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    }
    body = paths.get(name, paths["eye"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


def inject_css() -> None:
    """Inject calm dual-theme CSS tokens, typography, and component styles across all pages."""
    t = get_theme()
    st.markdown(
        f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

      @property --p {{ syntax: '<number>';  inherits: true; initial-value: 0; }}
      @property --n {{ syntax: '<integer>'; inherits: true; initial-value: 0; }}

      :root {{
        --bg: {t['bg']};
        --bg-elevated: {t['bg_elevated']};
        --surface: {t['surface']};
        --surface-soft: {t['surface_soft']};
        --surface-hover: {t['surface_hover']};
        --nav-bg: {t['nav_bg']};
        --border: {t['border']};
        --border-strong: {t['border_strong']};
        --text: {t['text']};
        --text-secondary: {t['text_secondary']};
        --text-muted: {t['text_muted']};
        --accent: {t['accent']};
        --accent-deep: {t['accent_deep']};
        --accent-soft: {t['accent_soft']};
        --cta-bg: {t['cta_bg']};
        --cta-text: {t['cta_text']};
        --good: {t['good']};
        --good-soft: {t['good_soft']};
        --warning: {t['warning']};
        --warning-soft: {t['warning_soft']};
        --danger: {t['danger']};
        --danger-soft: {t['danger_soft']};
        --neutral: {t['neutral']};
        --hero-bg: {t['hero_bg']};
        --hero-text: {t['hero_text']};
        --hero-sub: {t['hero_sub']};
        --hero-chip-bg: {t['hero_chip_bg']};
        --hero-chip-border: {t['hero_chip_border']};
        --horizon-ridge: {t['horizon_ridge']};
        --horizon-ridge-mid: {t['horizon_ridge_mid']};
        --ambient-glow: {t['ambient_glow']};
        --shadow: {t['shadow']};
        --shadow-soft: {t['shadow_soft']};
        --font-sans: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
      }}

      html, body, .stApp {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
        -webkit-font-smoothing: antialiased;
      }}

      .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background: var(--ambient-glow);
      }}

      [data-testid="stHeader"] {{
        background: transparent !important;
      }}

      [data-testid="stSidebar"] {{
        background: var(--bg-elevated) !important;
        border-right: 1px solid var(--border) !important;
      }}
      [data-testid="stSidebar"] * {{
        color: var(--text) !important;
      }}

      .main .block-container,
      [data-testid="stMainBlockContainer"] {{
        max-width: 1140px !important;
        padding-top: 3.1rem !important;
        padding-bottom: 3.5rem !important;
        position: relative;
        z-index: 1;
      }}

      h1, h2, h3, h4, .hero h1, .sec-head h2 {{
        font-family: var(--font-sans) !important;
        color: var(--text);
        letter-spacing: -0.025em;
      }}

      p, li, span, label, .stMarkdown, .stMarkdown p {{
        font-family: var(--font-body);
        color: var(--text);
      }}

      [data-testid="stCaptionContainer"] p,
      .stCaption {{
        color: var(--text-muted) !important;
        font-size: 0.86rem !important;
      }}

      /* ---------- Hide sidebar & sidebar toggle entirely ---------- */
      [data-testid="stSidebar"],
      [data-testid="stSidebarNav"],
      [data-testid="collapsedControl"],
      button[kind="header"][aria-label="Open sidebar"],
      section[data-testid="stSidebar"] {{
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
      }}


      /* ---------- Top Navigation Bar ---------- */
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) {{
        background: var(--nav-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 6px 14px;
        margin-bottom: 22px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: var(--shadow-soft);
        align-items: center !important;
        min-height: 52px;
      }}
      /* Strip Streamlit's default wrapper margins inside navbar so heights match exactly */
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) > div[data-testid="column"] {{
        display: flex !important;
        align-items: center !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) .stButton {{
        width: 100%;
        margin: 0 !important;
        padding: 0 !important;
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) [data-testid="stPageLink"] {{
        width: 100%;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .nav-brand {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: 0.96rem;
        letter-spacing: -0.02em;
        color: var(--text);
        padding: 2px 4px;
        white-space: nowrap;
        height: 36px;
      }}
      .nav-brand-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--border-strong);
        flex-shrink: 0;
      }}
      .nav-tag {{
        font-family: var(--font-mono);
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 999px;
        background: var(--surface-soft);
        color: var(--text-muted);
        border: 1px solid var(--border);
        white-space: nowrap;
      }}
      /* Page links: exact 36px height, vertically centered */
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) [data-testid="stPageLink-NavLink"] {{
        border-radius: 10px;
        padding: 0 12px;
        justify-content: center;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        transition: background 0.18s ease;
        width: 100%;
        box-sizing: border-box;
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) [data-testid="stPageLink-NavLink"] p {{
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        white-space: nowrap;
        margin: 0;
        line-height: 1;
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) [data-testid="stPageLink-NavLink"]:hover {{
        background: var(--surface-hover);
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) [data-testid="stPageLink-NavLink"]:hover p {{
        color: var(--text) !important;
      }}
      /* Theme button: exact same 36px height as page links */
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) .stButton > button {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 10px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        height: 36px !important;
        min-height: 36px !important;
        max-height: 36px !important;
        padding: 0 12px !important;
        width: 100% !important;
        white-space: nowrap !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        line-height: 1 !important;
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) .stButton > button p {{
        margin: 0 !important;
        line-height: 1 !important;
      }}
      div[data-testid="stHorizontalBlock"]:has(.nav-hook) .stButton > button:hover {{
        background: var(--surface-hover) !important;
        border-color: var(--accent) !important;
        color: var(--text) !important;
      }}
      @media (max-width: 960px) {{
        .nav-tag {{ display: none; }}
      }}

      /* ---------- Atmospheric Hero ---------- */
      .hero {{
        position: relative;
        overflow: hidden;
        padding: 54px 36px 68px 36px;
        border-radius: 26px;
        margin-bottom: 28px;
        text-align: center;
        background: var(--hero-bg);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
      }}
      .hero-eyebrow {{
        display: inline-block;
        padding: 5px 13px;
        border-radius: 999px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--hero-sub);
        background: var(--hero-chip-bg);
        border: 1px solid var(--hero-chip-border);
        margin-bottom: 16px;
      }}
      .hero h1 {{
        margin: 0 auto;
        max-width: 720px;
        font-size: clamp(1.95rem, 4vw, 3.05rem);
        font-weight: 600;
        line-height: 1.12;
        letter-spacing: -0.035em;
        color: var(--hero-text) !important;
      }}
      .hero p {{
        margin: 14px auto 0 auto;
        max-width: 600px;
        font-size: 1.02rem;
        line-height: 1.62;
        color: var(--hero-sub) !important;
      }}
      .chips {{
        margin-top: 22px;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
      }}
      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 0.79rem;
        font-weight: 500;
        color: var(--hero-text);
        background: var(--hero-chip-bg);
        border: 1px solid var(--hero-chip-border);
        backdrop-filter: blur(8px);
      }}
      .hero-horizon {{
        position: absolute;
        left: 0;
        right: 0;
        bottom: -1px;
        width: 100%;
        height: 48px;
        pointer-events: none;
      }}

      /* ---------- Editorial Section Headers ---------- */
      .sec-head {{
        text-align: center;
        margin: 34px auto 22px auto;
        max-width: 640px;
      }}
      .sec-head.left {{
        text-align: left;
        margin-left: 0;
      }}
      .sec-eyebrow {{
        font-family: var(--font-mono);
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 8px;
        font-weight: 500;
      }}
      .sec-head h2 {{
        margin: 0;
        font-size: clamp(1.45rem, 2.6vw, 2.05rem);
        font-weight: 600;
        line-height: 1.2;
        color: var(--text) !important;
      }}
      .sec-head p {{
        margin: 10px 0 0 0;
        font-size: 0.96rem;
        line-height: 1.6;
        color: var(--text-secondary) !important;
      }}

      /* ---------- Calm Cards & Containers ---------- */
      [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        box-shadow: var(--shadow-soft);
      }}
      .calm-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: var(--shadow-soft);
        height: 100%;
        transition: border-color 0.2s ease, transform 0.2s ease;
      }}
      .calm-card:hover {{
        border-color: var(--border-strong);
      }}
      .stat-num {{
        font-family: var(--font-sans);
        font-variant-numeric: tabular-nums;
        font-size: 2.35rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        line-height: 1.05;
        color: var(--text);
        margin: 0 0 8px 0;
      }}
      .stat-title {{
        font-family: var(--font-sans);
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 6px 0;
      }}
      .stat-desc {{
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--text-secondary);
        margin: 0;
      }}

      .preview-box {{
        border-radius: 14px;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        padding: 16px;
        margin-bottom: 16px;
        min-height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }}
      .card-icon-wrap {{
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--border);
        margin-bottom: 12px;
      }}
      .card-h {{
        font-family: var(--font-sans);
        font-size: 1.03rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 6px 0;
        letter-spacing: -0.015em;
      }}
      .card-p {{
        font-size: 0.87rem;
        line-height: 1.58;
        color: var(--text-secondary);
        margin: 0;
      }}

      /* ---------- Animated Radial Gauge ---------- */
      .gauge {{
        position: relative;
        width: var(--size, 200px);
        height: var(--size, 200px);
        margin: 8px auto;
        animation: fill 1.3s cubic-bezier(.2,.8,.2,1) forwards;
      }}
      @keyframes fill {{ to {{ --p: var(--target); --n: var(--target-int); }} }}
      .gauge .ring {{
        position: absolute;
        inset: 0;
        border-radius: 50%;
        background: conic-gradient(var(--tint) calc(var(--p) * 1%), var(--border) 0);
        -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - var(--thick, 14px)), #000 calc(100% - var(--thick, 14px) + 1px));
                mask: radial-gradient(farthest-side, transparent calc(100% - var(--thick, 14px)), #000 calc(100% - var(--thick, 14px) + 1px));
      }}
      .gauge .gnum {{
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }}
      .gauge .pct::after {{
        counter-reset: pct var(--n);
        content: counter(pct) "%";
        font-family: var(--font-sans);
        font-variant-numeric: tabular-nums;
        font-weight: 600;
        font-size: calc(var(--size, 200px) * .22);
        line-height: 1;
        color: var(--tint);
      }}
      .gauge small {{
        margin-top: 6px;
        font-family: var(--font-mono);
        font-size: calc(var(--size, 200px) * .058);
        letter-spacing: 0.1em;
        color: var(--text-muted);
        text-transform: uppercase;
      }}

      /* ---------- Verdict Pills & Alerts ---------- */
      .verdict-pill {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 13px 18px;
        border-radius: 14px;
        font-family: var(--font-sans);
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin: 10px 0 6px 0;
        animation: pop .45s ease both;
      }}
      .verdict-pill.open {{
        background: var(--good-soft);
        color: var(--good);
        border: 1px solid var(--good);
      }}
      .verdict-pill.closed {{
        background: var(--danger-soft);
        color: var(--danger);
        border: 1px solid var(--danger);
      }}
      .verdict-pill.noface {{
        background: var(--surface-soft);
        color: var(--text-secondary);
        border: 1px solid var(--border-strong);
      }}
      @keyframes pop {{ from {{ transform: scale(.94); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}

      .alert-calm {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        border-radius: 16px;
        padding: 16px 20px;
        background: var(--danger-soft);
        border: 1px solid var(--danger);
        color: var(--danger);
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: 1rem;
        margin-top: 12px;
        animation: calm-pulse 1.8s ease-in-out infinite;
      }}
      @keyframes calm-pulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(220, 80, 80, 0.0); }}
        50% {{ box-shadow: 0 0 0 6px rgba(220, 80, 80, 0.14); }}
      }}

      /* ---------- Loading Wheel ---------- */
      .loader {{
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 30px 0;
      }}
      .spin {{
        width: 76px;
        height: 76px;
        border-radius: 50%;
        border: 4px solid var(--border);
        border-top-color: var(--accent);
        animation: rot .85s linear infinite;
      }}
      @keyframes rot {{ to {{ transform: rotate(360deg); }} }}
      .loader p {{
        margin-top: 14px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        color: var(--text-muted);
      }}

      /* ---------- History Token Strip ---------- */
      .hist-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 12px 0;
      }}
      .hist-tok {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 11px;
        border-radius: 999px;
        font-family: var(--font-mono);
        font-size: 0.74rem;
        font-weight: 500;
      }}
      .hist-tok.open {{
        background: var(--good-soft);
        color: var(--good);
        border: 1px solid var(--border);
      }}
      .hist-tok.closed {{
        background: var(--danger-soft);
        color: var(--danger);
        border: 1px solid var(--border);
      }}
      .hist-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: currentColor;
      }}

      /* ---------- Native Streamlit Widget Refinement ---------- */
      [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
        margin-bottom: 12px;
      }}
      [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
      }}
      button[data-baseweb="tab"] {{
        border-radius: 10px !important;
        padding: 7px 16px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        transition: all 0.16s ease !important;
      }}
      button[data-baseweb="tab"]:hover {{
        color: var(--text) !important;
        background: var(--surface-soft) !important;
      }}
      button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--text) !important;
        background: var(--surface) !important;
        border: 1px solid var(--border-strong) !important;
        box-shadow: var(--shadow-soft) !important;
      }}

      [data-testid="stExpander"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
      }}
      [data-testid="stExpander"] summary,
      [data-testid="stExpander"] summary p {{
        color: var(--text) !important;
        font-weight: 500 !important;
      }}

      .stButton > button {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 999px !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.25rem !important;
        transition: all 0.18s ease !important;
      }}
      .stButton > button:hover {{
        background: var(--surface-hover) !important;
        border-color: var(--accent) !important;
      }}

      [data-testid="stFileUploaderDropzone"] {{
        background: var(--surface) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: 18px !important;
        color: var(--text) !important;
      }}
      [data-testid="stFileUploaderDropzone"] * {{
        color: var(--text-secondary) !important;
      }}
      [data-testid="stFileUploaderDropzone"] button {{
        background: var(--cta-bg) !important;
        color: var(--cta-text) !important;
        border: none !important;
        border-radius: 999px !important;
      }}
      [data-testid="stCameraInput"] > div {{
        border-radius: 18px !important;
        overflow: hidden;
        border: 1px solid var(--border-strong) !important;
        background: var(--surface) !important;
      }}
      [data-testid="stCameraInput"] button {{
        border-radius: 999px !important;
      }}

      [data-testid="stAlert"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 14px !important;
        color: var(--text) !important;
      }}
      [data-testid="stAlert"] p {{
        color: var(--text-secondary) !important;
      }}

      [data-baseweb="select"] > div {{
        background: var(--surface) !important;
        border-color: var(--border-strong) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
      }}

      [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
      }}

      .calm-table-wrap {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow-soft);
        overflow-x: auto;
        margin-bottom: 14px;
      }}
      .calm-table-wrap table {{
        width: 100%;
        border-collapse: collapse;
        font-family: var(--font-body);
        font-variant-numeric: tabular-nums;
        font-size: 0.87rem;
        color: var(--text);
      }}
      .calm-table-wrap thead th {{
        background: var(--bg-elevated);
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        text-align: left;
        padding: 12px 14px;
        border-bottom: 1px solid var(--border-strong);
        white-space: nowrap;
      }}
      .calm-table-wrap tbody td {{
        padding: 11px 14px;
        border-bottom: 1px solid var(--border);
        color: var(--text);
        white-space: nowrap;
      }}
      .calm-table-wrap tbody tr:last-child td {{
        border-bottom: none;
      }}
      .calm-table-wrap tbody tr:hover td {{
        background: var(--surface-hover);
      }}

      hr {{
        border-color: var(--border) !important;
      }}

      /* ---------- CTA & Footer ---------- */
      .cta-banner {{
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 42px 32px;
        margin-top: 34px;
        background: var(--hero-bg);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        text-align: center;
      }}
      .cta-banner h3 {{
        margin: 0;
        font-size: clamp(1.35rem, 2.4vw, 1.85rem);
        font-weight: 600;
        color: var(--hero-text) !important;
      }}
      .cta-banner p {{
        margin: 10px auto 0 auto;
        max-width: 540px;
        color: var(--hero-sub) !important;
        font-size: 0.95rem;
        line-height: 1.58;
      }}

      .site-footer {{
        margin-top: 36px;
        padding-top: 22px;
        border-top: 1px solid var(--border);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        font-size: 0.82rem;
        color: var(--text-muted);
      }}
      .site-footer strong {{
        color: var(--text-secondary);
        font-weight: 600;
      }}

      /* ---------- Mobile / Narrow Screens ---------- */
      @media (max-width: 760px) {{
        .main .block-container,
        [data-testid="stMainBlockContainer"] {{
          padding: 0.7rem 0.75rem 2.8rem !important;
        }}
        .hero {{
          padding: 32px 18px 48px 18px;
          border-radius: 20px;
          margin-bottom: 18px;
        }}
        .gauge.big {{
          --size: 168px !important;
          --thick: 13px !important;
        }}
        .stat-num {{
          font-size: 1.85rem;
        }}
        button[data-baseweb="tab"] {{
          padding: 6px 11px !important;
          font-size: 0.82rem !important;
        }}
        .nav-tag {{ display: none; }}
      }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_navbar(active: str = "home") -> None:
    """Render the shared top bar: brand on left, 4 aligned nav items on right."""
    c_brand, c_links = st.columns([2.2, 5.6], gap="small")
    with c_brand:
        st.markdown(
            '<span class="nav-hook"></span>'
            '<div class="nav-brand">'
            f'<span class="nav-brand-badge">{svg_icon("eye", 16, 1.8)}</span>'
            '<span>DrowsAI</span>'
            '<span class="nav-tag">v2</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_links:
        n1, n2, n3, n4 = st.columns(4, gap="small")
        with n1:
            st.page_link("Home.py", label="Detector", use_container_width=True)
        with n2:
            st.page_link("pages/1_Model_Results.py", label="Model Results", use_container_width=True)
        with n3:
            st.page_link("pages/2_Live_Monitor.py", label="Live Monitor", use_container_width=True)
        with n4:
            if st.button("Theme", key=f"theme_toggle_{active}", use_container_width=True):
                toggle_theme()
                st.rerun()


def hero(title: str, subtitle: str = "", chips=(), eyebrow: str = "DRIVER DROWSINESS SYSTEM"):
    """Render the atmospheric horizon hero section with subtle ridge silhouette."""
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    ridge_svg = (
        '<svg class="hero-horizon" viewBox="0 0 1200 60" preserveAspectRatio="none" aria-hidden="true">'
        '<path d="M0,44 C240,14 420,52 620,30 C820,8 990,46 1200,22 L1200,60 L0,60 Z" '
        'fill="var(--horizon-ridge-mid)"></path>'
        '<path d="M0,52 C280,26 480,58 700,38 C910,18 1060,48 1200,36 L1200,60 L0,60 Z" '
        'fill="var(--horizon-ridge)"></path>'
        "</svg>"
    )
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1>"
        f"<p>{subtitle}</p>"
        f'<div class="chips">{chips_html}</div>'
        f"{ridge_svg}"
        f"</div>",
        unsafe_allow_html=True,
    )


def section_header(eyebrow: str, title: str, subtitle: str = "", left: bool = False) -> None:
    """Render an editorial section header with eyebrow, display title, and subtitle."""
    cls = "sec-head left" if left else "sec-head"
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="{cls}"><div class="sec-eyebrow">{eyebrow}</div><h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_footer(note: str = "MRL Eye Dataset  |  Subject-wise Split  |  64x64 Grayscale Inference") -> None:
    """Render a calm, understated footer across all pages."""
    t = get_theme()
    st.markdown(
        f'<div class="site-footer">'
        f'<div><strong>DrowsAI</strong>  -  Driver Drowsiness &amp; Eye-State Monitor</div>'
        f'<div>{note}  |  Active theme: {t["name"]}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def gauge(p: float, size: int = 200, thick: int = 15, label: str = "", tint: str | None = None, cls: str = "") -> str:
    """Animated circular gauge (HTML string) that counts from 0% up to p (0-1)."""
    t = get_theme()
    pct = max(0.0, min(100.0, p * 100))
    tint = tint or (t["danger"] if p >= 0.5 else t["good"])
    return (
        f'<div class="gauge {cls}" style="--size:{size}px;--thick:{thick}px;--tint:{tint};'
        f'--target:{pct:.1f};--target-int:{round(pct)};">'
        f'<div class="ring"></div><div class="gnum"><span class="pct"></span><small>{label}</small></div></div>'
    )


def render_table(data) -> None:
    """Render a DataFrame or pandas Styler as a theme-matched HTML table."""
    styler = data.style if isinstance(data, pd.DataFrame) else data
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    html = styler.to_html()
    st.markdown(f'<div class="calm-table-wrap">{html}</div>', unsafe_allow_html=True)


def style_altair(chart: alt.Chart) -> alt.Chart:
    """Apply active Midnight Blue or Warm Skin theme styling to an Altair chart."""
    t = get_theme()
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=t["text_secondary"],
            titleColor=t["text_muted"],
            gridColor=t["chart_grid"],
            domainColor=t["border_strong"],
            tickColor=t["border_strong"],
            labelFont="Inter",
            titleFont="Plus Jakarta Sans",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=t["text_secondary"],
            titleColor=t["text_muted"],
            labelFont="Inter",
            titleFont="Plus Jakarta Sans",
        )
        .configure_title(
            color=t["text"],
            font="Plus Jakarta Sans",
            fontSize=14,
            fontWeight=600,
            anchor="start",
        )
    )


# ---------- audio ----------
@st.cache_data
def make_alarm_wav() -> bytes:
    """2.4 s wailing siren alarm."""
    sr, dur, period = 22050, 2.4, 0.8
    t = np.arange(int(sr * dur)) / sr
    tri = 1 - np.abs(2 * ((t % period) / period) - 1)
    phase = 2 * np.pi * np.cumsum(650 + 750 * tri) / sr
    phase -= (phase[-1] - 2 * np.pi * round(phase[-1] / (2 * np.pi))) * (t / dur)
    sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    sig = sig / np.abs(sig).max() * 0.92
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ---------- files, config, models ----------
@st.cache_resource
def load_config():
    p = MODEL_DIR / "phase2_config.json"
    return json.load(open(p)) if p.exists() else {"model_name": "best_model", "threshold": 0.5}


@st.cache_resource(show_spinner="Loading model ...")
def load_keras(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def best_model():
    cfg = load_config()
    return load_keras(str(MODEL_DIR / "best_model.keras")), float(cfg["threshold"]), cfg


def load_csv(name: str):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def available_models():
    """{model name: (path, threshold)} for every deep model that has a saved .keras file."""
    comp, out = load_csv("phase2_model_comparison.csv"), {}
    if comp is not None:
        for _, r in comp[comp.family == "Deep learning"].iterrows():
            f = MODEL_DIR / f"p2_{r.model.replace(' ', '_').replace('(', '').replace(')', '')}.keras"
            if f.exists():
                out[r.model] = (str(f), float(r.threshold))
    if not out and (MODEL_DIR / "best_model.keras").exists():
        cfg = load_config()
        out[cfg["model_name"]] = (str(MODEL_DIR / "best_model.keras"), float(cfg["threshold"]))
    return out


# ---------- image handling ----------
def read_image(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    img.thumbnail((1280, 1280))
    return np.array(img)


@st.cache_resource
def _cascades():
    base = cv2.data.haarcascades
    return (
        cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(base + "haarcascade_eye.xml"),
    )


def _square(x, y, w, h, W, H, pad=0.25):
    s = max(w, h) * (1 + pad)
    cx, cy = x + w / 2, y + h / 2
    return (
        int(max(cx - s / 2, 0)),
        int(max(cy - s / 2, 0)),
        int(min(cx + s / 2, W)),
        int(min(cy + s / 2, H)),
    )


def find_eyes(rgb):
    """Find the face with a Haar cascade, then place two square eye boxes from face geometry.
    (The Haar *eye* cascade is trained on open eyes and often misses closed ones, so it is not used.)"""
    face_c, _ = _cascades()
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    H, W = gray.shape
    faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return []
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    side = 0.36 * fw
    boxes = []
    for cx in (fx + 0.30 * fw, fx + 0.70 * fw):  # left / right eye centre
        cy = fy + 0.39 * fh
        boxes.append(
            (
                int(max(cx - side / 2, 0)),
                int(max(cy - side / 2, 0)),
                int(min(cx + side / 2, W)),
                int(min(cy + side / 2, H)),
            )
        )
    return boxes


def to_model_input(rgb, box=None):
    """Grayscale -> (square) crop -> 64x64 -> [0,1]. Same recipe as the training data."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if box is not None:
        x0, y0, x1, y1 = box
        gray = gray[y0:y1, x0:x1]
    h, w = gray.shape
    if h != w:  # pad to square, keep the aspect ratio
        d = abs(h - w)
        a, b = d // 2, d - d // 2
        gray = (
            cv2.copyMakeBorder(gray, a, b, 0, 0, cv2.BORDER_REPLICATE)
            if w > h
            else cv2.copyMakeBorder(gray, 0, 0, a, b, cv2.BORDER_REPLICATE)
        )
    gray = np.asarray(Image.fromarray(gray).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
    return (gray.astype("float32") / 255.0)[..., None]


def get_crops(rgb, mode):
    """mode: 'auto' (find eyes in a face photo) or 'crop' (image already is an eye)."""
    if mode == "crop":
        return [to_model_input(rgb)], [None]
    boxes = find_eyes(rgb)
    return [to_model_input(rgb, b) for b in boxes], boxes


def predict(model, crops):
    return model.predict(np.stack(crops), verbose=0).ravel() if crops else np.array([])


def annotate(rgb, boxes, closed_flags):
    out = rgb.copy()
    for b, c in zip(boxes, closed_flags):
        if b is not None:
            cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), RED if c else GREEN, max(2, rgb.shape[1] // 200))
    return out
