"""Number and currency formatting shared by templates and the text report.

Registered as Jinja filters by ``create_app`` and used directly by
``presentation.reports``; kept here so the two cannot drift apart.

Money rules (one format per magnitude, so a column never mixes styles):
zero is ``$0``; anything under a dollar is ``<$1``; whole dollars with
thousands separators below $10k; then one decimal with a k/M/B suffix.
Per-million-token prices are the exception: they are routinely below a
dollar, so ``fmt_price_pm`` keeps cents and never rounds a price to zero.
"""

from __future__ import annotations

import math
import re


def fmt_num(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


def fmt_int(n) -> str:
    """Whole number with thousands separators: 5000000 -> '5,000,000'."""
    return f"{float(n or 0.0):,.0f}"


def fmt_money(value):
    value = float(value or 0.0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.1f}M"
    if value >= 1e4:
        return f"{sign}${value / 1e3:.1f}k"
    if value >= 0.5:
        return f"{sign}${value:,.0f}"
    if value > 0:
        return f"{sign}<$1"
    return "$0"


def fmt_price_pm(value) -> str:
    """Price per million tokens: '$1.02', '<$0.01', or '$0' only when truly zero."""
    value = float(value or 0.0)
    if value <= 0:
        return "$0"
    if value < 0.01:
        return "<$0.01"
    return f"${value:,.2f}"


def fmt_mult(value) -> str:
    """Ratio such as value/cost: '96×' at or above 10, '1.4×' below."""
    value = float(value or 0.0)
    if not math.isfinite(value):
        return "—"
    if abs(value) >= 10:
        return f"{value:,.0f}×"
    return f"{value:.1f}×"


def layout_label(tp: int, pp: int, dp: int) -> str:
    """Parallel layout without the x1 factors: TP2, TP2×DP2, TP1 for a single GPU."""
    parts = [f"{name}{n}" for name, n in (("TP", tp), ("PP", pp), ("DP", dp)) if n > 1]
    return "×".join(parts) or "TP1"


def log2int(n):
    return int(math.log2(n)) if n > 0 else 0


_ACTIVE_PARAMS = re.compile(r"-A[\d.]+B$")


def short_model_name(name: str) -> str:
    """Drop the MoE active-parameter suffix: "Qwen 3.5 35B-A3B" -> "Qwen 3.5 35B"."""
    return _ACTIVE_PARAMS.sub("", str(name))
