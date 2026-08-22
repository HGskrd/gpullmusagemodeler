"""Number and currency formatting shared by templates and the text report.

Registered as Jinja filters by ``create_app`` and used directly by
``presentation.reports``; kept here so the two cannot drift apart.
"""

from __future__ import annotations

import math


def fmt_num(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


def fmt_money(value):
    value = float(value or 0.0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.1f}k"
    if value >= 100:
        return f"{sign}${value:,.0f}"
    if value >= 1:
        return f"{sign}${value:,.2f}"
    return f"{sign}${value:,.3f}"


def log2int(n):
    return int(math.log2(n)) if n > 0 else 0
