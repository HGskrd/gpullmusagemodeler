"""Import paths shared by every test module.

Tests import both repository-root modules (`app`, `state`, `calc`) and helpers
that sit beside them in `tests/` (`app_factory`, `characterization_support`).
Only `python -m pytest` put the repository root on `sys.path`, via the CWD entry
that `-m` adds; a bare `pytest` invocation failed collection on all 17 modules.
Make both entries explicit so the suite runs the same way either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

for path in (REPO_ROOT, TESTS_DIR):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
