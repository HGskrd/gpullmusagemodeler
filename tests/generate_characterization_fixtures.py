"""Regenerate estimator goldens after intentionally reviewed output changes."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from characterization_support import (  # noqa: E402
    FIXTURE_DIR,
    canonical_json,
    chart_outputs,
    projection_outputs,
)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "charts.json": chart_outputs(),
        "projections.json": projection_outputs(),
    }
    for name, output in fixtures.items():
        path = FIXTURE_DIR / name
        path.write_text(canonical_json(output), encoding="utf-8")
        print(f"wrote {path.relative_to(FIXTURE_DIR.parents[2])}")


if __name__ == "__main__":
    main()
