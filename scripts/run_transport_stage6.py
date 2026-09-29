"""Write final Stage 1–5 synthesis tables and overview figure from saved data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_transport.synthesis import create_overview_figure, write_final_summary


def main() -> None:
    """Regenerate only the small Stage 6 synthesis artifacts."""
    results = PROJECT_ROOT / "results"
    analysis = write_final_summary(results)
    create_overview_figure(results, PROJECT_ROOT / "figures" / "final_project_overview.png")
    print(f"Wrote {len(analysis['summary_rows'])} stage summary rows from saved Stage 1–5 data.")
    print(f"Stage 4 retained samples: {analysis['metrics']['stage4']['retained_rows']}")
    print(f"Stage 5 monotone gate scans: {len(analysis['metrics']['stage5']['nonincreasing_gate_scan_energies'])}/"
          f"{len(analysis['metrics']['stage5']['gate_scan_energies'])}")


if __name__ == "__main__":
    main()
