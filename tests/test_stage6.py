"""Integrity and reproducibility checks for the Stage 6 saved-data synthesis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantum_transport.synthesis import (
    SUMMARY_FIELDS,
    analyze_saved_results,
    load_csv,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_final_summary_artifacts_have_required_schema_and_finite_metrics() -> None:
    csv_path = RESULTS / "final_stage_summary.csv"
    json_path = RESULTS / "final_stage_summary.json"
    assert csv_path.is_file()
    assert json_path.is_file()

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert [row["stage"] for row in rows] == [f"Stage {n}" for n in range(1, 6)]
    assert all(all(row[field].strip() for field in SUMMARY_FIELDS) for row in rows)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary_rows"] == rows
    def check_finite(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, list):
            for item in value:
                check_finite(item)
        elif isinstance(value, float):
            assert np.isfinite(value)
    check_finite(data["metrics"])


def test_all_saved_stage_result_tables_load() -> None:
    names = (
        "stage1_transmission.csv",
        "stage2_transmission.csv",
        "stage2_dos.csv",
        "stage2_resonances.csv",
        "stage2_resonance_ldos.csv",
        "stage3_disorder_realizations.csv",
        "stage3_single_realization.csv",
        "stage3_ensemble.csv",
        "stage3_length_scaling.csv",
        "stage4_modes.csv",
        "stage4_multichannel.csv",
        "stage5_qpc_gate_scan.csv",
        "stage5_qpc_ldos.csv",
        "stage5_qpc_potential.csv",
        "stage5_qpc_transmission.csv",
    )
    for name in names:
        assert load_csv(RESULTS / name)


def test_saved_data_synthesis_is_deterministic() -> None:
    first = analyze_saved_results(RESULTS)
    second = analyze_saved_results(RESULTS)
    assert first == second


def test_stage1_clean_matched_chain_is_unity_in_safe_windows() -> None:
    metrics = analyze_saved_results(RESULTS)["metrics"]["stage1"]["window_metrics"]
    assert metrics["-1.5_to_1.5"]["points"] == 301
    assert metrics["-1_to_1"]["points"] == 201
    assert metrics["-0.5_to_0.5"]["points"] == 101
    assert metrics["-1.5_to_1.5"]["clean_max_abs_deviation_from_one"] < 1e-12


def test_stage2_resonance_is_resolved_and_central_ldos_is_selective() -> None:
    metrics = analyze_saved_results(RESULTS)["metrics"]["stage2"]
    samples = metrics["resonance_samples"]
    peak = next(row for row in samples if row["energy"] == metrics["selected_resonance_energy"])
    detuned = [row for row in samples if row is not peak]
    assert peak["double_barrier_transmission"] > 0.99
    assert all(peak["central_well_ldos_fraction"] > 5.0 * row["central_well_ldos_fraction"] for row in detuned)
    assert metrics["maximum_absolute_site_sum_vs_saved_total_dos_difference"] < 1e-11
    neighborhood = metrics["five_point_transmission_neighborhood"]
    assert neighborhood[2]["transmission"] > neighborhood[1]["transmission"]
    assert neighborhood[2]["transmission"] > neighborhood[3]["transmission"]


def test_stage3_mean_median_typical_and_mean_log_decrease_with_length() -> None:
    metrics = analyze_saved_results(RESULTS)["metrics"]["stage3"]
    assert metrics["length_metric_groups_checked"] == 6
    assert metrics["monotonicity_violations"] == []


def test_stage4_transmission_matches_open_modes_away_from_thresholds() -> None:
    metrics = analyze_saved_results(RESULTS)["metrics"]["stage4"]
    assert metrics["threshold_exclusion_distance"] == 0.05
    assert metrics["retained_rows"] == 671
    assert metrics["maximum_absolute_T_minus_N_open"] < 1e-12


def test_stage5_gate_scan_and_incoming_channel_bound() -> None:
    metrics = analyze_saved_results(RESULTS)["metrics"]["stage5"]
    assert metrics["nonmonotone_gate_scan_energies"] == []
    assert len(metrics["nonincreasing_gate_scan_energies"]) == 5
    assert metrics["maximum_positive_T_minus_N_open"] < 1e-12
    assert metrics["maximum_absolute_T_minus_N_open"] > 1.0
    scan = metrics["gate_scan_at_energy_minus_2"]
    assert [row["N_open_lead"] for row in scan] == [3] * 5
    assert [row["N_local_open_centre"] for row in scan] == [3, 2, 2, 1, 1]
    assert [row["site_count"] for row in metrics["ldos_sums_at_energy_minus_2"]] == [240, 240]
