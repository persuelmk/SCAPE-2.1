#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TABLES = (
    "pas_candidates.tsv", "pas_candidates_all.tsv", "r1_seed_audit.tsv",
    "molecule_assignments.tsv.gz", "read_pas_assignments.tsv.gz",
    "region_qc.tsv", "cell_pas_counts.tsv.gz", "pas_quantification.tsv",
    "pasite.csv.gz",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("observed", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = {"reference": str(args.reference), "observed": str(args.observed), "tables": {}}
    all_pass = True
    for name in TABLES:
        sep = "," if name.endswith(".csv.gz") else "\t"
        a = pd.read_csv(args.reference / name, sep=sep)
        b = pd.read_csv(args.observed / name, sep=sep)
        schema_equal = list(a.columns) == list(b.columns)
        shape_equal = a.shape == b.shape
        exact_columns, float_columns, failures = [], [], []
        if schema_equal and shape_equal:
            for column in a.columns:
                if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
                    av = a[column].to_numpy(float); bv = b[column].to_numpy(float)
                    same = np.allclose(av, bv, rtol=1e-6, atol=1e-8, equal_nan=True)
                    float_columns.append(column)
                else:
                    same = a[column].fillna("<NA>").astype(str).equals(b[column].fillna("<NA>").astype(str))
                    exact_columns.append(column)
                if not same:
                    failures.append(column)
        else:
            failures.append("schema_or_shape")
        passed = not failures
        all_pass &= passed
        report["tables"][name] = {
            "passed": passed, "schema_equal": schema_equal, "shape_equal": shape_equal,
            "reference_shape": list(a.shape), "observed_shape": list(b.shape),
            "exact_columns": exact_columns, "tolerant_float_columns": float_columns,
            "failed_columns": failures,
        }
    reference_summary = json.loads((args.reference / "run_summary.json").read_text())
    observed_summary = json.loads((args.observed / "run_summary.json").read_text())
    report["run_summary_exact"] = reference_summary == observed_summary
    all_pass &= report["run_summary_exact"]
    report["passed"] = all_pass
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
