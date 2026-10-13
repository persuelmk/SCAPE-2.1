"""Frozen SCAPE-v2.1-compatible defaults and validation."""

DEFAULTS = {
    "assignment_mode": "high-confidence",
    "cluster_radius": 50,
    "direct_mapq": 30,
    "direct_softclip": 2,
    "endpoint_bandwidth": 3.0,
    "gene_tag": "GX",
    "max_r1_reassign_distance": 25,
    "merge_distance": 25,
    "min_direct_support": 3,
    "min_final_endpoint_cells": 2,
    "min_final_endpoint_umis": 5,
    "min_final_umis": 5,
    "min_final_weight": 0.01,
    "min_pa_gap": 20,
    "min_posterior": 0.60,
    "min_posterior_margin": 0.20,
    "min_protected_r1_cell_fraction": 0.20,
    "min_protected_r1_fraction": 0.20,
    "min_reads": 50,
    "n_max_apa": 5,
    "n_min_apa": 1,
    "paired_bam_mode": False,
    "r1_caller": "cdf",
    "r1_fetch_extension": 200,
    "r1_known_terminal_clip": 0,
    "r1_orientation": "ignore",
    "r1_structure_policy": "require",
    "r1_structure_tag": "XT",
    "r2_min_mapq": 20,
    "semi_mapq": 20,
    "semi_softclip": 10,
    "verbose": False,
}


def validate(options):
    if options["n_min_apa"] < 1 or options["n_max_apa"] < options["n_min_apa"]:
        raise ValueError("require 1 <= n_min_apa <= n_max_apa")
    if options["min_reads"] < 0:
        raise ValueError("min_reads must be non-negative")
    if options["assignment_mode"] not in {"scape-compatible", "high-confidence"}:
        raise ValueError("invalid assignment_mode")
    return options
