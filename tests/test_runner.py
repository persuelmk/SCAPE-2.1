import gzip
from pathlib import Path

from scape2.runner import load_barcodes, load_regions


def test_regions(tmp_path):
    path = tmp_path / "regions.tsv"
    path.write_text("chr1\t10\t20\t+\n")
    assert load_regions(path) == ["chr1\t10\t20\t+"]


def test_gzip_barcodes_are_sorted_unique(tmp_path):
    path = tmp_path / "barcodes.tsv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("B\nA\nB\n")
    assert load_barcodes(path) == ["A", "B"]
