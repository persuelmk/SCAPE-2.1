"""Evidence contracts: lossless alignment geometry, no PAS inference."""
from dataclasses import dataclass
import re

SCHEMA_VERSION = "scape2-evidence-v1"

@dataclass(frozen=True)
class Region:
    region_id: str
    chrom: str
    start0: int
    end0: int
    strand: str

    def __post_init__(self):
        if not self.region_id or not self.chrom or self.strand not in ("+", "-"):
            raise ValueError("region_id, chrom and explicit +/- strand required")
        if self.start0 < 0 or self.end0 <= self.start0:
            raise ValueError("region must be nonempty 0-based half-open")

    @classmethod
    def from_legacy(cls, region_id, chrom, left1, right1, strand):
        return cls(region_id, chrom, int(left1)-1, int(right1), strand)

    def legacy_bounds(self):
        return self.start0+1, self.end0

@dataclass(frozen=True)
class PASCandidate:
    pas_id: str
    chrom: str
    position1: int
    strand: str
    source: str
    beta: float | None = None

    def __post_init__(self):
        if self.strand not in ("+", "-") or self.position1 < 1 or not self.chrom or not self.pas_id:
            raise ValueError("invalid 1-based PAS point")
        if self.beta is not None:
            import math
            if not math.isfinite(self.beta) or self.beta < 0:
                raise ValueError("beta must be finite nonnegative or None")

def normalize_gene_id(raw):
    """Annotation only. Never use normalized ID to merge molecule identity."""
    if raw is None:
        return None
    if re.fullmatch(r"ENS[A-Z]*G[0-9]+[.][0-9]+", raw):
        return raw.rsplit(".", 1)[0]
    return raw

def identity_present(value):
    return value not in (None, "", "-", ".", "NA")
