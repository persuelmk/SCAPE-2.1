import csv
from scape2.contracts import PASCandidate

def load_legacy_candidates(path):
    with open(path,newline="") as handle:
        rows=list(csv.DictReader(handle,delimiter="	"))
    return [PASCandidate(r["pas_id"],r["chrom"],int(r["position"]),r["strand"],r["source"],
                         None if r["beta"] in ("","NA") else float(r["beta"])) for r in rows]
