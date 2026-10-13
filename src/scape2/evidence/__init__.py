from collections import Counter, defaultdict
import hashlib
import json
import pysam
from scape2.contracts import normalize_gene_id, identity_present
from scape2.io import audit_bam

def read_tags(read):
    return {tag:read.get_tag(tag) if read.has_tag(tag) else None for tag in ("CB","UB","GX","DT","NH","XT")}

def parse_evidence(path, regions, library_id, mode="r2", min_mapq=0, max_records=100000):
    if not library_id or mode not in ("r2","paired-r2") or min_mapq<0 or max_records<1:
        raise ValueError("invalid evidence parameters")
    audit_bam(path,regions)
    rows=[]; groups=defaultdict(list); reasons=Counter()
    with pysam.AlignmentFile(str(path),"rb") as bam:
        for region in regions:
            for read in bam.fetch(region.chrom,region.start0,region.end0):
                if len(rows)>=max_records:
                    raise ValueError("record safety cap exceeded")
                tags=read_tags(read)
                dt=tags["DT"]
                if dt is not None and not isinstance(dt,str):
                    raise ValueError("DT must be string when present")
                reason="accepted"
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    reason="nonprimary"
                elif (mode=="paired-r2" and not read.is_read2) or (mode=="r2" and read.is_read1):
                    reason="wrong_read_role"
                elif read.mapping_quality<min_mapq:
                    reason="low_mapq"
                elif not all(identity_present(tags[k]) for k in ("CB","UB","GX")):
                    reason="missing_identity"
                row={"library_id":library_id,"region_id":region.region_id,"gene_strand":region.strand,
                     "qname":read.query_name,"flag":read.flag,"chrom":read.reference_name,
                     "start0":read.reference_start,"end0":read.reference_end,
                     "alignment_strand":"-" if read.is_reverse else "+",
                     "cigar":read.cigarstring,"mapq":read.mapping_quality,
                     "aligned_query_length":read.query_alignment_length,
                     "tags":tags,"gene_id_normalized":normalize_gene_id(tags["GX"]),
                     "dt_length":None if dt is None else len(dt),
                     "status":reason,"molecule_id":None}
                if reason=="accepted":
                    key=(library_id,region.region_id,str(tags["GX"]),str(tags["CB"]),str(tags["UB"]))
                    row["molecule_id"]=hashlib.sha256(json.dumps(key,separators=(",",":")).encode()).hexdigest()
                    groups[key].append(row)
                reasons[reason]+=1
                rows.append(row)
    rows.sort(key=lambda r:json.dumps(r,sort_keys=True))
    molecules=[]
    for key,items in sorted(groups.items()):
        molecules.append({"library_id":key[0],"region_id":key[1],"gene_id_raw":key[2],
                          "cell_barcode":key[3],"umi":key[4],
                          "molecule_id":items[0]["molecule_id"],
                          "n_alignment_records":len(items),
                          "n_read_ids":len({r["qname"] for r in items}),
                          "read_ids":sorted({r["qname"] for r in items}),
                          "alignment_evidence":[{k:r[k] for k in ("qname","start0","end0","alignment_strand","cigar","mapq","tags","dt_length")} for r in sorted(items,key=lambda r:json.dumps(r,sort_keys=True))],
                          "pas_id":None,"assignment_status":"NOT_IMPLEMENTED","posterior":None})
    qc={"n_alignment_region_records":len(rows),"n_distinct_read_ids":len({r["qname"] for r in rows}),
        "record_status_counts":dict(sorted(reasons.items())),"n_provisional_molecules":len(molecules),
        "n_keyed_alignment_records":sum(m["n_alignment_records"] for m in molecules),
        "n_unquantified_molecules":len(molecules),"n_assigned_molecules":None,
        "quantification_status":"NOT_IMPLEMENTED","scope":"library+region+raw_GX+CB+UB",
        "note":"No representative selection, gene-strand filtering, R1 join, geometry collapse or PAS inference"}
    assert len(rows)==sum(reasons.values())
    assert qc["n_keyed_alignment_records"]==reasons["accepted"]
    return rows,molecules,qc
