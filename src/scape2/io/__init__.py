import csv
from pathlib import Path
import pysam
from scape2.contracts import Region

def load_regions(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="	")
        required = {"region_id", "chrom", "start0", "end0", "strand"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("regions require region_id,chrom,start0,end0,strand")
        result = [Region(r["region_id"], r["chrom"], int(r["start0"]), int(r["end0"]), r["strand"]) for r in reader]
    if not result or len({r.region_id for r in result}) != len(result):
        raise ValueError("regions must be nonempty with unique region_id")
    # Prevent double counting alignments within one strand across overlapping regions.
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if a.chrom == b.chrom and a.strand == b.strand and max(a.start0,b.start0) < min(a.end0,b.end0):
                raise ValueError("overlapping same-strand regions require an explicit later policy")
    return sorted(result, key=lambda r:(r.chrom,r.start0,r.end0,r.strand,r.region_id))

def index_path(bam):
    p = Path(bam)
    for q in (Path(str(p)+".bai"), p.with_suffix(".bai"), Path(str(p)+".csi"), p.with_suffix(".csi")):
        if q.is_file():
            return q
    raise ValueError("BAM index missing; source indexes are never created automatically")

def audit_bam(path, regions):
    index = index_path(path)
    with pysam.AlignmentFile(str(path), "rb", index_filename=str(index)) as bam:
        if not bam.has_index():
            raise ValueError("BAM index unreadable")
        if bam.header.to_dict().get("HD", {}).get("SO") != "coordinate":
            raise ValueError("BAM must declare coordinate sort order")
        lengths = dict(zip(bam.references,bam.lengths))
        for r in regions:
            if r.chrom not in lengths or r.end0 > lengths[r.chrom]:
                raise ValueError("region outside BAM reference dictionary")
        return {"index":str(index.resolve()),"references":lengths,"regions":len(regions),
                "scope":"header/index validation; no exhaustive sort validation"}

def subset_bam(path, regions, output, max_records):
    """Bounded extraction; no selection based on new scientific output."""
    audit_bam(path, regions)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    # Merge windows regardless of strand so no alignment is duplicated on extraction.
    windows=[]
    for r in sorted(regions,key=lambda r:(r.chrom,r.start0,r.end0)):
        if windows and r.chrom==windows[-1][0] and r.start0 <= windows[-1][2]:
            windows[-1][2]=max(windows[-1][2],r.end0)
        else:
            windows.append([r.chrom,r.start0,r.end0])
    count=0
    seen_offsets=set()
    with pysam.AlignmentFile(str(path),"rb") as source:
        # Reference dictionary order, not lexicographic chr order, defines BAM sort.
        windows.sort(key=lambda w:(source.get_tid(w[0]),w[1]))
        with pysam.AlignmentFile(str(output),"wb",header=source.header) as dest:
            for chrom,start,end in windows:
                for read in source.fetch(chrom,start,end):
                    offset=source.tell()
                    if offset in seen_offsets:
                        continue
                    seen_offsets.add(offset)
                    count+=1
                    if count>max_records:
                        raise ValueError("fixture safety cap exceeded; partial output must not be used")
                    dest.write(read)
    pysam.index(str(output))
    return {"records":count}
