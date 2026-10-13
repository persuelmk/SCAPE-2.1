"""Legacy-equivalent downstream assembly; no model fitting or new UMI rules."""
import copy,csv,gzip,json,math
from collections import defaultdict,Counter
from pathlib import Path

ASSIGNMENT_FIELDS=('pas_id','posterior','noise_posterior','assignment','assignment_rule','unassigned_reason')
COUNT_COLUMNS=['region','cell_barcode','pas_id','direct_r1_count','r2_assigned_count','total_count','hard_umi_count','expected_umi_count']
QUANT_COLUMNS=['pas_id','direct_r1_count','r2_assigned_count','total_count']
def present(v):return v is not None and not (isinstance(v,float) and math.isnan(v))
def assemble_region(region,frame,read_rows,candidates,components,seeds,seed_audit,z,options,contribution_sink=None):
    """Apply native interval/argmax rules, then exact-key molecule contributions."""
    hard=defaultdict(int);direct=defaultdict(int);r2=defaultdict(int);expected=defaultdict(float);members=[];n_pas=len(candidates)
    retained=[c for c in components if c['candidate_status']=='retained']
    assert len(retained)==n_pas
    for i,row in enumerate(frame):
        p=z[i];best=max(range(len(p)),key=p.__getitem__);prob=float(p[best]);ranked=sorted(p,reverse=True);margin=ranked[0]-ranked[1] if len(ranked)>1 else ranked[0];reason=''
        if row['tier']!='R2-only' and present(row['direct_pas']):
            eligible=[k for k,(c,a) in enumerate(zip(candidates,retained)) if c['source']=='R1-direct' and a.get('assignment_interval_start') is not None and a.get('assignment_interval_end') is not None and float(a['assignment_interval_start'])<=float(row['direct_pas'])<=float(a['assignment_interval_end'])]
            assigned=bool(eligible);rule='r1_cdf_valley_interval'
            if eligible:best=min(eligible,key=lambda k:abs(float(row['direct_pas'])-candidates[k]['position']));prob=float(p[best])
            else:reason='no_cdf_pas_interval'
        elif row['tier']=='R2-only' and options['assignment_mode']=='high-confidence':
            assigned=best<n_pas and prob>=options['min_posterior'] and margin>=options.get('min_posterior_margin',.2);rule='r2_posterior_and_margin'
            if best<n_pas and not assigned:reason='ambiguous_r2_assignment'
        else:assigned=best<n_pas;rule='r1_interval' if row['tier']!='R2-only' else 'argmax_pas'
        members.append(dict(row,pas_id=candidates[best]['pas_id'] if assigned else 'UNASSIGNED',posterior=prob,noise_posterior=float(p[n_pas]),assignment='assigned' if assigned else 'unassigned',assignment_rule=rule,unassigned_reason=reason))
        if assigned:
            hard[(row['cb'],best)]+=1
            (r2 if row['tier']=='R2-only' else direct)[(row['cb'],best)]+=1
        for k in range(n_pas):expected[(row['cb'],k)]+=float(p[k])
        if contribution_sink is not None:contribution_sink.append(dict(qname=row['qname'],cell=row['cb'],gene=row['gene'],UMI=row['umi'],best_index=best,assigned=bool(assigned),tier=row['tier'],hard=[int(assigned and j==best) for j in range(n_pas)],soft=list(map(float,p[:n_pas])),noise=float(p[n_pas]),unassigned=int(not assigned)))
    by_key={(m['cb'],m['gene'],m['umi']):m for m in members};reads=[]
    for raw in read_rows:
        read=copy.deepcopy(raw);member=by_key.get(tuple(read.pop('molecule_key')))
        if member is None:continue
        reads.append(dict(read_id=read.pop('qname'),**read,**{k:member[k] for k in ASSIGNMENT_FIELDS}))
    relations=Counter(r['relation'] for r in read_rows if r['relation']!='missing')
    if relations:
        mode,count=relations.most_common(1)[0];frac=count/sum(relations.values());orientation=mode if frac>=.8 else 'ambiguous'
    else:orientation='unknown';frac=0.
    return dict(region=region,status='ok',n=len(frame),candidates=copy.deepcopy(candidates),molecules=members,reads=reads,all_candidates=copy.deepcopy(components),seed_audit=copy.deepcopy(seed_audit),hard_counts=dict(hard),hard_direct_counts=dict(direct),hard_r2_counts=dict(r2),expected_counts=dict(expected),qc=dict(r1_orientation_observed=orientation,r1_orientation_fraction=frac,n_direct_a=sum(r['tier']=='Direct-A' for r in frame),n_direct_b=sum(r['tier']=='Direct-B' for r in frame),n_r2_only=sum(r['tier']=='R2-only' for r in frame),n_reads_reported=len(reads),cell_barcode_filter='none',n_candidates_pre_filter=len(components),n_candidates_retained=n_pas,n_r1_seeds=len(seeds),n_r1_seeds_pruned=sum(s['seed_status']=='model_pruned' for s in seed_audit),n_unassigned=sum(m['assignment']=='unassigned' for m in members)))

def complete_region(audit,options,contribution_sink=None):
    r=audit['result']
    if r['status']!='ok':return copy.deepcopy(r)
    frame=[{k:v for k,v in row.items() if k not in ASSIGNMENT_FIELDS} for row in r['molecules']]
    # P2F numerics permit equivalent floats; native all-present endpoint columns
    # are integer columns. Restore their exact raw representation for serialization.
    raw={row['qname']:row for row in audit['input_rows']}
    for field in ('endpoint_clip','direct_pas'):
        if all(type(row[field]) is int for row in audit['input_rows']):
            for row in frame:row[field]=raw[row['qname']][field]
    return assemble_region(r['region'],frame,audit['read_rows'],r['candidates'],r['all_candidates'],audit['seeds'],r['seed_audit'],audit['posterior']['retained'],options,contribution_sink)

def quantify_region(case):
    """Real frozen-input API: existing region inference followed by native counts."""
    from .region import infer_region
    return complete_region(infer_region(case),case['options'])

def _write_table(path,rows,columns=None,sep='\t'):
    path=Path(path);columns=list(columns) if columns is not None else list(dict.fromkeys(k for row in rows for k in row))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='') as f:
        writer=csv.writer(f,delimiter=sep,lineterminator='\n')
        if not columns:f.write('\n');return
        writer.writerow(columns);floating=set()
        for key in columns:
            values=[r.get(key) for r in rows];known=[v for v in values if present(v)]
            if known and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in known) and (any(isinstance(v,float) for v in values) or any(not present(v) for v in values)):floating.add(key)
        for row in rows:
            writer.writerow(['' if not present(row.get(k)) else str(float(row[k])) if k in floating else row[k] for k in columns])

def write_outputs(results,outdir,assignment_mode):
    """Preserve native logical schemas/order and default missing/rounding rules."""
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    candidates=[];components=[];seeds=[];molecules=[];reads=[];qc=[]
    counts={k:defaultdict(float) for k in ('hard_counts','hard_direct_counts','hard_r2_counts','expected_counts')}
    for r in results:
        qc.append(dict(region=r['region'],status=r['status'],n_molecules=r.get('n',0),**r.get('qc',{})))
        components.extend(r.get('all_candidates',[]));seeds.extend(r.get('seed_audit',[]))
        if r['status']!='ok':continue
        candidates.extend(r['candidates']);molecules.extend(r['molecules']);reads.extend(r['reads'])
        for name,target in counts.items():
            for (cb,k),value in r[name].items():target[(r['region'],cb,k)]+=value
    for name,rows in [('pas_candidates.tsv',candidates),('pas_candidates_all.tsv',components),('r1_seed_audit.tsv',seeds),('molecule_assignments.tsv.gz',molecules),('read_pas_assignments.tsv.gz',reads),('region_qc.tsv',qc)]:_write_table(out/name,rows)
    candidate_map={(c['region'],c['pas_index']):c for c in candidates};matrix_ids={}
    for key,c in candidate_map.items():
        beta=c.get('beta');b='NA' if not present(beta) else str(int(beta)) if float(beta).is_integer() else f'{float(beta):g}'
        matrix_ids[key]=f"{c['chrom']}:{c['position']}:{b}:{c['strand']}"
    hard=counts['hard_counts'];expected=counts['expected_counts'];count_rows=[]
    for region,cb,k in sorted(set(hard)|set(expected)):
        key=(region,cb,k);value=hard.get(key,0.)
        count_rows.append(dict(region=region,cell_barcode=cb,pas_id=candidate_map[(region,k)]['pas_id'],hard_umi_count=value,direct_r1_count=counts['hard_direct_counts'].get(key,0.),r2_assigned_count=counts['hard_r2_counts'].get(key,0.),total_count=value,expected_umi_count=expected.get(key,0.),pas_matrix_id=matrix_ids[(region,k)]))
    _write_table(out/'cell_pas_counts.tsv.gz',count_rows,COUNT_COLUMNS)
    quant={};matrix=defaultdict(float);cells=set()
    for row in count_rows:
        q=quant.setdefault(row['pas_id'],dict(pas_id=row['pas_id'],direct_r1_count=0.,r2_assigned_count=0.,total_count=0.))
        for key in QUANT_COLUMNS[1:]:q[key]+=row[key]
        matrix[(row['pas_matrix_id'],row['cell_barcode'])]+=row['hard_umi_count'];cells.add(row['cell_barcode'])
    _write_table(out/'pas_quantification.tsv',[quant[k] for k in sorted(quant)],QUANT_COLUMNS)
    cells=sorted(cells);matrix_rows=[];integral=all(float(v).is_integer() for v in matrix.values())
    for pas in sorted({k[0] for k in matrix}):matrix_rows.append(dict(pas_id=pas,**{cb:int(matrix.get((pas,cb),0.)) if integral else matrix.get((pas,cb),0.) for cb in cells}))
    _write_table(out/'pasite.csv.gz',matrix_rows,['pas_id']+cells,sep=',')
    summary=dict(regions=len(results),regions_ok=sum(r['status']=='ok' for r in results),pas_candidates=len(candidates),molecules=len(molecules),pas_candidates_pre_filter=len(components),r1_seeds=len(seeds),r1_seeds_pruned=sum(s.get('seed_status')=='model_pruned' for s in seeds),reads_reported=len(reads),assignment_mode=assignment_mode)
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
    return summary
