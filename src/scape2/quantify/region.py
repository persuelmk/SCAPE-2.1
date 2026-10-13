"""Region inference and per-molecule membership; no cell/PAS aggregation."""
from collections import defaultdict,Counter
import copy,math,zlib
import numpy as np
import pysam
from scape2.evidence.geometry import legacy_anchor,strand_matches
from scape2.candidates.direct import direct_pas
from .model_selection import run_selection
from .em import state_from_case,prepare_likelihoods,e_step

def tag(read,key):
    try:return str(read.get_tag(key))
    except KeyError:return None
def nh(read):
    try:return int(read.get_tag('NH'))
    except KeyError:return 1
def primary(records,minimum=0):
    records=[r for r in records if not r.is_unmapped and not r.is_secondary and not r.is_supplementary and r.mapping_quality>=minimum]
    records.sort(key=lambda r:(nh(r)==1,r.mapping_quality,r.query_alignment_length or 0),reverse=True)
    if not records or len(records)>1 and records[0].mapping_quality==records[1].mapping_quality:return None
    return records[0]
def endpoint(read,known_clip):
    pairs=[p for p in read.get_aligned_pairs(matches_only=False) if p[0] is not None and p[1] is not None]
    if not pairs:return None,None
    q,r=(max if read.is_reverse else min)(pairs,key=lambda x:x[0])
    unmapped=read.query_length-1-q if read.is_reverse else q
    residual=max(0,unmapped-max(0,int(known_clip or 0)))
    return int(r+1+residual if read.is_reverse else r+1-residual),int(residual)
def tier(read,clip,opts):
    if read.is_unmapped or read.is_secondary or read.is_supplementary:return 'R2-only'
    policy=opts['r1_structure_policy'];structure=tag(read,opts['r1_structure_tag'])
    usable=policy=='ignore' or (policy=='if-present' if structure is None else structure.strip().lower() in ('exact','fuzzy','trusted_fuzzy'))
    if nh(read)!=1 or not usable:return 'R2-only'
    if read.mapping_quality>=opts['direct_mapq'] and clip<=opts['direct_softclip']:return 'Direct-A'
    if read.mapping_quality>=opts['semi_mapq'] and clip<=opts['semi_softclip']:return 'Direct-B'
    return 'R2-only'
def representative(items):
    def key(x):
        clip=x['endpoint_clip'];clip=math.inf if clip is None or isinstance(clip,float) and math.isnan(clip) else float(clip)
        return (-{'Direct-A':2,'Direct-B':1,'R2-only':0}.get(x['tier'],-1),-int(x['r1_mapq'] or -1),clip,-int(x['r2_mapq'] or -1),-int(x['r2_len'] or 0),str(x['qname'] or ''))
    return min(items,key=key)
def read_evidence(case):
    region=case['region'];chrom,left,right,strand=region.split('\t')[:4];left=int(left);right=int(right);o=case['options'];cbtag,umitag=[s.strip() for s in case['tag'].split(',')]
    allowed_barcodes=case.get('allowed_barcodes')
    if allowed_barcodes is not None and not isinstance(allowed_barcodes,(set,frozenset)):
        allowed_barcodes=frozenset(allowed_barcodes)
    r2names=defaultdict(list);r1names=defaultdict(list)
    with pysam.AlignmentFile(case['bam_r2'],'rb',check_sq=False) as bam:
        for read in bam.fetch(chrom,max(0,left-1),right+1):
            if o.get('paired_bam_mode') and not read.is_read2:continue
            if read.mapping_quality<o['r2_min_mapq'] or not strand_matches(read.flag,strand):continue
            r2names[read.query_name].append(read)
    ext=o['r1_fetch_extension']
    with pysam.AlignmentFile(case['bam_r1'],'rb',check_sq=False) as bam:
        for read in bam.fetch(chrom,max(0,left-1-ext),right+ext):
            if o.get('paired_bam_mode') and not read.is_read1:continue
            if not read.is_unmapped:r1names[read.query_name].append(read)
    molecules=defaultdict(list);reads=[];relations=[]
    for name,records in r2names.items():
        r2=primary(records,o['r2_min_mapq'])
        if r2 is None:continue
        cb,umi=tag(r2,cbtag),tag(r2,umitag)
        if not cb or not umi:continue
        if allowed_barcodes is not None and cb not in allowed_barcodes:continue
        gene=tag(r2,o['gene_tag']) or region;key=(cb,gene,umi)
        anchor=legacy_anchor(r2.reference_start,r2.reference_end,r2.cigartuples,r2.is_reverse)
        if strand=='+' and anchor>right or strand=='-' and anchor<left:continue
        start=anchor-left+1 if strand=='+' else right-anchor+1
        direct=clip=None;level='R2-only';relation='missing';structure='missing';r1=primary(r1names.get(name,[]))
        if r1 is not None:
            structure=tag(r1,o['r1_structure_tag']) or 'missing';relation='same' if ('-' if r1.is_reverse else '+')==strand else 'opposite';relations.append(relation)
            direct,clip=endpoint(r1,o.get('r1_known_terminal_clip',0));level=tier(r1,clip,o)
            if o['r1_orientation']!='ignore' and relation!=o['r1_orientation']:level='R2-only'
        try:tail=len(r2.get_tag('DT'))
        except KeyError:tail=math.nan
        relative=math.nan if direct is None else direct-left+1 if strand=='+' else right-direct+1
        if direct is not None and (relative<1 or relative>right-left+1+ext):level='R2-only';relative=math.nan
        item=dict(qname=name,cb=cb,umi=umi,gene=gene,r2_start=start,r2_len=r2.query_alignment_length or 0,r2_mapq=int(r2.mapping_quality),r1_mapq=int(r1.mapping_quality) if r1 is not None else -1,endpoint_clip=clip,r1_len=0,polya_len=tail,pa_site=relative if level!='R2-only' else math.nan,direct_pas=direct,tier=level,relation=relation,r1_structure_status=structure)
        reads.append(dict(item,region=region,molecule_key=key));molecules[key].append(item)
    rows=[representative(molecules[key]) for key in sorted(molecules)]
    return rows,reads,relations

def _present(value):return value is not None and not (isinstance(value,float) and math.isnan(value))
def nearby(direct,seed,position,o):
    if seed is not None and 'interval_start' in seed and 'interval_end' in seed:
        return [r for r in direct if seed['interval_start']<=int(r['direct_pas'])<=seed['interval_end']]
    return [r for r in direct if abs(int(r['direct_pas'])-position)<=o['cluster_radius']]
def match_seeds(positions,seeds,distance):
    pairs=sorted((abs(float(p)-float(s['position'])),i,j) for i,p in enumerate(positions) for j,s in enumerate(seeds) if abs(float(p)-float(s['position']))<=distance)
    result={};used=set()
    for _,i,j in pairs:
        if i not in result and j not in used:result[i]=j;used.add(j)
    return result

def infer_region(case):
    """From frozen indexed read evidence to region model/posterior/membership."""
    region=case['region'];chrom,left,right,strand=region.split('\t')[:4];left=int(left);right=int(right);o=case['options']
    if not o.get('cache_fingerprint'):raise ValueError('cache_fingerprint is required; refusing unsafe cache reuse')
    rows,reads,relations=read_evidence(case)
    result=dict(region=region,status=None,n=len(rows));audit=dict(input_rows=copy.deepcopy(rows),read_rows=copy.deepcopy(reads),seeds=[],model_input=None,selection=None,fitted_model=None,posterior=None,result=result)
    if len(reads)<o['min_reads']:
        result.update(status='too_few_reads',qc=dict(n_reads_reported=len(reads),cell_barcode_filter='none'));return audit
    direct=[r for r in rows if r['tier']=='Direct-A' and _present(r['direct_pas'])]
    seeds=direct_pas([int(r['direct_pas']) for r in direct],o['cluster_radius'],o['merge_distance'],o['min_direct_support'],o['endpoint_bandwidth'],method=o['r1_caller'])
    for s in seeds:
        near=nearby(direct,s,s['position'],o);s['endpoint_umi_support']=len(near);s['endpoint_cell_support']=len({r['cb'] for r in near})
        s['endpoint_fraction']=len(near)/max(1,len(direct));s['endpoint_cell_fraction']=s['endpoint_cell_support']/max(1,len({r['cb'] for r in direct}))
        s['protected']=s['endpoint_fraction']>=o.get('min_protected_r1_fraction',.2) and s['endpoint_cell_fraction']>=o.get('min_protected_r1_cell_fraction',.2)
    audit['seeds']=copy.deepcopy(seeds);protected=[s for s in seeds if s['protected']]
    alpha=[(s['position']-left+1 if strand=='+' else right-s['position']+1)+150 for s in seeds]
    frame=copy.deepcopy(rows)
    # Preserve pandas' numeric missing-value representation at the native frame boundary.
    for field in ('direct_pas','endpoint_clip'):
        if any(r[field] is not None for r in frame):
            for r in frame:r[field]=math.nan if r[field] is None else float(r[field])
    for r in frame:r['r2_start_model']=r['r2_start']+150;r['pa_site_model']=r['pa_site']+150
    length=max(int(max(r['r2_start_model'] for r in frame)+max(r['r2_len'] for r in frame)+50),right-left+1+300,1000);minimum=min(r['r2_len'] for r in frame)
    fixed=dict(molecules=[dict(id=r['qname'],x=int(r['r2_start_model']),l=int(r['r2_len']),r=float(r['r1_len']),s=float(r['polya_len']),known_pa=float(r['pa_site_model'])) for r in frame],theta=list(range(int(minimum),length,9)),beta_grid=list(np.arange(9.,73.,9.)),theta_step=9.,min_theta=float(min(minimum,min(alpha)) if alpha else minimum),L=float(length),mu=300.,sigma=50.,max_tail=150.,tail_lengths=case['la_dis_arr'],tail_weights=case['pmf_la_dis_arr'],protected=None)
    cfg=dict(min_k=o['n_min_apa'],max_k=max(o['n_max_apa'],len(alpha)),min_pa_gap=o['min_pa_gap'],min_ws=.01,seeds=alpha,seed_weights=[s['endpoint_umi_support'] for s in seeds],competitive=True)
    modelcase=dict(fixed=fixed,selection=cfg,numpy_seed=zlib.crc32(region.encode())&0xffffffff,python_seed=case['python_seed'],region=dict(start0=left-1,end0=right,strand=strand))
    audit['model_input']=copy.deepcopy(modelcase)
    dummy=dict(fixed=fixed,state=dict(ids=['PAS_0'],alpha=[fixed['theta'][0]],beta=[9.],weights=[.9,.1]));prepared=prepare_likelihoods(state_from_case(dummy).fixed)
    try:selection=run_selection(modelcase,prepared=prepared)
    except (TypeError,ValueError):
        result['status']='inference_failed';return audit
    audit['selection']=selection
    returned=selection['returned']
    if returned is None:result['status']='inference_failed';return audit
    state=returned['state'];audit['fitted_model']=dict(alpha=state['alpha'],beta=state['beta'],weights=state['weights'],lb=returned['lb'],bic=returned['bic'])
    z_full=np.asarray(e_step(state_from_case(dict(fixed=fixed,state=state)),prepared).responsibility)
    relative=np.asarray(state['alpha'])-150;genomic_full=left+relative-1 if strand=='+' else right-relative+1
    valid=np.where((genomic_full>=left)&(genomic_full<=right))[0];invalid=np.where((genomic_full<left)|(genomic_full>right))[0]
    if not len(valid):result['status']='no_valid_pas';return audit
    genomic=genomic_full[valid];betas=np.asarray(state['beta'])[valid];weights=np.asarray(state['weights'])[valid]
    noise=z_full[:,-1].copy()
    if len(invalid):noise+=z_full[:,invalid].sum(axis=1)
    z=np.column_stack([z_full[:,valid],noise]);matched=match_seeds(genomic,seeds,o['merge_distance'])
    sources=['R1-direct' if i in matched else 'R2-novel' for i in range(len(genomic))]
    reported=np.asarray([seeds[matched[i]]['position'] if i in matched else p for i,p in enumerate(genomic)],dtype=float)
    best=z.argmax(axis=1);components=[];keep=[]
    for k,(modelpos,pos,beta,source,weight) in enumerate(zip(genomic,reported,betas,sources,weights)):
        seed=seeds[matched[k]] if k in matched else None;near=nearby(direct,seed,pos,o);umis=len(near);cells=len({r['cb'] for r in near});assigned=int(np.sum(best==k));expected=float(np.sum(z[:,k]));reasons=[]
        if float(weight)<o['min_final_weight']:reasons.append('low_weight')
        if assigned<o['min_final_umis']:reasons.append('low_assigned_umis')
        if source=='R1-direct':
            if umis<o['min_final_endpoint_umis']:reasons.append('low_endpoint_umis')
            if cells<o['min_final_endpoint_cells']:reasons.append('low_endpoint_cells')
        if seed is not None and seed.get('protected'):reasons=[]
        if not reasons:keep.append(k)
        components.append(dict(region=region,original_pas_index=k,pas_id=f'{chrom}:{round(pos)}:{strand}',chrom=chrom,position=round(pos),model_position=round(modelpos),strand=strand,source=source,beta=float(beta),mixture_weight=float(weight),endpoint_umi_support=umis,endpoint_cell_support=cells,preliminary_hard_umi_support=assigned,preliminary_expected_umi_support=expected,protected_r1=bool(seed and seed.get('protected')),assignment_interval_start=seed.get('interval_start') if seed else None,assignment_interval_end=seed.get('interval_end') if seed else None,candidate_status='retained' if not reasons else 'pruned',prune_reason=','.join(reasons)))
    if not keep:
        result.update(status='no_confident_pas',all_candidates=components,seed_audit=[dict(region=region,**s,seed_status='model_pruned') for s in seeds],qc=dict(n_reads_reported=len(reads),n_candidates_pre_filter=len(components),n_candidates_retained=0));return audit
    z=z[:,keep+[len(genomic)]];z=z/z.sum(axis=1,keepdims=True);genomic=genomic[keep];reported=reported[keep];betas=betas[keep];weights=weights[keep];sources=[sources[i] for i in keep];retained=[components[i] for i in keep]
    missing=[s for s in protected if not len(reported) or np.min(abs(reported-s['position']))>o['merge_distance']]
    for s in missing:
        z=np.column_stack([z[:,:-1],np.zeros(len(z)),z[:,-1]]);genomic=np.append(genomic,float(s['position']));reported=np.append(reported,float(s['position']));betas=np.append(betas,np.nan);weights=np.append(weights,0.);sources.append('R1-direct')
        a=dict(region=region,original_pas_index=None,pas_id=f"{chrom}:{s['position']}:{strand}",chrom=chrom,position=s['position'],model_position=s['position'],strand=strand,source='R1-direct',beta=np.nan,mixture_weight=0.,endpoint_umi_support=s['endpoint_umi_support'],endpoint_cell_support=s['endpoint_cell_support'],preliminary_hard_umi_support=0,preliminary_expected_umi_support=0.,protected_r1=True,assignment_interval_start=s['interval_start'],assignment_interval_end=s['interval_end'],candidate_status='retained',prune_reason='protected_r1_rescue');retained.append(a);components.append(a)
    membership=[];noise_index=len(genomic)
    for i,row in enumerate(frame):
        best=int(np.argmax(z[i]));prob=float(z[i,best]);reason='';ordered=np.sort(z[i])[::-1];margin=float(ordered[0]-ordered[1]) if len(ordered)>1 else float(ordered[0])
        if row['tier']!='R2-only' and _present(row['direct_pas']):
            eligible=[k for k,(source,a) in enumerate(zip(sources,retained)) if source=='R1-direct' and a.get('assignment_interval_start') is not None and a.get('assignment_interval_end') is not None and float(a['assignment_interval_start'])<=float(row['direct_pas'])<=float(a['assignment_interval_end'])]
            rule='r1_cdf_valley_interval';assigned=bool(eligible)
            if eligible:best=min(eligible,key=lambda k:abs(float(row['direct_pas'])-reported[k]));prob=float(z[i,best])
            else:reason='no_cdf_pas_interval'
        elif row['tier']=='R2-only' and o['assignment_mode']=='high-confidence':
            assigned=best<noise_index and prob>=o['min_posterior'] and margin>=o.get('min_posterior_margin',.2);rule='r2_posterior_and_margin'
            if best<noise_index and not assigned:reason='ambiguous_r2_assignment'
        else:assigned=best<noise_index;rule='r1_interval' if row['tier']!='R2-only' else 'argmax_pas'
        membership.append(dict(row,pas_id=f'{chrom}:{round(reported[best])}:{strand}' if assigned else 'UNASSIGNED',posterior=prob,noise_posterior=float(z[i,noise_index]),assignment='assigned' if assigned else 'unassigned',assignment_rule=rule,unassigned_reason=reason))
    candidates=[]
    for k,(pos,beta,source,weight,a) in enumerate(zip(reported,betas,sources,weights,retained)):
        candidates.append(dict(region=region,pas_index=k,pas_id=f'{chrom}:{round(pos)}:{strand}',chrom=chrom,position=round(pos),strand=strand,model_position=a['model_position'],source=source,beta=float(beta),mixture_weight=float(weight),direct_support=a['endpoint_umi_support'],**{key:a[key] for key in ('endpoint_umi_support','endpoint_cell_support','preliminary_hard_umi_support','preliminary_expected_umi_support')}))
    seed_audit=[]
    for s in seeds:
        distance=float(np.min(abs(reported-s['position']))) if len(reported) else math.nan
        seed_audit.append(dict(region=region,**s,nearest_retained_pas_distance=distance,seed_status='retained_or_merged' if distance<=o['merge_distance'] else 'model_pruned'))
    result.update(status='ok',candidates=candidates,molecules=membership,all_candidates=components,seed_audit=seed_audit)
    audit['posterior']=dict(full=z_full.tolist(),retained=z.tolist(),row_keys=[[r['cb'],r['gene'],r['umi'],r['qname']] for r in frame],column_ids=[c['pas_id'] for c in candidates]+['NOISE'],valid_component_indices=valid.tolist(),invalid_component_indices=invalid.tolist(),kept_component_indices=keep,noise_index=noise_index)
    return audit
