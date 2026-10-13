"""Legacy BIC, initialization and selection orchestration; no alternative objective.

Source contract: docs/P2E_BIC_K_SELECTION_CONTRACT.md. NumPy RNG and Python
RNG are local and persist across descending K and the five trials per K.
"""
import copy
import math
import random
from dataclasses import replace
import numpy as np
from .em import state_from_case, prepare_likelihoods, e_step
from .single_pas import run_fixed_k


def legacy_bic(log_joint, responsibility):
    z=np.asarray(responsibility,dtype=float);joint=np.asarray(log_joint,dtype=float)
    with np.errstate(all='ignore'):
        terms=z*joint;terms[z==0]=0
        q=float(np.sum(terms));n=z.shape[0];k=z.shape[1]-1;p=3*k+1
        score=float(-2*q+p*np.log(n))
    return {'q':q,'N':n,'K':k,'p':p,'bic':score}


def candidate_range(n_min,n_max,n_seeds=0,competitive=False):
    maximum=max(n_max,n_seeds)
    minimum=max(n_min,n_seeds) if not competitive else n_min
    return list(range(maximum,minimum-1,-1))


def choose_trials(trials,gap):
    scores=np.full(len(trials),np.inf)
    for i,t in enumerate(trials):
        if t.get('state') is not None and not np.any(np.diff(t['state']['alpha'])<gap):
            scores[i]=t['bic']
    index=int(np.argmin(scores))
    return index,scores.tolist()


def choose_candidates(candidates):
    valid=[(i,c) for i,c in enumerate(candidates) if c.get('state') is not None and np.isfinite(c['bic'])]
    if not valid:return None
    # min preserves the first element on exact equality, matching APA.inference.
    return min(valid,key=lambda item:item[1]['bic'])[0]


def schedule(k,nround,rng):
    if k<=1:return [k-1]*nround
    pool=list(range(k));last=None;out=[]
    for _ in range(nround):
        rng.shuffle(pool)
        if pool[0]==last:
            j=rng.randrange(1,k);pool[0],pool[j]=pool[j],pool[0]
        out.extend(pool);last=pool[-1]
    return out


def data_windows(fixed):
    """Coverage-window preprocessing for the existing legacy initializer."""
    length=int(fixed['L']);coverage=np.zeros(length)
    for m in fixed['molecules']:
        idx=int(m['x'])-1+np.arange(int(m['l']))
        coverage[idx]+=1
    sigma=max(1.,5.*fixed['theta_step']);radius=int(4*sigma+.5)
    offsets=np.arange(-radius,radius+1);weights=np.exp(-.5/sigma**2*offsets**2);weights/=weights.sum()
    padded=np.pad(coverage,(radius,radius),mode='edge')
    smooth=weights[radius]*coverage
    for j in range(radius):
        smooth+=(padded[j:j+length]+padded[2*radius-j:2*radius-j+length])*weights[j]
    signs=np.sign(np.diff(np.r_[-1.,smooth]))
    if signs[0]!=1:raise ValueError('First sign value not 1, split_data func')
    def runs(x):
        start=np.r_[0,np.flatnonzero(~np.isclose(x[1:],x[:-1],equal_nan=True))+1]
        return start,np.diff(np.r_[start,len(x)]),x[start]
    starts,lens,values=runs(signs);ends=np.cumsum(lens)
    for i in np.flatnonzero(values==0):
        lo,hi=int(starts[i]),int(ends[i])
        if i==len(values)-1 or hi-lo<=1:signs[lo:hi]=values[i-1]
        else:
            mid=int(np.round((lo+hi+1)/2));signs[lo:mid]=values[i-1];signs[mid:hi]=values[i+1]
    _,lens,_=runs(signs);changes=np.cumsum(lens)
    modes=changes[::2];borders=changes[1::2][:len(modes)-1]
    starts=np.r_[1,borders+1];ends=np.r_[borders,length]
    weights=np.array([sum(coverage[a:b]) for a,b in zip(starts,ends)]);weights/=weights.sum()
    return dict(st_arr=starts,en_arr=ends,mode_arr=modes,ws_arr=weights)


def initialize_components(k,fixed,windows,config,rng):
    grid=np.asarray(fixed['theta'],dtype=float);betas=np.asarray(fixed['beta_grid'],dtype=float)
    seeds=np.asarray(config.get('seeds',[]),dtype=float);gap=config['min_pa_gap']
    half=betas[:max(1,int(len(betas)/2+1))]
    if len(seeds):
        snapped=np.array([grid[np.argmin(abs(grid-v))] for v in sorted(seeds)])
        if k<len(snapped):
            if not config['competitive']:raise ValueError('n_apa smaller than number of R1 PAS seeds')
            weights=np.asarray(config.get('seed_weights',[1.]*len(snapped)),dtype=float)
            if not np.isfinite(weights).all() or weights.sum()<=0:weights=np.ones(len(snapped))
            chosen=rng.choice(len(snapped),size=k,replace=False,p=weights/weights.sum())
            selected=np.sort(snapped[chosen])
        else:selected=np.sort(snapped)
        need=k-len(selected);extras=np.array([])
        valid=grid[grid>=fixed['min_theta']]
        valid=valid[np.array([np.all(abs(selected-v)>=gap) for v in valid])]
        for _ in range(500):
            if need==0:break
            if len(valid)<need:raise ValueError('not enough space for additional PAS components')
            proposed=np.sort(rng.choice(valid,size=need,replace=False))
            if np.all(np.diff(np.sort(np.r_[selected,proposed]))>=gap):extras=proposed;break
        if need and len(extras)!=need:raise ValueError('failed to initialize R2-novel PAS components')
        return np.sort(np.r_[selected,extras]),rng.choice(half,size=k,replace=True)

    def sample(indices,mode):
        result=[]
        for i in indices:
            lo=windows['mode_arr'][i] if mode=='right' else windows['st_arr'][i]
            hi=windows['mode_arr'][i] if mode=='left' else windows['en_arr'][i]
            possible=grid[(grid>=lo)&(grid<hi)]
            result.append(float(rng.choice(possible,size=1,replace=False)[0]) if len(possible) else -1.)
        return np.sort(result)
    n=len(windows['ws_arr']);big=np.flatnonzero(windows['ws_arr']>.1)
    def second(indices):
        indices=np.asarray(indices)
        a=sample([n-1],'full') if np.any(indices==n) else np.array([])
        b=sample(indices[indices!=n],'left') if np.any(indices!=n) else np.array([])
        return np.r_[a,b]
    for attempt in range(200):
        if n>=k:
            inds=np.sort(rng.choice(n,size=k,p=windows['ws_arr'],replace=False));alpha=sample(inds,'right')
        elif n>=k-len(big):
            a=sample(np.arange(n),'right')
            if len(big)==1:inds=big+1
            else:
                w=windows['ws_arr'][big];inds=rng.choice(big+1,size=k-n,p=w/w.sum(),replace=False)
            alpha=np.sort(np.r_[a,second(inds)])
        else:
            a=sample(np.arange(n),'right');b=second(big+1)
            inds=rng.choice(n,size=k-n-len(big),p=windows['ws_arr'])
            alpha=np.sort(np.r_[a,b,sample(inds,'full')])
        if np.all(np.diff(alpha)>=gap) and np.all(alpha>0):break
    if attempt==199:raise ValueError('Failed to generate valid theta after 200 trial.')
    return alpha,rng.choice(half,size=k)


def run_selection(case, *, prepared=None):
    fixed=copy.deepcopy(case['fixed']);config=case['selection'];rng=np.random.RandomState(case['numpy_seed']);py=random.Random(case['python_seed'])
    seeds=config.get('seeds',[]);grid=np.asarray(fixed['theta'])
    snapped=[float(grid[np.argmin(abs(grid-v))]) for v in sorted(seeds)]
    fixed['protected']=snapped or None
    windows=data_windows(fixed)
    dummy={'fixed':fixed,'state':{'ids':['PAS_0'],'alpha':[fixed['theta'][0]],'beta':[fixed['beta_grid'][0]],'weights':[.9,.1]}}
    if prepared is None:prepared=prepare_likelihoods(state_from_case(dummy).fixed)
    candidates=candidate_range(config['min_k'],config['max_k'],len(seeds),config['competitive'])
    trials=[];per_k=[]
    def fit(alpha,beta,weights,coordinates_fixed=False):
        k=len(alpha);state_dict={'ids':[f'PAS_{i}' for i in range(k)],'alpha':list(map(float,alpha)),'beta':list(map(float,beta)),'weights':list(map(float,weights))}
        trial={'initial':state_dict,'state':None,'bic':None,'q':None,'p':3*k+1,'N':len(fixed['molecules']),'iterations':0,'converged':False,'termination':'NOT_STARTED','schedule':[]}
        update=schedule(k,200 if coordinates_fixed else 50,py);trial['schedule']=update
        state=state_from_case({'fixed':fixed,'state':state_dict})
        result=run_fixed_k(state,prepared,nround=200 if coordinates_fixed else 50,region=case['region'],update_schedule=tuple(update),fixed_coordinates=coordinates_fixed)
        final=result['final'];last=result['trace'][-1]
        trial.update(iterations=final['iterations'],converged=final['converged'],termination=final['termination_reason'])
        if not final['legacy_returned']:
            trial['termination']='EM_FAILURE';return trial,None
        finalstate=final['state'];trial.update(state=finalstate,**{key:value for key,value in legacy_bic(last['log_joint'],last['responsibility']).items() if key!='K'},lb=[t['pre_update_elbo'] for t in result['trace'] if t['pre_update_elbo'] is not None])
        if not coordinates_fixed and seeds:
            initially_protected=np.isin(alpha,snapped)
            if any(any(abs(a-s)<config['min_pa_gap'] for s in seeds) for a,protected in zip(finalstate['alpha'],initially_protected) if not protected):
                trial['state']=None;trial['termination']='POST_FIT_REJECTED'
        return trial,last
    for order,k in enumerate(candidates):
        current=[]
        for trial_index in range(5):
            t={'initial':None,'state':None,'bic':None,'q':None,'p':3*k+1,'N':len(fixed['molecules']),'iterations':0,'converged':False,'termination':'INITIALIZATION_FAILURE','schedule':[]}
            try:
                weights=rng.uniform(size=k+1)+1;weights[k]-=1;weights/=weights.sum()
                alpha,beta=initialize_components(k,fixed,windows,config,rng)
                t,_=fit(alpha,beta,weights)
            except (ValueError,IndexError,ZeroDivisionError):pass
            t.update(K=k,order=order,trial=trial_index);current.append(t);trials.append(t)
        chosen,scores=choose_trials(current,config['min_pa_gap'])
        winner=copy.deepcopy(current[chosen]);winner['trial_scores']=scores;winner['chosen_trial']=chosen;per_k.append(winner)
    selected=choose_candidates(per_k)
    output={'candidate_k':candidates,'windows':{k:v.tolist() for k,v in windows.items()},'trials':trials,'per_k':per_k,'selected_k':None,'returned_k':None,'selected_candidate_index':selected,'fallback':'NO_VALID_CANDIDATE' if selected is None else 'NONE','returned':None,'pruning':None}
    if selected is None:return output
    winner=copy.deepcopy(per_k[selected]);output['selected_k']=winner['K']
    state=winner['state'];k=winner['K'];protected=snapped if not config['competitive'] else []
    keep=np.asarray(state['weights'][:-1])>=config['min_ws']
    if protected:keep|=np.isin(state['alpha'],protected)
    if keep.sum()==k or keep.sum()==0:
        e=e_step(state_from_case({'fixed':fixed,'state':state}),prepared)
        winner['labels']=np.asarray(e.log_joint).argmax(axis=1).tolist();output['pruning']='NONE' if keep.sum()==k else 'ALL_BELOW_THRESHOLD_KEEP_ALL'
    else:
        alpha=np.asarray(state['alpha'])[keep];beta=np.asarray(state['beta'])[keep]
        weights=rng.uniform(size=len(alpha)+1)+1;weights[-1]-=1;weights/=weights.sum()
        refit,last=fit(alpha,beta,weights,coordinates_fixed=True)
        if refit['state'] is None:output['fallback']='LEGACY_RUNTIME_LIMITATION';output['pruning']='REFIT_FAILURE';return output
        winner=refit;winner['labels']=np.asarray(last['log_joint']).argmax(axis=1).tolist();output['pruning']='WEIGHT_ONLY_REFIT'
    output['returned']=winner;output['returned_k']=len(winner['state']['alpha'])
    return output
