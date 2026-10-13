"""Shared legacy fixed-K iteration driver, with the frozen single-PAS entrypoint."""
import copy
import math
from dataclasses import asdict, replace
from .em import EMState, PreparedLikelihoods, state_from_case, e_step, m_step, update_weights, _immutable_shape_inputs
from scape2.evidence.geometry import genomic_coordinate

def _parameters(state: EMState) -> dict:
    return {key:list(value) for key,value in state.parameters().items()}

def _genomic(alpha: float, region: dict) -> int:
    return genomic_coordinate(int(alpha),int(region['start0'])+1,int(region['end0']),region['strand'])

def initialize_single_pas(case: dict, prepared: PreparedLikelihoods) -> tuple[EMState, dict]:
    """Replay a bounded legacy K=1 initializer using a local RandomState."""
    import numpy as np
    init=case['initialization'];fixed=case['fixed'];mode=init['mode']
    if len(case['state']['alpha'])!=1:
        raise ValueError('single PAS only')
    if mode=='provided':parameters=copy.deepcopy(init['provided'])
    elif mode in ('seeded','window'):
        rng=np.random.RandomState(init['seed'])
        weights=rng.uniform(size=2)+1.;weights[1]-=1.;weights=weights/weights.sum()
        grid=np.asarray(fixed['theta'],dtype=float);betas=np.asarray(fixed['beta_grid'],dtype=float)
        if mode=='seeded':alpha=float(grid[np.argmin(abs(grid-init['seed_position']))])
        else:
            # Preserve both source draws, including the single-window choice.
            rng.choice(a=1,size=1,p=np.array([1.]),replace=False)
            eligible=grid[(grid>=init['window_mode']) & (grid<init['window_end'])]
            if not len(eligible) or min(eligible)<=0:
                raise ValueError('the frozen one-window initializer needs positive eligible coordinates')
            alpha=float(rng.choice(a=eligible,size=1,replace=False)[0])
        beta=float(rng.choice(betas[:max(1,int(len(betas)/2+1))],size=1,replace=True)[0])
        parameters={'ids':['PAS_0'],'alpha':[alpha],'beta':[beta],'weights':weights.tolist()}
    else:raise ValueError('unsupported single-PAS initialization mode')
    configured=copy.deepcopy(case);configured['state']=parameters
    configured['fixed']['protected']=[parameters['alpha'][0]] if mode=='seeded' else None
    state=state_from_case(configured)
    initial=e_step(state,prepared)
    result={'state':_parameters(state),'fixed':configured['fixed'],'seed':init['seed'],'mode':mode,'initial_log_joint':initial.log_joint,'genomic_position':_genomic(state.alpha[0],case['region'])}
    return state,result

def _diagnostics(prepared: PreparedLikelihoods, state: EMState) -> list[str]:
    flat=[v for row in prepared.raw for v in row];result=[]
    if any(math.isnan(v) for v in flat):result.append('NAN')
    if any(math.isinf(v) for v in flat):result.append('INF')
    if all(v==0 for v in flat):result.append('ZERO_SUPPORT')
    if all(len(set(row))==1 for row in prepared.raw):result.append('DEGENERATE_STATE')
    if state.weights[-1]>sum(state.weights[:-1]):result.append('NOISE_DOMINATED')
    return result

def run_fixed_k(state: EMState, prepared: PreparedLikelihoods, *, nround: int,
                region: dict, update_schedule: tuple[int, ...] | None = None,
                fixed_coordinates: bool = False) -> dict:
    """Shared loop. Returned parameters are post-M; score uses cached pre-M Z."""
    if not isinstance(nround,int) or nround<=0:raise ValueError('positive integer nround required')
    k=len(state.alpha)
    schedule=(0,)*nround if k==1 and update_schedule is None else update_schedule
    if schedule is None or len(schedule)<nround or any(type(i) is not int or not 0<=i<k for i in schedule):
        raise ValueError('fixed-K loop requires a valid component update schedule')
    def positions(s):
        values=[_genomic(a,region) for a in s.alpha]
        return values[0] if k==1 else values
    initial=e_step(state,prepared);cached=initial.log_joint
    validated_fixed=state.fixed if _immutable_shape_inputs(state,prepared) else None
    previous=-math.inf;trace=[];error=None
    for index in range(nround):
        row={'iteration':index+1,'pre_state':_parameters(state),'previous_lb':previous,'post_state':None,'pre_update_elbo':None,'absolute_delta':None,'threshold':None,'converged':False,'stop':False,'termination_reason':None}
        selected=schedule[index]
        if k>1:row['update_component']=selected
        shape_validated=validated_fixed is not None and state.fixed is validated_fixed
        e=e_step(state,prepared,cached_log_joint=cached,update_index=selected,
                 _shape_validated=shape_validated);cached=e.log_joint
        row.update(log_joint=e.log_joint,responsibility=e.responsibility,expected_complete_log=e.bookkeeping.expected_complete_log)
        try:
            if fixed_coordinates:
                state=replace(state,weights=update_weights(e.responsibility))
            else:
                updated=m_step(state,prepared,e.responsibility,selected,
                               _shape_validated=shape_validated)
                state=updated.state
            row.update(post_state=_parameters(state),genomic_position=positions(state))
            if e.bookkeeping.elbo_error:raise ValueError('lower bounder is na.')
        except (ValueError,IndexError,ZeroDivisionError) as exc:
            error=type(exc).__name__
            row.update(stop=True,termination_reason='NAN' if 'lower bounder' in str(exc) else 'DEGENERATE_STATE',exception=error)
            trace.append(row);break
        value=e.bookkeeping.pre_update_elbo
        delta=abs(value-previous);threshold=abs(1e-6*previous)
        negative=value==-math.inf;converged=not negative and delta<threshold;last=index+1==nround
        row.update(pre_update_elbo=value,absolute_delta=delta,threshold=threshold,converged=converged,stop=negative or converged or last,termination_reason='NEG_INF' if negative else 'CONVERGED' if converged else 'MAX_ITER_REACHED' if last else None)
        trace.append(row)
        if negative:previous=-math.inf;break
        if converged:break
        previous=value
    last=trace[-1]
    final={'state':_parameters(state),'iterations':len(trace),'scored_iterations':sum(x['pre_update_elbo'] is not None for x in trace),'converged':last['converged'],'termination_reason':last['termination_reason'],'last_pre_update_elbo':last['pre_update_elbo'],'legacy_lb_variable':previous,'legacy_returned':error is None,'exception':error,'genomic_position':positions(state),'diagnostic_flags':_diagnostics(prepared,state),'last_responsibility':last.get('responsibility')}
    return {'trace':trace,'final':final}


def run_single_pas(state: EMState, prepared: PreparedLikelihoods, *, nround: int,
                   region: dict) -> dict:
    """Preserve the existing K=1 API and result schema."""
    if len(state.alpha)!=1:raise ValueError('single PAS only')
    return run_fixed_k(state,prepared,nround=nround,region=region)
