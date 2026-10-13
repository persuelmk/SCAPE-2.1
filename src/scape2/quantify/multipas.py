"""Provided-state multi-PAS orchestration for the shared fixed-K EM driver."""
import random
from .em import state_from_case, e_step
from .single_pas import run_fixed_k, _parameters, _genomic


def component_schedule(k: int, nround: int, seed: int) -> tuple[int, ...]:
    """Reproduce gen_k_arr including all random draws, without global RNG state."""
    if type(k) is not int or k<1 or type(nround) is not int or nround<1:
        raise ValueError('positive K and nround required')
    if k==1:return (0,)*nround
    rng=random.Random(seed);pool=list(range(k));last=None;result=[]
    for _ in range(nround):
        rng.shuffle(pool)
        if pool[0]==last:
            other=rng.randrange(1,len(pool));pool[0],pool[other]=pool[other],pool[0]
        result.extend(pool);last=pool[-1]
    return tuple(result)


def initialize_fixed_k(case, prepared):
    """Accept a fully frozen state; no discovery, sorting, or new initializer."""
    if case['initialization']['mode']!='provided' or case['initialization']['provided']!=case['state']:
        raise ValueError('explicit frozen provided initialization required')
    state=state_from_case(case)
    if len(state.alpha)!=case['K']:raise ValueError('BIOLOGICAL_K disagrees with provided state')
    initial=e_step(state,prepared)
    return state,{'state':_parameters(state),'initial_log_joint':initial.log_joint,
                  'genomic_position':[_genomic(a,case['region']) for a in state.alpha]}


def run_multipas(state, prepared, *, nround, region, schedule_seed):
    if len(state.alpha)<2:raise ValueError('multi-PAS requires K>=2')
    schedule=component_schedule(len(state.alpha),nround,schedule_seed)
    result=run_fixed_k(state,prepared,nround=nround,region=region,update_schedule=schedule)
    result['schedule']=list(schedule)
    return result
