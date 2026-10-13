"""N x (K+1) E-step boundary; never copies the likelihood cube."""
import math
import numpy as np

def accelerated(state,prepared,*,cached_log_joint=None,update_index=None,
                _shape_validated=False,implementation='C1'):
    from scape2.quantify import em
    if not _shape_validated:em._shape(state,prepared)
    f=state.fixed;k=len(state.alpha);n=len(f.molecules)
    lookup=[(f.beta_grid.index(b),f.theta.index(a)) for a,b in zip(state.alpha,state.beta)]
    selected=np.asarray([[prepared.cube[bi][i][ti] for bi,ti in lookup]+[prepared.noise_log]
                         for i in range(n)],dtype=np.float64,order='C')
    selected_index=0
    has_cache=cached_log_joint is not None
    if has_cache:
        selected_index=0 if update_index is None and k==1 else update_index
        if (type(selected_index) is not int or not 0<=selected_index<k or
            len(cached_log_joint)!=n or any(len(row)!=k+1 for row in cached_log_joint)):
            raise ValueError('cached logjoint requires N by K+1 and a selected PAS index')
        cached=np.asarray(cached_log_joint,dtype=np.float64,order='C')
    else:cached=np.empty((0,0),dtype=np.float64)
    if implementation=='C1':
        from ._estep_c1 import strict_loop as core
    elif implementation=='C2':
        from ._estep_c2 import vectorized as core
    else:raise ValueError('unknown E-step implementation')
    values=core(selected,np.asarray(state.weights,dtype=np.float64),cached,selected_index,has_cache)
    likelihood,joint,z,raw,shifted,scales,raw_den,observed,q,entropy,bound=values
    matrix=lambda a:tuple(map(tuple,a.tolist()))
    likelihood=matrix(likelihood);observed=tuple(observed.tolist())
    error='ValueError' if math.isnan(bound) else None
    return em.EStepResult(tuple(row[:k] for row in likelihood),tuple(row[-1] for row in likelihood),
        matrix(joint),matrix(raw),matrix(shifted),tuple(raw_den.tolist()),tuple(scales.tolist()),
        observed,observed,matrix(z),em.Bookkeeping(q,entropy,None if error else bound,error),
        state.ids+('NOISE',),(True,)*(k+1))
