"""C3: chunked NumPy E-step, preserving native immutable cached-float sharing."""
import math
import numpy as np
from ._estep_c2 import vectorized

DEFAULT_CHUNK_SIZE = 512


def chunked(state, prepared, *, cached_log_joint=None, update_index=None,
            _shape_validated=False, chunk_size=None):
    from . import em
    if not _shape_validated:em._shape(state, prepared)
    size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
    if size not in (256, 512, 1024, 2048, 4096):
        raise ValueError('unsupported frozen E-step chunk size')
    f=state.fixed; k=len(state.alpha); n=len(f.molecules)
    lookup=[(f.beta_grid.index(b), f.theta.index(a)) for a,b in zip(state.alpha,state.beta)]
    has_cache=cached_log_joint is not None
    selected_index=0 if update_index is None and k==1 else update_index
    if has_cache and (type(selected_index) is not int or not 0<=selected_index<k or
                      len(cached_log_joint)!=n or any(len(row)!=k+1 for row in cached_log_joint)):
        raise ValueError('cached logjoint requires N by K+1 and a selected PAS index')
    weights=np.asarray(state.weights,dtype=np.float64)
    component=[];noise=[];joints=[];responsibility=[];raw_rows=[];shifted_rows=[]
    raw_denominators=[];scaled_denominators=[];observed_values=[]
    q=0.;entropy=0.
    for start in range(0,n,size):
        end=min(start+size,n)
        selected=np.asarray([[prepared.cube[bi][i][ti] for bi,ti in lookup]+[prepared.noise_log]
                             for i in range(start,end)],dtype=np.float64,order='C')
        cache=np.asarray(cached_log_joint[start:end],dtype=np.float64,order='C') if has_cache else np.empty((0,0),dtype=np.float64)
        arrays=vectorized(selected,weights,cache,selected_index if has_cache else 0,has_cache)
        likelihood,joint,z,raw,shifted,scales,raw_den,observed,_,_,_=arrays
        # Carry the exact running scalar into the next ordered cumulative sum.
        # Summing chunk totals would change reduction order and is forbidden.
        with np.errstate(all='ignore'):
            terms=np.where(z==0.,0.,z*joint).ravel(order='C')
            q=float(np.cumsum(np.concatenate((np.asarray([q]),terms)))[-1])
            terms=np.where(z==0.,0.,-z*np.log(z)).ravel(order='C')
            entropy=float(np.cumsum(np.concatenate((np.asarray([entropy]),terms)))[-1])
        for row in likelihood.tolist():component.append(tuple(row[:k]));noise.append(row[-1])
        joint_rows=joint.tolist()
        if has_cache:
            # Reference E-step retains these float objects in every saved trace.
            # Their binary64 values are unchanged; this preserves native sharing.
            joints.extend(tuple(row[j] if j==selected_index else cached_log_joint[start+i][j]
                                for j in range(k+1)) for i,row in enumerate(joint_rows))
        else:joints.extend(map(tuple,joint_rows))
        responsibility.extend(map(tuple,z.tolist()))
        raw_rows.extend(map(tuple,raw.tolist()))
        shifted_rows.extend(map(tuple,shifted.tolist()))
        raw_denominators.extend(raw_den.tolist());scaled_denominators.extend(scales.tolist())
        observed_values.extend(observed.tolist())
        del arrays,likelihood,joint,z,raw,shifted,scales,raw_den,observed,terms,joint_rows,selected,cache
    bound=q+entropy;error='ValueError' if math.isnan(bound) else None
    observed_values=tuple(observed_values)
    return em.EStepResult(tuple(component),tuple(noise),tuple(joints),tuple(raw_rows),tuple(shifted_rows),
        tuple(raw_denominators),tuple(scaled_denominators),observed_values,observed_values,
        tuple(responsibility),em.Bookkeeping(q,entropy,None if error else bound,error),
        state.ids+('NOISE',),(True,)*(k+1))
