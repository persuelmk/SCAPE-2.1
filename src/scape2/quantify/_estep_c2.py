"""C2: binary64 vectorization with ordered cumulative reductions."""
import numpy as np

def vectorized(selected,weights,cached,update_index,has_cache):
    with np.errstate(all='ignore'):
        likelihood=np.exp(selected)
        joint=np.log(np.maximum(weights,np.finfo(np.float64).tiny))[None,:]+selected
        if has_cache:
            fresh=joint[:,update_index].copy()
            joint=cached.copy();joint[:,update_index]=fresh
        maxima=np.max(joint,axis=1)
        shifted=np.exp(joint-maxima[:,None])
        scales=np.cumsum(shifted,axis=1)[:,-1]
        z=shifted/scales[:,None]
        raw=np.exp(joint)
        raw_den=np.cumsum(raw,axis=1)[:,-1]
        observed=maxima+np.log(scales)
        q=float(np.cumsum(np.where(z==0.,0.,z*joint).ravel(order='C'))[-1])
        entropy=float(np.cumsum(np.where(z==0.,0.,-z*np.log(z)).ravel(order='C'))[-1])
        bound=q+entropy
    return likelihood,joint,z,raw,shifted,scales,raw_den,observed,q,entropy,bound
