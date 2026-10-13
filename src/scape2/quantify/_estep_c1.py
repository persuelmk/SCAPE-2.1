"""P3C implementation candidates; formulas and output contract remain unchanged."""
import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, parallel=False, error_model='numpy')
def _log(x):
    return math.log(x) if x > 0. else -math.inf if x == 0. else math.nan


@njit(cache=True, fastmath=False, parallel=False, error_model='numpy')
def strict_loop(selected, weights, cached, update_index, has_cache):
    n,d=selected.shape
    likelihood=np.empty((n,d),np.float64)
    joint=np.empty((n,d),np.float64)
    z=np.empty((n,d),np.float64)
    raw=np.empty((n,d),np.float64)
    shifted=np.empty((n,d),np.float64)
    maxima=np.empty(n,np.float64)
    scales=np.empty(n,np.float64)
    raw_den=np.empty(n,np.float64)
    observed=np.empty(n,np.float64)
    for i in range(n):
        for j in range(d):likelihood[i,j]=math.exp(selected[i,j])
        for j in range(d):
            w=weights[j]
            # Python max(weight, tiny) keeps a NaN first argument.
            if w < 2.2250738585072014e-308:w=2.2250738585072014e-308
            joint[i,j]=_log(w)+selected[i,j]
    if has_cache:
        for i in range(n):
            for j in range(d):
                if j!=update_index:joint[i,j]=cached[i,j]
    for i in range(n):
        maximum=joint[i,0]
        for j in range(d):
            v=joint[i,j]
            if math.isnan(v):maximum=math.nan
            elif v>maximum:maximum=v
        total=0.
        for j in range(d):
            z[i,j]=math.exp(joint[i,j]-maximum)
            total=total+z[i,j]
        for j in range(d):z[i,j]=z[i,j]/total
    for i in range(n):
        for j in range(d):raw[i,j]=math.exp(joint[i,j])
    for i in range(n):
        maximum=joint[i,0]
        for j in range(d):
            v=joint[i,j]
            if math.isnan(v):maximum=math.nan
            elif v>maximum:maximum=v
        maxima[i]=maximum
    for i in range(n):
        for j in range(d):shifted[i,j]=math.exp(joint[i,j]-maxima[i])
    for i in range(n):
        total=0.
        for j in range(d):total=total+shifted[i,j]
        scales[i]=total
    for i in range(n):observed[i]=maxima[i]+_log(scales[i])
    q=0.
    for i in range(n):
        for j in range(d):
            v=0. if z[i,j]==0. else z[i,j]*joint[i,j]
            q=q+v
    entropy=0.
    for i in range(n):
        for j in range(d):
            v=0. if z[i,j]==0. else -z[i,j]*_log(z[i,j])
            entropy=entropy+v
    bound=q+entropy
    for i in range(n):
        total=0.
        for j in range(d):total=total+raw[i,j]
        raw_den[i]=total
    return likelihood,joint,z,raw,shifted,scales,raw_den,observed,q,entropy,bound
