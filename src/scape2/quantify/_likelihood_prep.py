"""Exact P3C smoothing geometry with ordered float64 batch multiplication/sums.

Kernel support, asymmetric left boundary, normalization order and scalar log
are native. No parameter-dependent cache; geometry lives for one beta call.
"""
import math
import sys
import numpy as np

def _log(value):
    return math.log(value) if value > 0 else -math.inf if value == 0 else math.nan

def _plan(width,beta,step):
    scale=beta/step;half=math.ceil(3*scale)
    kernel=[math.exp(-.5*(t/scale)**2)/math.sqrt(2*math.pi)/scale for t in range(-half,half)]
    total=sum(kernel);kernel=[w/total for w in kernel];out=[]
    for column in range(width):
        indices=list(range(column-half,column+half))
        if column<half:
            indices=[j for j in indices if j>=0];weights=[kernel[j] for j in indices]
            norm=sum(weights);weights=[w/norm for w in weights]
        elif column>=width-half:
            weights=[w for j,w in zip(indices,kernel) if j<width];indices=[j for j in indices if j<width]
            norm=sum(weights);weights=[w/norm for w in weights]
        else:weights=kernel
        out.append((indices,weights))
    return out

def _smooth_batch(raw,beta,step):
    geometry=_plan(len(raw[0]),beta,step)
    data=np.asarray(raw,dtype=np.float64);out=np.empty(data.shape,dtype=np.float64)
    # Each column retains native index/weight order. cumsum is sequential, unlike sum/dot.
    # No padding: 0*NaN from outside the native support would change masks.
    for column,(indices,weights) in enumerate(geometry):
        products=data[:,indices]*np.asarray(weights,dtype=np.float64)
        out[:,column]=np.cumsum(products,axis=1)[:,-1]
    # Scalar libm log preserves P3C exact canonical floats (np.log need not).
    return tuple(tuple(_log(max(float(value),sys.float_info.min)) for value in row) for row in out)


def smooth(raw,beta,step):
    # L4's exact scalar sharing plus bounded molecule materialization; no recompute.
    floor=_log(sys.float_info.min)
    rows=[]
    for start in range(0,len(raw),256):
        block=_smooth_batch(raw[start:start+256],beta,step)
        rows.extend(tuple(floor if value==floor else value for value in row) for row in block)
    return tuple(rows)
