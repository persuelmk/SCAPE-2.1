"""Legacy endpoint callers at the region boundary; no sandbox candidates."""
from collections import Counter
import numpy as np

def smooth(values, bandwidth, edge):
    data=np.asarray(values,dtype=float)
    if not len(data):return data.copy()
    radius=max(2,int(np.ceil(4*bandwidth)));x=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(x/max(bandwidth,1.))**2);kernel/=kernel.sum()
    return np.convolve(np.pad(data,(radius,radius),mode=edge),kernel,mode='valid')

def direct_pas(positions,radius=50,merge_distance=25,min_support=3,bandwidth=3.,method='cdf'):
    if method=='gaussian':return gaussian_pas(positions,radius,merge_distance,min_support,bandwidth)
    if method!='cdf':raise ValueError("method must be 'cdf' or 'gaussian'")
    if not positions:return []
    raw=np.asarray(list(map(int,positions)));counts=Counter(raw.tolist())
    anchors=sorted(p for p,n in counts.items() if n>=3);intervals=[]
    for p in anchors:
        lo,hi=p-10,p+10
        if intervals and lo<=intervals[-1][1]+1:intervals[-1][1]=max(intervals[-1][1],hi)
        else:intervals.append([lo,hi])
    peaks=[];density_by_region={}
    for rid,(lo,hi) in enumerate(intervals):
        grid=np.arange(lo,hi+1);hist=np.asarray([counts.get(int(p),0) for p in grid],dtype=float)
        if hist.sum()<min_support or len(grid)<3:continue
        cdf=np.cumsum(hist)/hist.sum();density=smooth(np.gradient(smooth(cdf,bandwidth,'edge')),bandwidth,'edge')
        density_by_region[rid]=(grid,density)
        maxima=[i for i,v in enumerate(density) if v>=(density[i-1] if i else -np.inf) and v>=(density[i+1] if i+1<len(density) else -np.inf)]
        if not maxima and len(density):maxima=[int(np.argmax(density))]
        for i in maxima:peaks.append(dict(region=rid,index=i,position=int(grid[i]),density=float(density[i])))
    retained=[]
    for p in sorted(peaks,key=lambda p:(p['region'],p['position'])):
        if retained and p['region']==retained[-1]['region'] and p['position']-retained[-1]['position']<=20:
            if p['density']>retained[-1]['density']:retained[-1]=p
        else:retained.append(p)
    result=[]
    for rid in sorted({p['region'] for p in retained}):
        local=[p for p in retained if p['region']==rid];grid,density=density_by_region[rid];valleys=[]
        for a,b in zip(local,local[1:]):
            lo,hi=sorted([a['index'],b['index']]);valleys.append(int(grid[lo+int(np.argmin(density[lo:hi+1]))]))
        for i,p in enumerate(local):
            basin_lo=int(grid[0]) if i==0 else valleys[i-1]+1;basin_hi=int(grid[-1]) if i+1==len(local) else valleys[i]
            lo=max(basin_lo,p['position']-10);hi=min(basin_hi,p['position']+10)
            observed=[v for v in counts if lo<=v<=hi]
            if not observed:continue
            maximum=max(counts[v] for v in observed)
            center=min([v for v in observed if counts[v]==maximum],key=lambda v:(abs(v-p['position']),v))
            lo=max(basin_lo,center-25);hi=min(basin_hi,center+25);support=raw[(raw>=lo)&(raw<=hi)]
            if len(support)<min_support:continue
            result.append(dict(position=int(center),density_position=p['position'],direct_support=len(support),endpoint_median=int(np.median(support)),interval_start=int(lo),interval_end=int(hi),peak_density=p['density']))
    return sorted(result,key=lambda x:x['position'])

def gaussian_pas(positions,radius,merge_distance,min_support,bandwidth):
    groups=[]
    for p in sorted(map(int,positions)):
        if not groups or p-groups[-1][0]>radius:groups.append([p])
        else:groups[-1].append(p)
    calls=[]
    for group in groups:
        if len(group)<min_support:continue
        grid=np.arange(min(group),max(group)+1);counts=Counter(group);values=smooth([counts.get(int(p),0) for p in grid],bandwidth,'constant')
        maxima=[i for i in range(len(grid)) if values[i]>=values[max(0,i-1)] and values[i]>=values[min(len(grid)-1,i+1)]] or [int(np.argmax(values))]
        center=int(grid[max(maxima,key=lambda i:values[i])])
        calls.append(dict(position=center,direct_support=sum(abs(v-center)<=radius for v in group),endpoint_median=int(np.median(group))))
    result=[]
    for c in sorted(calls,key=lambda x:x['position']):
        if result and c['position']-result[-1]['position']<=merge_distance:
            old=result[-1];total=old['direct_support']+c['direct_support'];old['position']=int(round((old['position']*old['direct_support']+c['position']*c['direct_support'])/total));old['direct_support']=total
        else:result.append(dict(c))
    return result
