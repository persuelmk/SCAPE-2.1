"""One E-step and one selected-component M-step; no iterative fitting API.

Preserves SCAPE's supplied component order, float64 floors, beta-grid boundary
smoothing, and pre-update bookkeeping. See the frozen EM primitive contracts.
"""
from dataclasses import dataclass, asdict, replace
import math
import sys
from scape2.likelihood.low_level import (fragment_likelihood, known_pa_likelihood,
                                        polya_prior, noise_likelihood)

Matrix = tuple[tuple[float, ...], ...]

def _log(value: float) -> float:
    return math.log(value) if value > 0 else -math.inf if value == 0 else math.nan

def _exp(value: float) -> float:
    try: return math.exp(value)
    except OverflowError: return math.inf

def _divide(a: float, b: float) -> float:
    return a / b if b != 0 else math.nan if a == 0 or math.isnan(a) else math.copysign(math.inf, a)

def _max(values: tuple[float, ...] | list[float]) -> float:
    return math.nan if any(math.isnan(v) for v in values) else max(values)

def _argmax(values: list[float]) -> int:
    if not values:
        raise ValueError('empty parameter support')
    for i, value in enumerate(values):
        if math.isnan(value): return i
    return max(range(len(values)), key=values.__getitem__)

@dataclass(frozen=True)
class Molecule:
    id: str
    x: float | None
    l: float | None
    r: float | None
    s: float | None
    known_pa: float | None

@dataclass(frozen=True)
class FixedInputs:
    molecules: tuple[Molecule, ...]
    theta: tuple[float, ...]
    beta_grid: tuple[float, ...]
    theta_step: float
    min_theta: float
    L: float
    mu: float = 300.
    sigma: float = 50.
    max_tail: float = 150.
    tail_lengths: tuple[float, ...] | None = None
    tail_weights: tuple[float, ...] | None = None
    protected: tuple[float, ...] | None = None

    def __post_init__(self):
        if not self.molecules or not self.theta or not self.beta_grid:
            raise ValueError('nonempty molecules and grids required')
        if any(not math.isfinite(t) for t in self.theta) or any(b <= a for a,b in zip(self.theta,self.theta[1:])):
            raise ValueError('theta grid must be finite, unique and increasing')
        if self.theta_step <= 0 or any(not math.isfinite(b) or b <= 0 for b in self.beta_grid) or len(set(self.beta_grid)) != len(self.beta_grid):
            raise ValueError('invalid beta grid or theta step')
        if len(self.theta) < 2 * math.ceil(3 * max(self.beta_grid) / self.theta_step):
            raise ValueError('grid too short for legacy smoothing boundary contract')

@dataclass(frozen=True)
class EMState:
    ids: tuple[str, ...]
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    weights: tuple[float, ...]
    fixed: FixedInputs

    def __post_init__(self):
        k=len(self.alpha)
        if k == 0 or len(self.beta)!=k or len(self.ids)!=k or len(self.weights)!=k+1 or len(set(self.ids))!=k:
            raise ValueError('K PAS IDs/parameters and K+1 weights required')
        if any(a not in self.fixed.theta for a in self.alpha) or any(b not in self.fixed.beta_grid for b in self.beta):
            raise ValueError('PAS parameters require exact grid membership')

    def parameters(self) -> dict:
        return {'ids':self.ids,'alpha':self.alpha,'beta':self.beta,'weights':self.weights}

@dataclass(frozen=True)
class PreparedLikelihoods:
    raw: Matrix
    cube: tuple[Matrix, ...]
    noise_log: float

@dataclass(frozen=True)
class Bookkeeping:
    expected_complete_log: float
    entropy: float
    pre_update_elbo: float | None
    elbo_error: str | None

@dataclass(frozen=True)
class EStepResult:
    component_likelihood: Matrix
    noise_likelihood: tuple[float, ...]
    log_joint: Matrix
    unnormalized: Matrix
    shifted_unnormalized: Matrix
    normalization_denominator: tuple[float, ...]
    scaled_denominator: tuple[float, ...]
    log_normalization: tuple[float, ...]
    log_likelihood_contribution: tuple[float, ...]
    responsibility: Matrix
    bookkeeping: Bookkeeping
    component_ids: tuple[str, ...]
    component_present: tuple[bool, ...]

@dataclass(frozen=True)
class MStepResult:
    state: EMState
    eligible_indices: tuple[int, ...]
    objective_by_beta: tuple[float, ...]
    selected_beta_index: int
    selected_objective: float


def state_from_case(case: dict) -> EMState:
    f=case['fixed']
    fixed=FixedInputs(molecules=tuple(Molecule(**m) for m in f['molecules']),
        theta=tuple(float(v) for v in f['theta']),beta_grid=tuple(float(v) for v in f['beta_grid']),
        theta_step=float(f['theta_step']),min_theta=float(f['min_theta']),L=float(f['L']),
        mu=float(f['mu']),sigma=float(f['sigma']),max_tail=float(f['max_tail']),
        tail_lengths=None if f['tail_lengths'] is None else tuple(f['tail_lengths']),
        tail_weights=None if f['tail_weights'] is None else tuple(f['tail_weights']),
        protected=None if f['protected'] is None else tuple(f['protected']))
    s=case['state']
    return EMState(tuple(s['ids']),tuple(map(float,s['alpha'])),tuple(map(float,s['beta'])),tuple(map(float,s['weights'])),fixed)


def _smooth(raw: Matrix, beta: float, step: float) -> Matrix:
    scale=beta/step;half=math.ceil(3*scale);width=len(raw[0])
    kernel=[math.exp(-.5*(t/scale)**2)/math.sqrt(2*math.pi)/scale for t in range(-half,half)]
    total=sum(kernel);kernel=[w/total for w in kernel]
    rows=[]
    for row in raw:
        values=[]
        for column in range(width):
            indices=list(range(column-half,column+half))
            if column < half:
                indices=[j for j in indices if j>=0]
                weights=[kernel[j] for j in indices]
                norm=sum(weights);weights=[w/norm for w in weights]
            elif column >= width-half:
                weights=[w for j,w in zip(indices,kernel) if j<width]
                indices=[j for j in indices if j<width]
                norm=sum(weights);weights=[w/norm for w in weights]
            else:weights=kernel
            value=sum(w*row[j] for j,w in zip(indices,weights))
            values.append(_log(max(value,sys.float_info.min)))
        rows.append(tuple(values))
    return tuple(rows)


def prepare_likelihoods_reference(fixed: FixedInputs) -> PreparedLikelihoods:
    """Prepare supplied grids using previously validated fragment/poly(A)/noise."""
    prior=polya_prior(fixed.tail_lengths,fixed.tail_weights);raw=[]
    for molecule in fixed.molecules:
        args=(molecule.x,molecule.l,molecule.r,molecule.s)
        if molecule.known_pa is None or math.isnan(molecule.known_pa):
            row=tuple(fragment_likelihood(*args,t,fixed.mu,fixed.sigma,prior=prior) for t in fixed.theta)
        else:
            nearest=min(range(len(fixed.theta)),key=lambda j:abs(fixed.theta[j]-molecule.known_pa))
            row=tuple(known_pa_likelihood(*args,t,prior=prior) if j==nearest else 0. for j,t in enumerate(fixed.theta))
        raw.append(row)
    raw=tuple(raw)
    return PreparedLikelihoods(raw,tuple(_smooth(raw,b,fixed.theta_step) for b in fixed.beta_grid),noise_likelihood(fixed.L,fixed.max_tail,log=True))


def prepare_likelihoods(fixed: FixedInputs) -> PreparedLikelihoods:
    """Prepare supplied grids using previously validated fragment/poly(A)/noise."""
    from ._likelihood_prep import smooth
    prior=polya_prior(fixed.tail_lengths,fixed.tail_weights);raw=[]
    for molecule in fixed.molecules:
        args=(molecule.x,molecule.l,molecule.r,molecule.s)
        if molecule.known_pa is None or math.isnan(molecule.known_pa):
            row=tuple(fragment_likelihood(*args,t,fixed.mu,fixed.sigma,prior=prior) for t in fixed.theta)
        else:
            nearest=min(range(len(fixed.theta)),key=lambda j:abs(fixed.theta[j]-molecule.known_pa))
            row=tuple(known_pa_likelihood(*args,t,prior=prior) if j==nearest else 0. for j,t in enumerate(fixed.theta))
        raw.append(row)
    raw=tuple(raw)
    return PreparedLikelihoods(raw,tuple(smooth(raw,b,fixed.theta_step) for b in fixed.beta_grid),noise_likelihood(fixed.L,fixed.max_tail,log=True))


def _shape(state: EMState, prepared: PreparedLikelihoods) -> None:
    n=len(state.fixed.molecules);g=len(state.fixed.theta)
    if len(prepared.raw)!=n or any(len(row)!=g for row in prepared.raw) or len(prepared.cube)!=len(state.fixed.beta_grid):
        raise ValueError('likelihood dimensions disagree with fixed state')
    if any(len(matrix)!=n or any(len(row)!=g for row in matrix) for matrix in prepared.cube):
        raise ValueError('beta likelihood matrix dimensions disagree with fixed state')


def _immutable_shape_inputs(state: EMState, prepared: PreparedLikelihoods) -> bool:
    """Only deeply immutable shape inputs may reuse fixed-K validation."""
    f=state.fixed
    return (type(f.molecules) is tuple and type(f.theta) is tuple and
            type(f.beta_grid) is tuple and type(prepared.raw) is tuple and
            all(type(row) is tuple for row in prepared.raw) and
            type(prepared.cube) is tuple and
            all(type(matrix) is tuple and all(type(row) is tuple for row in matrix)
                for matrix in prepared.cube))


def normalize_responsibilities(log_joint: Matrix) -> Matrix:
    if not log_joint or not log_joint[0] or any(len(row)!=len(log_joint[0]) for row in log_joint):
        raise ValueError('nonempty rectangular log-joint matrix required')
    result=[]
    for row in log_joint:
        shift=_max(row);values=tuple(_exp(v-shift) for v in row);denominator=sum(values)
        result.append(tuple(_divide(v,denominator) for v in values))
    return tuple(result)


def expected_complete_log(log_joint: Matrix, responsibility: Matrix) -> float:
    return sum(0. if z==0 else z*a for row,zrow in zip(log_joint,responsibility,strict=True) for a,z in zip(row,zrow,strict=True))


def elbo(log_joint: Matrix, responsibility: Matrix) -> float:
    q=expected_complete_log(log_joint,responsibility)
    entropy=sum(0. if z==0 else -z*_log(z) for row in responsibility for z in row)
    result=q+entropy
    if math.isnan(result):raise ValueError('lower bounder is na.')
    return result


def e_step_reference(state: EMState, prepared: PreparedLikelihoods, *, cached_log_joint: Matrix | None = None,
           update_index: int | None = None, _shape_validated: bool = False) -> EStepResult:
    """Complete first E-step: fill all columns once, then normalize once."""
    if not _shape_validated:_shape(state,prepared)
    f=state.fixed;k=len(state.alpha);likelihood=[];joints=[]
    lookup=[(f.beta_grid.index(b),f.theta.index(a)) for a,b in zip(state.alpha,state.beta)]
    for i in range(len(f.molecules)):
        selected=[prepared.cube[bi][i][ti] for bi,ti in lookup]+[prepared.noise_log]
        likelihood.append(tuple(_exp(v) for v in selected))
        joints.append(tuple(_log(max(w,sys.float_info.min))+v for w,v in zip(state.weights,selected)))
    joint=tuple(joints)
    if cached_log_joint is not None:
        # Legacy em_algo refreshes only the selected PAS column. Other PAS
        # columns and noise retain their previous log-joints, including weights.
        selected=0 if update_index is None and k==1 else update_index
        if (type(selected) is not int or not 0<=selected<k or
            len(cached_log_joint)!=len(joint) or any(len(row)!=k+1 for row in cached_log_joint)):
            raise ValueError('cached logjoint requires N by K+1 and a selected PAS index')
        joint=tuple(tuple(row[j] if j==selected else cached[j] for j in range(k+1))
                    for row,cached in zip(joint,cached_log_joint))
    z=normalize_responsibilities(joint)
    raw=tuple(tuple(_exp(v) for v in row) for row in joint)
    row_max=tuple(_max(row) for row in joint)
    shifted=tuple(tuple(_exp(v-maximum) for v in row) for row,maximum in zip(joint,row_max))
    scales=tuple(sum(row) for row in shifted)
    observed=tuple(maximum+_log(total) for maximum,total in zip(row_max,scales))
    q=expected_complete_log(joint,z);entropy=sum(0. if v==0 else -v*_log(v) for row in z for v in row)
    error=None
    try:
        bound=q+entropy
        if math.isnan(bound):raise ValueError('lower bounder is na.')
    except ValueError:bound=None;error='ValueError'
    return EStepResult(tuple(row[:k] for row in likelihood),tuple(row[-1] for row in likelihood),joint,raw,shifted,tuple(sum(row) for row in raw),scales,observed,observed,z,Bookkeeping(q,entropy,bound,error),state.ids+('NOISE',),(True,)*(k+1))


_ESTEP_IMPLEMENTATION = 'C3'

def e_step(state: EMState, prepared: PreparedLikelihoods, *, cached_log_joint: Matrix | None = None,
           update_index: int | None = None, _shape_validated: bool = False,
           implementation: str | None = None) -> EStepResult:
    """Dispatch implementation only; the P3B reference is retained verbatim."""
    selected = _ESTEP_IMPLEMENTATION if implementation is None else implementation
    if selected == 'reference':
        return e_step_reference(state, prepared, cached_log_joint=cached_log_joint,
                                update_index=update_index, _shape_validated=_shape_validated)
    if selected == 'C3':
        from ._estep_c3 import chunked
        return chunked(state, prepared, cached_log_joint=cached_log_joint,
                       update_index=update_index, _shape_validated=_shape_validated)
    from ._estep_adapter import accelerated
    return accelerated(state, prepared, cached_log_joint=cached_log_joint,
                       update_index=update_index, _shape_validated=_shape_validated,
                       implementation=selected)


def update_weights(responsibility: Matrix) -> tuple[float, ...]:
    if not responsibility or not responsibility[0] or any(len(row)!=len(responsibility[0]) for row in responsibility):
        raise ValueError('nonempty rectangular responsibility matrix required')
    return tuple(sum(row[j] for row in responsibility)/len(responsibility) for j in range(len(responsibility[0])))


def m_step(state: EMState, prepared: PreparedLikelihoods, responsibility: Matrix, k: int,
           *, _shape_validated: bool = False) -> MStepResult:
    """Update every weight and exactly one PAS alpha/beta, without reordering."""
    if not _shape_validated:_shape(state,prepared)
    f=state.fixed;K=len(state.alpha)
    if not isinstance(k,int) or not 0<=k<K:raise ValueError('update index must name a PAS component')
    if len(responsibility)!=len(f.molecules) or any(len(row)!=K+1 for row in responsibility):
        raise ValueError('responsibility matrix requires N by K+1 including noise')
    weights=update_weights(responsibility)
    if f.protected is not None and state.alpha[k] in f.protected:
        eligible=tuple(j for j,t in enumerate(f.theta) if t==state.alpha[k])
    else:
        lower=f.min_theta if k==0 else state.alpha[k-1]+state.beta[k-1]
        upper=f.L if k==K-1 else state.alpha[k+1]
        eligible=tuple(j for j,t in enumerate(f.theta) if lower<=t<=upper)
    if not eligible:raise ValueError('empty parameter support')
    objectives=[];positions=[]
    for matrix in prepared.cube:
        scores=[sum(row[k]*matrix[i][j] for i,row in enumerate(responsibility) if row[k]>0) for j in eligible]
        index=_argmax(scores);objectives.append(scores[index]);positions.append(f.theta[eligible[index]])
    winner=_argmax(objectives);alpha=list(state.alpha);beta=list(state.beta)
    alpha[k]=positions[winner];beta[k]=f.beta_grid[winner]
    updated=replace(state,alpha=tuple(alpha),beta=tuple(beta),weights=weights)
    return MStepResult(updated,eligible,tuple(objectives),winner,objectives[winner])


def m_step_output(state: EMState, prepared: PreparedLikelihoods, responsibility: Matrix,
                  k: int, pre_update_bookkeeping: dict) -> dict:
    """Serialization boundary retains legacy empty-support errors in audits."""
    try:
        result=m_step(state,prepared,responsibility,k)
    except ValueError as error:
        if str(error)!='empty parameter support':raise
        return {'status':'ERROR','error_class':'ValueError','error_scope':'empty_parameter_support','component_ids':state.ids+('NOISE',)}
    return {'status':'VALUE','state':result.state.parameters(),'eligible_indices':result.eligible_indices,'objective_by_beta':result.objective_by_beta,'selected_beta_index':result.selected_beta_index,'selected_objective':result.selected_objective,'component_present':(True,)*len(state.weights),'positive_weight':tuple(w>0 for w in result.state.weights),'pre_update_bookkeeping':pre_update_bookkeeping}


def one_cycle(state: EMState, prepared: PreparedLikelihoods, k: int) -> dict:
    """Exactly one E then one M. No likelihood refresh or second E after M."""
    e=e_step(state,prepared)
    m=m_step_output(state,prepared,e.responsibility,k,asdict(e.bookkeeping))
    return {'estep':asdict(e),'mstep':m,'e_calls':1,'m_calls':1}
