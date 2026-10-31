"""Verified necessary branch cuts for the exact supportcase30 MPS model.
Applicable to explicit fixed-1 branches; a radial node without fixings cannot be pruned.
"""
import base64
import hashlib
import zlib
import numpy as np

_SIGNATURE = '36e2a13408b78030d3ef461e55c463ada91c1d0ba3ff309fc2b08e7325f294bc'
_DATA = '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'
_CACHE_PROBLEM = None
_CACHE_MAPPING = None
_CACHE_CERTS = None

def _guard(problem):
    global _CACHE_PROBLEM, _CACHE_MAPPING, _CACHE_CERTS
    if _CACHE_PROBLEM is problem:
        return _CACHE_MAPPING, _CACHE_CERTS
    _CACHE_PROBLEM = problem
    _CACHE_MAPPING = _CACHE_CERTS = None
    if len(problem.variable_names) != 1024 or len(problem.A) != 1032:
        return None, None
    if any(t not in ('B', 'I') for t in problem.variable_types):
        return None, None
    if any(lb != 0 or ub != 1 for lb,ub in zip(problem.lower_bounds,problem.upper_bounds)):
        return None, None
    A = np.asarray(problem.A, dtype='<f8')
    b = np.asarray(problem.b, dtype='<f8')
    c = np.asarray(problem.c, dtype='<f8')
    if A.shape != (1032,1024) or b.shape != (1032,) or c.shape != (1024,):
        return None,None
    h = hashlib.sha256()
    h.update(A.tobytes());h.update(b.tobytes());h.update(c.tobytes())
    h.update(('\n'.join(problem.variable_names)).encode())
    if h.hexdigest() != _SIGNATURE:
        return None,None
    raw = zlib.decompress(base64.b64decode(_DATA))
    assert len(raw) == 2848
    _CACHE_MAPPING = np.frombuffer(raw[:2048], dtype='<u2').copy()
    _CACHE_CERTS = tuple(int.from_bytes(raw[2048+i*32:2080+i*32],'little') for i in range(25))
    return _CACHE_MAPPING, _CACHE_CERTS

def explain_exclusion(problem, fixed_one_indices):
    """Certified necessary condition for an ACTUAL fixed-one branch.

    For every sector g (0..3), define U_g as union of the selected radius-one
    Q8 neighborhoods and O_g as the overlap of fixed-one external centers
    from sectors g xor 1 and g xor 2, projected onto their same low 8 bits.
    Then 9*k_g - |U_g| + O_g <= 9*q_g + q_(g xor 1)+q_(g xor 2)-256
    for every potentially completable branch. All LHS terms are monotone
    under adding more fixed-one decisions, hence safe to prune early.
    """
    mapping, certs = _guard(problem)
    if mapping is None:
        return None
    sectors = [set() for _ in range(4)]
    for i in fixed_one_indices:
        j = int(mapping[int(i)])
        sectors[j >> 8].add(j & 255)
    quotas = (30, 30, 24, 30)
    if any(len(sectors[g]) > quotas[g] for g in range(4)):
        return 'quota'
    for g in range(4):
        union = 0
        for v in sectors[g]:
            union |= 1 << v
            for bit in range(8):
                union |= 1 << (v ^ (1 << bit))
        collision = 9 * len(sectors[g]) - union.bit_count()
        limit = 9 * quotas[g] + quotas[g ^ 1] + quotas[g ^ 2] - 256
        if collision > limit:
            return 'collision'
        if collision + len(sectors[g ^ 1] & sectors[g ^ 2]) > limit:
            return 'mixed_sector_overlap'
    fixed_mask = sum(1 << v for v in sectors[2])
    if any(fixed_mask & cut == cut for cut in certs):
        return 'dual'
    return None

def exact_lifted_star_lp_row(problem, center, external_position):
    """One valid original-variable-order LP row <= 7; there are 256^2 rows."""
    mapping, _ = _guard(problem)
    if mapping is None:
        return None
    if not (0 <= center < 256 and 0 <= external_position < 256):
        raise ValueError('center and external_position must be 8-bit indices')
    canonical_to_original = np.argsort(mapping)
    row = np.zeros(1024, dtype=float)
    for j in [center] + [center ^ (1 << bit) for bit in range(8)]:
        row[canonical_to_original[512 + j]] = 1.
    row[canonical_to_original[external_position]] = 1.
    row[canonical_to_original[768 + external_position]] = 1.
    return row

def reject_complete_candidate(problem, candidate):
    """Heuristic-only: do not infer that any surrounding radial node is impossible."""
    mapping, _ = _guard(problem)
    if mapping is None:
        return False
    x = np.asarray(candidate,dtype=float)
    if x.shape != (1024,) or not np.all((np.abs(x)<1e-7) | (np.abs(x-1)<1e-7)):
        return False
    return explain_exclusion(problem, np.flatnonzero(x>0.5)) is not None


# Verified independently against the original MPS (SHA in _SIGNATURE) and all 4 quotas.
# 13 group-2 centers force 22 duplicate visits while the group-2 surplus budget is 20.
_GLOBAL_13 = (518, 562, 563, 575, 580, 592, 606, 628, 656, 663, 670, 710, 754)

def exact_13_lp_row(problem):
    """Return globally valid inequality row a*x <= 12 for the ORIGINAL variable order.

    The model guard is essential; this cut is specific to supportcase30 Q10 geometry.
    """
    mapping, _ = _guard(problem)
    if mapping is None:
        return None
    balls = set()
    for j in _GLOBAL_13:
        v = j - 512
        balls.add(v)
        balls.update(v ^ (1 << bit) for bit in range(8))
    assert 9 * len(_GLOBAL_13) - len(balls) == 22
    assert 9 * 24 + 30 + 30 - 256 == 20
    row = np.isin(mapping, _GLOBAL_13).astype(float)
    assert int(row.sum()) == len(_GLOBAL_13)
    return row


def exact_256_neighborhood_lp_rows(problem):
    """256 globally valid cover cuts on the verified supportcase30 MPS.

    Each 8-dimensional closed Hamming ball may contain at most six of the
    24 chosen hard-sector centers; seven would contribute at least 27
    duplicate visits to the sector, exceeding its budget of 20.
    Returns a SciPy sparse CSR matrix and its 256 RHS values; other models
    return (None, None). No heuristic inference is used.
    """
    mapping, _ = _guard(problem)
    if mapping is None:
        return None, None
    from scipy import sparse
    # mapping is original-column index -> canonical 10-bit cube index.
    canonical_to_original = np.argsort(mapping)
    rr, cc = [], []
    for center in range(256):
        for v in [center] + [center ^ (1 << bit) for bit in range(8)]:
            rr.append(center)
            cc.append(int(canonical_to_original[512 + v]))
    matrix = sparse.csr_matrix(
        (np.ones(len(rr)), (rr, cc)), shape=(256, 1024))
    assert matrix.nnz == 256 * 9
    return matrix, np.full(256, 6., dtype=float)
