"""Optional, exact-quota-preserving Q10 feasibility repair for supportcase30.
Heuristic only: not a proof of global feasibility, optimality, or speedup.
The canonical mapping is applied ONLY after the exact full-model fingerprint passes.
"""
from math import exp
from random import Random
import numpy as np

_QUOTAS = (30, 30, 24, 30)
_NEIGHBORS = tuple(tuple([i] + [i ^ (1 << k) for k in range(10)]) for i in range(1024))

def _one_restart(rng, values, moves):
    chosen = []
    complement = []
    x = [0] * 1024
    for group, quota in enumerate(_QUOTAS):
        ids = list(range(group * 256, (group + 1) * 256))
        # The LP relaxation guides the first projection; small jitter handles ties.
        scores = [(float(values[j]) + 0.002 * rng.random(), j) for j in ids]
        scores.sort(reverse=True)
        sel = [j for _, j in scores[:quota]]
        rem = [j for _, j in scores[quota:]]
        chosen.append(sel)
        complement.append(rem)
        for j in sel:
            x[j] = 1
    cover = [0] * 1024
    for j in range(1024):
        if x[j]:
            for i in _NEIGHBORS[j]:
                cover[i] += 1
    missing = sum(v == 0 for v in cover)
    best_missing = missing
    best = x.copy()
    for step in range(moves):
        g = rng.randrange(4)
        aidx = rng.randrange(_QUOTAS[g])
        bidx = rng.randrange(256 - _QUOTAS[g])
        a, b = chosen[g][aidx], complement[g][bidx]
        delta_missing = delta_pair = 0
        for i in _NEIGHBORS[a]:
            old = cover[i]
            cover[i] = old - 1
            delta_missing += (old == 1) - (old == 0)
            delta_pair += 1 - old
        for i in _NEIGHBORS[b]:
            old = cover[i]
            cover[i] = old + 1
            delta_missing += -(old == 0)
            delta_pair += old
        fraction = (step % 15000) / 15000.0
        temperature = 1.6 * (0.06 ** fraction)
        energy = delta_missing + (0.02 + 0.01 * fraction) * delta_pair
        if energy <= 0 or rng.random() < exp(-energy / temperature):
            missing += delta_missing
            chosen[g][aidx], complement[g][bidx] = b, a
            x[a], x[b] = 0, 1
            if missing < best_missing:
                best_missing = missing
                best = x.copy()
                if missing == 0:
                    break
        else:
            for i in _NEIGHBORS[b]:
                cover[i] -= 1
            for i in _NEIGHBORS[a]:
                cover[i] += 1
    assert [sum(best[g * 256:(g + 1) * 256]) for g in range(4)] == list(_QUOTAS)
    return best_missing, best

def verified_quota_repair(problem, lp_x, moves=45000, restarts=3, seed=20261031):
    """Return a full original-column-order candidate, or None for other models."""
    from ahrh_exact_pruners import _guard
    mapping, _ = _guard(problem)
    if mapping is None:
        return None
    if lp_x is None:
        weights = np.zeros(1024, dtype=float)
    else:
        source = np.asarray(lp_x, dtype=float)
        if source.shape != (1024,) or not np.isfinite(source).all():
            return None
        weights = np.empty(1024, dtype=float)
        weights[mapping] = source
    rng = Random(int(seed))
    best_missing = 1025
    best = None
    for repeat in range(int(restarts)):
        deficit, x = _one_restart(rng, weights, int(moves))
        if deficit < best_missing:
            best_missing, best = deficit, x
        if best_missing == 0:
            break
    return np.asarray(best, dtype=float)[mapping]
