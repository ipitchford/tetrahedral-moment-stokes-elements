"""Independent exact reconstruction of the main tetrahedral facts.

Usage: python3 code/independent_check.py [--out certificates/independent_check.json]

Self-contained: own sparse polynomial arithmetic, own moment assembly and own
exact Gaussian elimination over Q (no shared library).  Checks the ranks of
M_{3,n} on P_{n+2}, the ridge functions, A(Delta4) = 1/34650, the 50 x 50
moment matrix of P4 + 14 quintics + Delta4, the ridge Gram matrix, and a few
exact ranks in dimensions 2, 4, 5.
"""
import argparse
import json
from fractions import Fraction as Fr
from itertools import combinations
from math import comb, factorial, prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exps(total, k):
    if k == 1:
        return [(total,)]
    return [(a,) + b for a in range(total + 1) for b in exps(total - a, k - 1)]


class Poly:
    def __init__(self, terms=None):
        self.t = {a: Fr(c) for a, c in (terms or {}).items() if c}

    @staticmethod
    def lam(i, nv=4):
        return Poly({tuple(int(j == i) for j in range(nv)): 1})

    @staticmethod
    def const(c, nv=4):
        return Poly({(0,) * nv: c})

    def __add__(self, o):
        r = dict(self.t)
        for a, c in o.t.items():
            r[a] = r.get(a, 0) + c
        return Poly(r)

    def __neg__(self):
        return Poly({a: -c for a, c in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if not isinstance(o, Poly):
            return Poly({a: c * o for a, c in self.t.items()})
        r = {}
        for a, c in self.t.items():
            for b, e in o.t.items():
                k = tuple(x + y for x, y in zip(a, b))
                r[k] = r.get(k, 0) + c * e
        return Poly(r)

    __rmul__ = __mul__

    def __pow__(self, k):
        r = Poly.const(1, len(next(iter(self.t))))
        for _ in range(k):
            r = r * self
        return r

    def d(self, i):
        r = {}
        for a, c in self.t.items():
            if a[i]:
                b = list(a)
                b[i] -= 1
                r[tuple(b)] = c * a[i]
        return Poly(r)

    def __eq__(self, o):
        return self.t == o.t


def dirichlet(a):
    k = len(a) - 1
    return Fr(factorial(k) * prod(factorial(x) for x in a), factorial(sum(a) + k))


def integral(p, face=None):
    s = Fr(0)
    for a, c in p.t.items():
        if face is None:
            s += c * dirichlet(a)
        elif a[face] == 0:
            s += c * dirichlet(a[:face] + a[face + 1:])
    return s


def tests(d, n):
    out = []
    for f in range(d + 1):
        for q in exps(n, d):
            out.append((f, q[:f] + (0,) + q[f:]))
    out += [(None, q) for q in exps(n - 1, d + 1)]
    return out


def moments(p, d, n):
    return [integral(p * Poly({q: 1}), f) for f, q in tests(d, n)]


def rank(rows):
    A = [list(r) for r in rows]
    rk = 0
    ncol = len(A[0]) if A else 0
    for c in range(ncol):
        piv = next((i for i in range(rk, len(A)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[rk], A[piv] = A[piv], A[rk]
        for i in range(rk + 1, len(A)):
            if A[i][c] != 0:
                f = A[i][c] / A[rk][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[rk])]
        rk += 1
    return rk


def determinant(rows):
    A = [list(r) for r in rows]
    n = len(A)
    det = Fr(1)
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            return Fr(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            det = -det
        det *= A[c][c]
        for i in range(c + 1, n):
            if A[i][c] != 0:
                f = A[i][c] / A[c][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return det


def inverse(rows):
    n = len(rows)
    A = [list(r) + [Fr(int(i == j)) for j in range(n)] for i, r in enumerate(rows)]
    for c in range(n):
        piv = next(i for i in range(c, n) if A[i][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        inv = 1 / A[c][c]
        A[c] = [x * inv for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [r[n:] for r in A]


def univariate(coeffs, s):
    r = Poly.const(0)
    for k, c in enumerate(coeffs):
        r = r + (s ** k) * c
    return r


def ridges(m):
    leg = [Fr((-1) ** (m - k) * comb(m, k) * comb(m + k, k)) for k in range(m + 1)]
    vert = [Fr(0)] + [k * leg[k] / (m + 1) for k in range(1, m + 1)]
    L = [Poly.lam(i) for i in range(4)]
    return [univariate(vert, L[i]) for i in range(4)] + [univariate(leg, L[0] + L[j]) for j in (1, 2, 3)]


def vdm(idx):
    r = Poly.const(1)
    for j, k in combinations(idx, 2):
        r = r * (Poly.lam(k) - Poly.lam(j))
    return r


def energy(u, v):
    return sum((integral((u.d(i) - u.d(j)) * (v.d(i) - v.d(j))) for i, j in combinations(range(4), 2)), Fr(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "certificates" / "independent_check.json"))
    args = ap.parse_args()
    res = {"description": "Independent exact reconstruction (own polynomial and elimination code)"}
    rr = []
    for n in range(1, 5):
        basis = [Poly({a: 1}) for a in exps(n + 2, 4)]
        M = list(zip(*(moments(p, 3, n) for p in basis)))
        r = rank(M)
        rr.append({"n": n, "rows": len(M), "columns": len(basis), "rank": r,
                   "kernel": len(basis) - r, "deficit": len(M) - r,
                   "ok": len(basis) - r == 7 and len(M) - r == (n - 1) * (n - 2) // 2})
    res["rank_P_n+2_rational"] = rr
    res["ridges_annihilate_moments_n1..6"] = all(
        not any(moments(k, 3, n)) for n in range(1, 7) for k in ridges(n + 2))
    D4 = vdm(range(4))
    fd = [vdm([j for j in range(4) if j != i]) for i in range(4)]

    def A(p):
        return sum(((-1) ** i * integral(p * fd[i], i) for i in range(4)), Fr(0))

    res["A_on_P5_monomials_zero"] = all(A(Poly({a: 1})) == 0 for t in range(6) for a in exps(t, 4))
    res["A_Delta4"] = str(A(D4))
    quint = [(0, 1, 4), (0, 2, 3), (0, 3, 2), (1, 0, 4), (1, 1, 3), (1, 2, 2), (1, 3, 1),
             (1, 4, 0), (2, 0, 3), (2, 1, 2), (2, 2, 1), (2, 3, 0), (3, 0, 2), (3, 2, 0)]
    cols = [Poly({(0,) + a: 1}) for t in range(5) for a in exps(t, 3)]
    cols += [Poly({(0,) + q: 1}) for q in quint] + [D4]
    M50 = list(zip(*(moments(p, 3, 3) for p in cols)))
    res["fifty_matrix_rank"] = rank(M50)
    res["fifty_matrix_determinant"] = str(determinant(M50))
    ks = ridges(5)
    G = [[energy(a, b) for b in ks] for a in ks]
    res["ridge_gram_m5"] = [[str(x) for x in r] for r in G]
    res["ridge_gram_m5_inverse"] = [[str(x) for x in r] for r in inverse(G)]
    gen = []
    for d, n in [(2, 1), (2, 3), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2)]:
        basis = [Poly({a: 1}) for a in exps(n + 2, d + 1)]
        M = list(zip(*(moments(p, d, n) for p in basis)))
        r = rank(M)
        k = sum(comb(d, j) for j in range(1, min(d, n + 2) + 1))
        gen.append({"d": d, "n": n, "rows": len(M), "columns": len(basis), "rank": r,
                    "kernel": len(basis) - r, "k(d,n+2)": k, "ok": len(basis) - r == k})
    res["general_dimension_rank_P_n+2_rational"] = gen
    res["all_checks_pass"] = bool(
        all(e["ok"] for e in rr) and res["ridges_annihilate_moments_n1..6"]
        and res["A_on_P5_monomials_zero"] and A(D4) == Fr(1, 34650)
        and res["fifty_matrix_rank"] == 50 and all(e["ok"] for e in gen))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, sort_keys=True, indent=1) + "\n")
    print("all_checks_pass:", res["all_checks_pass"])


if __name__ == "__main__":
    main()
