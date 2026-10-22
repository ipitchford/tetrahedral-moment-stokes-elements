"""Semantic negative controls (standard library only, exact rational arithmetic).

Recomputes, independently of code/, the alternating face functional
A(v) = sum_i avg_{F_i} v * eps_i of Proposition 3.2 on every homogeneous
quintic monomial in the barycentric coordinates and on the Vandermonde sextic
Delta_4, checks the rank table of certificates/tetrahedron_exact.json against
the closed forms of Theorem 2.1, and confirms that plausible corruptions are
rejected. Run: python3 -B -O negative_controls.py
"""
import json
import sys
from fractions import Fraction
from itertools import combinations
from math import comb, factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def mul(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items():
            e = tuple(i + j for i, j in zip(a, b))
            out[e] = out.get(e, 0) + x * y
    return {e: c for e, c in out.items() if c}


def lin(i, j):  # lambda_j - lambda_i
    ei = tuple(1 if k == i else 0 for k in range(4))
    ej = tuple(1 if k == j else 0 for k in range(4))
    return {ej: 1, ei: -1}


def vandermonde(idx):
    p = {(0, 0, 0, 0): 1}
    for i, j in combinations(idx, 2):
        p = mul(p, lin(i, j))
    return p


def face_avg(p, i):
    """Normalised integral over the face lambda_i = 0 of a homogeneous polynomial."""
    total = Fraction(0)
    for e, c in p.items():
        if e[i]:
            continue
        k = sum(e)
        num = 2
        for j in range(4):
            if j != i:
                num *= factorial(e[j])
        total += Fraction(c * num, factorial(k + 2))
    return total


def eps(i, signed=True):
    p = vandermonde([j for j in range(4) if j != i])
    s = (-1) ** i if signed else 1
    return {e: s * c for e, c in p.items()}


def A(v, signed=True):
    return sum(face_avg(mul(v, eps(i, signed)), i) for i in range(4))


def quintic_monomials():
    for a in range(6):
        for b in range(6 - a):
            for c in range(6 - a - b):
                yield {(a, b, c, 5 - a - b - c): 1}


def check(cond, msg):
    if not cond:
        print("FAIL ", msg)
        sys.exit(1)
    print("PASS ", msg)


def main():
    quintics = list(quintic_monomials())
    check(len(quintics) == 56, "56 homogeneous quintic monomials span P5")
    check(all(A(q) == 0 for q in quintics), "A vanishes on every quintic")
    d4 = vandermonde(range(4))
    val = A(d4)
    check(val == Fraction(1, 34650), "A(Delta_4) = 1/34650")
    # Negative controls: corrupted claims must be rejected.
    check(val != Fraction(1, 34651), "control: wrong obstruction value 1/34651 rejected")
    check(any(A(q, signed=False) != 0 for q in quintics),
          "control: dropping the alternating signs (-1)^i breaks vanishing on P5")
    cert = json.loads((ROOT / "certificates" / "tetrahedron_exact.json").read_text())
    rows = cert["rank_table"]
    check(len(rows) >= 6, "certificate rank table covers n = 1..6")
    for n, r in enumerate(rows, start=1):
        N = 4 * comb(n + 2, 2) + comb(n + 2, 3)
        check(r["N"] == N, f"n={n}: N_n = {N}")
        check(r["P_n+1"]["rank"] == comb(n + 4, 3), f"n={n}: injective on P_(n+1)")
        rk = r["P_n+2"]["rank"]
        check(rk == comb(n + 5, 3) - 7, f"n={n}: rank on P_(n+2) = C(n+5,3) - 7 = {rk}")
        check(N - rk == (n - 1) * (n - 2) // 2, f"n={n}: deficit (n-1)(n-2)/2 = {N - rk}")
        if n >= 3:
            check(N - rk != n * (n - 1) // 2, f"n={n}: control: corrupted deficit n(n-1)/2 rejected")
        check(r["P_n+3"]["rank"] == N, f"n={n}: surjective on P_(n+3)")
    print("OVERALL PASS")


if __name__ == "__main__":
    main()
