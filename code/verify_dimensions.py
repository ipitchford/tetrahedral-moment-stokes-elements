"""Deficit table and exact ranks of M_{d,n} on P_{n+2} and P_{n+3} for d = 2, 4, 5, 6.

Usage: python3 code/verify_dimensions.py [--out certificates/dimensions_exact.json]

delta(d,n) = N_{d,n} - (C(n+2+d,d) - k(d,n+2)), with k(d,m) = sum_{r=1}^{min(d,m)} C(d,r).

Binomial-basis closed form.  Writing every term in the basis C(d,r) (Vandermonde
convolution) gives
    delta(d,n) = sum_{r=3}^{min(d,n)} C(d,r) [ (r-2) C(n,r-1) - C(n,r-2) + 1 ],
whose coefficients are positive for 3 <= r <= n.

Ranks: exact rational elimination for small matrices, rank modulo p = 2^61-1
otherwise (a lower bound for the rank over Q).  Full row rank modulo p on
P_{n+3} is therefore an exact certificate of surjectivity.  On P_{n+2} the
modular rank is a lower bound; the matching upper bound is the kernel
dimension k(d,n+2) of the kernel theorem.
"""
import argparse
from math import comb
from pathlib import Path

import sympy as sp

import fe_exact_lib as L

ROOT = Path(__file__).resolve().parents[1]

RANK_CASES = {2: range(1, 7), 4: range(1, 6), 5: range(1, 5), 6: range(1, 4)}


def delta(d, n):
    return L.N_dofs(d, n) - (L.dim_P(n + 2, d) - L.kernel_dim(d, n + 2))


def delta_binomial(d, n):
    return sum(comb(d, r) * ((r - 2) * comb(n, r - 1) - comb(n, r - 2) + 1)
               for r in range(3, min(d, n) + 1))


def closed_forms():
    n = sp.symbols("n")
    out = {}
    for d in range(2, 9):
        N = (d + 1) * sp.binomial(n + d - 1, d - 1) + sp.binomial(n + d - 1, d)
        P = sp.binomial(n + d + 2, d)
        poly = sp.factor(sp.expand(sp.expand_func(N - P)) + 2 ** d - 1)
        valid_from = max(1, d - 2)
        ok = all(poly.subs(n, k) == delta(d, k) for k in range(valid_from, 41))
        exceptions = {str(k): [delta(d, k), int(poly.subs(n, k))]
                      for k in range(1, valid_from) if poly.subs(n, k) != delta(d, k)}
        out[str(d)] = {"polynomial_in_n": str(poly), "valid_for_n_>=": valid_from,
                       "checked_n_range": [valid_from, 40], "agrees": ok,
                       "n_below_range_where_polynomial_differs_[delta,poly]": exceptions}
    return out


def rank_cases():
    out = []
    for d, ns in RANK_CASES.items():
        for n in ns:
            N = L.N_dofs(d, n)
            e = {"d": d, "n": n, "N": N}
            for label, D in (("P_n+2", n + 2), ("P_n+3", n + 3)):
                M, rows, cols = L.moment_matrix_monomials(d, n, D)
                r, method = L.rank_certified(M)
                e[label] = {"degree": D, "rows": len(rows), "columns": len(cols),
                            "rank": r, "method": method}
            pred = L.dim_P(n + 2, d) - L.kernel_dim(d, n + 2)
            e["predicted_rank_P_n+2"] = pred
            e["kernel_dim_k(d,n+2)"] = L.kernel_dim(d, n + 2)
            e["rank_P_n+2_matches_prediction"] = e["P_n+2"]["rank"] == pred
            e["deficit_P_n+2"] = N - e["P_n+2"]["rank"]
            e["surjective_on_P_n+3"] = e["P_n+3"]["rank"] == N
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "certificates" / "dimensions_exact.json"))
    args = ap.parse_args()
    table = {str(d): [delta(d, n) for n in range(1, 11)] for d in range(2, 9)}
    binom_ok = all(delta(d, n) == delta_binomial(d, n) for d in range(1, 13) for n in range(1, 41))
    positivity = {
        "delta(2,n)=0_for_n<=40": all(delta(2, n) == 0 for n in range(1, 41)),
        "delta(d,1)=delta(d,2)=0_for_d<=12": all(delta(d, 1) == 0 == delta(d, 2) for d in range(1, 13)),
        "delta(d,n)>0_for_3<=d<=12,3<=n<=40": all(delta(d, n) > 0 for d in range(3, 13) for n in range(3, 41)),
        "delta(d,3)=C(d,3)_for_d<=12": all(delta(d, 3) == comb(d, 3) for d in range(1, 13)),
    }
    cases = rank_cases()
    data = {
        "description": "Deficit delta(d,n) and exact ranks of M_{d,n}",
        "prime": "2^61-1",
        "delta_table_d2..8_n1..10": table,
        "delta_closed_form_binomial":
            "delta(d,n) = sum_{r=3}^{min(d,n)} C(d,r) [ (r-2) C(n,r-1) - C(n,r-2) + 1 ]",
        "delta_closed_form_binomial_checked_d1..12_n1..40": binom_ok,
        "delta_polynomial_in_n_per_d": closed_forms(),
        "delta_sign_pattern": positivity,
        "rank_cases": cases,
    }
    data["all_checks_pass"] = bool(binom_ok and all(positivity.values())
                                   and all(v["agrees"] for v in data["delta_polynomial_in_n_per_d"].values())
                                   and all(c["rank_P_n+2_matches_prediction"] and c["surjective_on_P_n+3"]
                                           for c in cases))
    L.write_json(data, args.out)
    print("all_checks_pass:", data["all_checks_pass"])


if __name__ == "__main__":
    main()
