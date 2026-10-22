"""Exact checks for the tetrahedral moment map M_{3,n} and the 50-dof element.

Usage: python3 code/verify_tetrahedron.py [--out certificates/tetrahedron_exact.json]

Rank statements use exact rational elimination for small matrices and rank
modulo p = 2^61 - 1 otherwise.  A rank modulo p is a lower bound for the rank
over Q, so full row or column rank modulo p is an exact certificate; the
kernel on P_{n+2} is bounded below exactly by the seven ridge functions.
"""
import argparse
import random
from fractions import Fraction as F
from math import comb
from pathlib import Path

import fe_exact_lib as L

ROOT = Path(__file__).resolve().parents[1]

QUINTICS = [(0, 1, 4), (0, 2, 3), (0, 3, 2), (1, 0, 4), (1, 1, 3), (1, 2, 2), (1, 3, 1),
            (1, 4, 0), (2, 0, 3), (2, 1, 2), (2, 2, 1), (2, 3, 0), (3, 0, 2), (3, 2, 0)]
EXCLUDED = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (4, 1, 0), (4, 0, 1), (0, 4, 1), (3, 1, 1)]
S = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]


def coeff_vector(p, basis):
    return [p.get(a, F(0)) for a in basis]


def rank_table():
    out = []
    for n in range(1, 7):
        entry = {"n": n, "N": L.N_dofs(3, n)}
        rows = L.moment_rows(3, n)
        for label, D in (("P_n+1", n + 1), ("P_n+2", n + 2), ("P_n+3", n + 3)):
            M, _, cols = L.moment_matrix_monomials(3, n, D)
            r, method = L.rank_certified(M)
            entry[label] = {"degree": D, "columns": len(cols), "rows": len(rows),
                            "rank": r, "method": method}
        e = entry
        e["injective_on_P_n+1"] = e["P_n+1"]["rank"] == e["P_n+1"]["columns"]
        e["kernel_P_n+2"] = e["P_n+2"]["columns"] - e["P_n+2"]["rank"]
        e["deficit_P_n+2"] = e["N"] - e["P_n+2"]["rank"]
        e["deficit_formula_(n-1)(n-2)/2"] = (n - 1) * (n - 2) // 2
        e["surjective_on_P_n+3"] = e["P_n+3"]["rank"] == e["N"]
        # ridge functions: exact zero moments and exact independence
        m = n + 2
        ks = L.tet_ridges(m)
        zero = all(not any(L.poly_moment_vector(k, rows)) for k in ks)
        basis = list(L.compositions(m, 4))
        hk = [L.homogenise(k, m, 4) for k in ks]
        indep = L.rank_exact([coeff_vector(k, basis) for k in hk])
        e["ridges_annihilate_all_moments"] = zero
        e["ridges_independent_rank"] = indep
        e["kernel_P_n+2_exact"] = (zero and indep == 7 and e["kernel_P_n+2"] == 7)
        e["all_pass"] = (e["injective_on_P_n+1"] and e["kernel_P_n+2_exact"]
                         and e["deficit_P_n+2"] == e["deficit_formula_(n-1)(n-2)/2"]
                         and e["surjective_on_P_n+3"])
        out.append(e)
    return out


def random_poly(rng, deg, terms):
    p = {}
    for _ in range(terms):
        D = rng.randint(0, deg)
        mons = list(L.compositions(D, 4))
        a = mons[rng.randrange(len(mons))]
        p[a] = p.get(a, F(0)) + F(rng.randint(-9, 9), rng.randint(1, 5))
    return L.clean(p)


def alternating_checks():
    out = {}
    out["A_Delta4"] = str(L.alternating_functional(L.DELTA4))
    out["A_Delta4_equals_1/34650"] = L.alternating_functional(L.DELTA4) == F(1, 34650)
    out["A_vanishes_on_P5_monomials"] = all(
        L.alternating_functional({a: F(1)}) == 0 for D in range(6) for a in L.compositions(D, 4))
    rng = random.Random(20261022)
    polys = [random_poly(rng, 7, 12) for _ in range(4)]
    ok = True
    for p in polys:
        Ap = L.alternating_functional(p)
        for s in L.ALL_PERMS4:
            if L.alternating_functional(L.relabel(p, s)) != L.perm_sign(s) * Ap:
                ok = False
    out["equivariance_test_polynomials"] = [L.serialise_poly(p) for p in polys]
    out["equivariance_A(v o sigma)=sgn(sigma)A(v)_all_24"] = ok
    out["Delta4_o_sigma=sgn(sigma)Delta4_all_24"] = all(
        L.relabel(L.DELTA4, s) == L.scale(L.DELTA4, L.perm_sign(s)) for s in L.ALL_PERMS4)
    return out


def gram_checks():
    ks = L.tet_ridges(5)
    G = [[L.edge_energy(a, b) for b in ks] for a in ks]
    G_expected = [[F(230 * (i == j) - 5, 11) if i < 4 and j < 4 else
                   F(720 * (i == j), 11) if i >= 4 and j >= 4 else
                   F(60 * (S[i][j - 4] if i < 4 else S[j][i - 4]), 11)
                   for j in range(7)] for i in range(7)]
    Ginv_expected = [[F(11 * (i == j), 210) if i < 4 and j < 4 else
                      F(11 * 23 * (i == j), 15120) if i >= 4 and j >= 4 else
                      -F(11 * (S[i][j - 4] if i < 4 else S[j][i - 4]), 2520)
                      for j in range(7)] for i in range(7)]
    aug = [row + [F(int(i == j)) for j in range(7)] for i, row in enumerate(G)]
    R, piv = L.rref(aug)
    Ginv = [row[7:] for row in R]
    # one-dimensional reductions
    t = {(1,): F(1)}
    f = L.substitute_univariate(L.vertex_ridge_coeffs(5), t, 1)
    fp = L.deriv(f, 0)
    fp2 = L.mul(fp, fp)
    beta13 = {(0,): F(3), (1,): F(-6), (2,): F(3)}  # 3(1-t)^2

    def integrate01(p):
        return sum((c / (a[0] + 1) for a, c in p.items()), F(0))

    vertex_diag = 3 * integrate01(L.mul(fp2, beta13))
    # E[f'(l_i) f'(l_j)] with joint density 6(1-t-u) on the triangle
    f2 = {(a[0], 0): c for a, c in fp.items()}
    g2 = {(0, a[0]): c for a, c in fp.items()}
    dens = {(0, 0): F(6), (1, 0): F(-6), (0, 1): F(-6)}
    prodp = L.mul(L.mul(f2, g2), dens)
    # int over triangle of t^a u^b = a! b! / (a+b+2)!
    from math import factorial
    tri = sum((c * F(factorial(a) * factorial(b), factorial(a + b + 2))
               for (a, b), c in prodp.items()), F(0))
    vertex_off = -tri
    g = L.substitute_univariate(L.shifted_legendre(5), t, 1)
    gp = L.deriv(g, 0)
    pair_diag = 4 * integrate01(L.mul(L.mul(gp, gp), {(1,): F(6), (2,): F(-6)}))
    return {
        "ridge_functions_m5": [L.serialise_poly(k) for k in ks],
        "ridge_order": ["U_0", "U_1", "U_2", "U_3", "V_01", "V_02", "V_03"],
        "S": S,
        "gram_matrix": L.frac_matrix(G),
        "gram_matches_(1/11)[[230I-5J,60S],[60S^T,720I]]": G == G_expected,
        "gram_inverse": L.frac_matrix(Ginv),
        "gram_nonsingular": piv == list(range(7)),
        "gram_inverse_matches_11[[I/210,-S/2520],[-S^T/2520,23I/15120]]": Ginv == Ginv_expected,
        "one_dimensional": {
            "3*E[f'(l)^2], l~Beta(1,3), f=tP4^(1,1)(2t-1)": str(vertex_diag),
            "-E[f'(l_i)f'(l_j)], Dirichlet(1,1,2)": str(vertex_off),
            "4*int 6s(1-s) g'(s)^2, g=P5(2s-1)": str(pair_diag),
            "agree_with_gram_entries": (vertex_diag == G[0][0] and vertex_off == G[0][1]
                                        and pair_diag == G[4][4]),
        },
    }, ks


def fifty_dof_certificate(ks):
    rows = L.moment_rows(3, 3)
    p4 = [(0,) + a for D in range(5) for a in L.compositions(D, 3)]
    quint = [(0,) + a for a in QUINTICS]
    polys = [{a: F(1)} for a in p4 + quint] + [L.DELTA4]
    M = L.moment_matrix_polys(polys, rows)
    rank = L.rank_exact(M)
    det = L.det_rational(M)
    out = {
        "ordering": "rows: facet moments F_0..F_3 (test monomials of degree 3, lexicographic), "
                    "then interior moments (degree 2); columns: monomials x1^a x2^b x3^c of degree <= 4 "
                    "(x_i = l_i, degree then lexicographic), the 14 quintics in the listed order, Delta4",
        "quintics_x1x2x3": [list(q) for q in QUINTICS],
        "matrix_size": [len(M), len(M[0])],
        "rank": rank,
        "rank_method": "exact_rational_elimination",
        "determinant": str(det),
        "determinant_numerator_factorisation": L.factor_small(det.numerator),
        "determinant_denominator_factorisation": L.factor_small(det.denominator),
        "determinant_equals_-1/(2^174 3^112 5^65 7^46 11)":
            det == F(-1, 2 ** 174 * 3 ** 112 * 5 ** 65 * 7 ** 46 * 11),
    }
    # leading forms of the seven kernel functions in x1, x2, x3 (l_0 = 1 - x1 - x2 - x3)
    subsets = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    lead = []
    for Ssub in subsets:
        s = L.add(*(L.var(i - 1, 3) for i in Ssub))
        form = L.power(s, 5, 3)
        lead.append([str(form.get(e, F(0))) for e in EXCLUDED])
    Fm = [[F(x) for x in r] for r in lead]
    lower = all(Fm[i][j] == 0 for i in range(7) for j in range(i + 1, 7))
    out["excluded_monomials"] = [list(e) for e in EXCLUDED]
    out["leading_form_subsets"] = [list(s) for s in subsets]
    out["leading_form_coefficient_matrix"] = lead
    out["leading_form_matrix_lower_triangular"] = lower
    out["leading_form_matrix_diagonal"] = [lead[i][i] for i in range(7)]
    out["leading_form_matrix_nonsingular"] = lower and all(Fm[i][i] for i in range(7))
    # kernel leading forms: check that each ridge has the stated leading form up to a scalar
    xmap = []
    for k in ks:
        # substitute l_0 = 1 - x1 - x2 - x3 and take the degree-5 part
        sub = {}
        l0 = L.add(L.one(3), *(L.scale(L.var(i, 3), -1) for i in range(3)))
        for a, c in k.items():
            term = L.scale(L.power(l0, a[0], 3), c)
            for i in range(1, 4):
                term = L.mul(term, L.power(L.var(i - 1, 3), a[i], 3))
            sub = L.add(sub, term)
        xmap.append({e: c for e, c in sub.items() if sum(e) == 5})
    # U_0 ~ (x1+x2+x3)^5, U_i ~ x_i^5, V_0j ~ (sum of the two other x)^5
    target = {0: (1, 2, 3), 1: (1,), 2: (2,), 3: (3,), 4: (2, 3), 5: (1, 3), 6: (1, 2)}
    ok = True
    for idx, lf in enumerate(xmap):
        s = L.add(*(L.var(i - 1, 3) for i in target[idx]))
        form = L.power(s, 5, 3)
        e0 = next(iter(form))
        c = lf.get(e0, F(0)) / form[e0]
        ok = ok and c != 0 and lf == L.scale(form, c)
    out["ridge_leading_forms_are_(sum_S x_i)^5"] = ok
    return out, polys


def invariance_check(polys, ks):
    """The 50-dimensional space P4 + projected quintics + Delta4 is S4-invariant."""
    p4 = polys[:35]
    quint = polys[35:49]
    G = [[L.edge_energy(a, b) for b in ks] for a in ks]
    rhs = [[L.edge_energy(k, q) for q in quint] for k in ks]
    aug = [g + r for g, r in zip(G, rhs)]
    R, piv = L.rref(aug)
    cs = [row[7:] for row in R]
    projected = [L.add(q, *(L.scale(k, -cs[j][i]) for j, k in enumerate(ks)))
                 for i, q in enumerate(quint)]
    p4_orth = all(L.edge_energy(p, k) == 0 for p in p4 for k in ks)
    proj_orth = all(L.edge_energy(p, k) == 0 for p in projected for k in ks)
    rows = L.moment_rows(3, 3)
    space = p4 + projected + [L.DELTA4]
    rank50 = L.rank_exact(L.moment_matrix_polys(space, rows))
    inv = True
    for s in L.ALL_PERMS4:
        for p in projected:
            q = L.relabel(p, s)
            if any(L.edge_energy(q, k) for k in ks):
                inv = False
        if L.relabel(L.DELTA4, s) != L.scale(L.DELTA4, L.perm_sign(s)):
            inv = False
    return {
        "projection": "quintics minus their a_B-orthogonal projection onto span(ridges, m=5)",
        "projection_coefficients": L.frac_matrix(cs),
        "P4_a_B_orthogonal_to_ridges": p4_orth,
        "projected_quintics_a_B_orthogonal_to_ridges": proj_orth,
        "projected_space_moment_rank": rank50,
        "argument": "the a_B-orthogonal complement of the ridge span in P5 has dimension 49 "
                    "(Gram nonsingular), contains P4 and the projected quintics, and is S4-invariant "
                    "because a_B and the ridge span are; every relabelled projected quintic is checked "
                    "to remain a_B-orthogonal to the ridges, and Delta4 o sigma = sgn(sigma) Delta4",
        "invariant_under_all_24_vertex_permutations": inv and p4_orth and proj_orth and rank50 == 50,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "certificates" / "tetrahedron_exact.json"))
    args = ap.parse_args()
    data = {"description": "Exact checks for the tetrahedral moment map M_{3,n}",
            "prime": "2^61-1"}
    data["rank_table"] = rank_table()
    data["alternating_functional"] = alternating_checks()
    gram, ks = gram_checks()
    data["gram_m5"] = gram
    cert, polys = fifty_dof_certificate(ks)
    data["fifty_dof_certificate"] = cert
    data["projected_space_invariance"] = invariance_check(polys, ks)
    a = data["alternating_functional"]
    data["all_checks_pass"] = bool(
        all(e["all_pass"] for e in data["rank_table"])
        and a["A_Delta4_equals_1/34650"] and a["A_vanishes_on_P5_monomials"]
        and a["equivariance_A(v o sigma)=sgn(sigma)A(v)_all_24"]
        and gram["gram_matches_(1/11)[[230I-5J,60S],[60S^T,720I]]"]
        and gram["gram_inverse_matches_11[[I/210,-S/2520],[-S^T/2520,23I/15120]]"]
        and gram["one_dimensional"]["agree_with_gram_entries"]
        and cert["rank"] == 50 and cert["leading_form_matrix_nonsingular"]
        and cert["ridge_leading_forms_are_(sum_S x_i)^5"]
        and data["projected_space_invariance"]["invariant_under_all_24_vertex_permutations"])
    L.write_json(data, args.out)
    print("all_checks_pass:", data["all_checks_pass"])


if __name__ == "__main__":
    main()
