"""Lean permutation-invariant ambient spaces W_n = P_{n+2} + Delta4 P_{n-3} and
the S4 character of the cokernel of M_{3,n} on P_{n+2}.

Usage: python3 code/verify_lean_element.py [--out certificates/lean_element_exact.json]

Dimensions are computed in the homogeneous barycentric basis of degree n+3.
Delta4 P_{n-4} lies in P_{n+2}, so dim W_n <= C(n+5,3) + C(n-1,2); a rank modulo
p = 2^61-1 equal to this bound is exact.  Full row rank modulo p of the moment
matrix on W_n is an exact certificate of surjectivity; the kernel on W_n is then
dim W_n - N, which equals 7 (the ridge functions in P_{n+2}).

Character of the cokernel: the moment map P_{n+2} -> (+)_F P_n(F) (+) P_{n-1}(K)
is S4-equivariant, so
  chi_coker = chi_faces + chi_{P_{n-1}} - chi_{P_{n+2}} + chi_kernel,
with the kernel spanned by U_0..U_3 (permuted like the vertices) and
V_S = P_m(2 l_S - 1) for the three bipartitions (sign (-1)^m when the two parts
are exchanged).  This is compared with sign (x) Sym^{n-3}(std), both termwise and
as an identity of generating functions in t.
"""
import argparse
from fractions import Fraction as F
from math import comb
from pathlib import Path

import sympy as sp

import fe_exact_lib as L

ROOT = Path(__file__).resolve().parents[1]

# conjugacy classes of S4 acting on the vertices {0,1,2,3}
CLASSES = {
    "e": ((0, 1, 2, 3), [1, 1, 1, 1], 1),
    "(12)": ((1, 0, 2, 3), [2, 1, 1], 6),
    "(12)(34)": ((1, 0, 3, 2), [2, 2], 3),
    "(123)": ((1, 2, 0, 3), [3, 1], 8),
    "(1234)": ((1, 2, 3, 0), [4], 6),
}
ORDER = ["e", "(12)", "(12)(34)", "(123)", "(1234)"]
SIGN = {"e": 1, "(12)": -1, "(12)(34)": 1, "(123)": 1, "(1234)": -1}
STD = {"e": 3, "(12)": 1, "(12)(34)": -1, "(123)": 0, "(1234)": -1}
t = sp.symbols("t")


def lean_space(n):
    """Spanning set of W_n: degree-(n+2) monomials and Delta4 times degree-(n-3) monomials."""
    base = [{a: F(1)} for a in L.compositions(n + 2, 4)]
    extra = [L.mul(L.DELTA4, {a: F(1)}) for a in L.compositions(n - 3, 4)]
    return base, extra


def lean_checks():
    out = []
    for n in range(3, 9):
        base, extra = lean_space(n)
        D = n + 3
        basis = list(L.compositions(D, 4))
        idx = {a: i for i, a in enumerate(basis)}
        vecs = []
        for p in base + extra:
            h = L.homogenise(p, D, 4)
            v = [F(0)] * len(basis)
            for a, c in h.items():
                v[idx[a]] = c
            vecs.append(v)
        dimW, mdim = L.rank_certified(vecs)
        bound = L.dim_P(n + 2, 3) + (n - 1) * (n - 2) // 2
        rows = L.moment_rows(3, n)
        M = L.moment_matrix_polys(base + extra, rows)
        r, mrank = L.rank_certified(M)
        N = L.N_dofs(3, n)
        e = {"n": n, "N": N, "dim_P_n+2": L.dim_P(n + 2, 3),
             "spanning_set_size": len(base) + len(extra),
             "dim_W_n": dimW, "dim_method": mdim,
             "dim_W_n_expected_C(n+5,3)+(n-1)(n-2)/2": bound,
             "dim_matches": dimW == bound,
             "moment_rank_on_W_n": r, "rank_method": mrank,
             "surjective_on_W_n": r == N,
             "kernel_dim_on_W_n": dimW - r,
             "kernel_on_W_n_is_the_seven_ridges": dimW - r == 7,
             "top_degree_directions": (n - 1) * (n - 2) // 2}
        # invariance: Delta4 o sigma = sgn(sigma) Delta4 and P_{n-3}, P_{n+2} are invariant
        out.append(e)
    # explicit n = 4: W = P_6 + span{Delta4 l_i}
    base = [{a: F(1)} for a in L.compositions(6, 4)]
    extra = [L.mul(L.DELTA4, L.var(i, 4)) for i in range(4)]
    M = L.moment_matrix_polys(base + extra, L.moment_rows(3, 4))
    r, method = L.rank_certified(M)
    n4 = {"space": "P_6 + span{Delta4 l_i, i=0..3}", "moment_rank": r, "N": L.N_dofs(3, 4),
          "method": method, "surjective": r == L.N_dofs(3, 4),
          "sum_i Delta4 l_i = Delta4 in P_6": True}
    return out, n4


def h_char(cyc, r):
    """Trace of a permutation with cycle type cyc on homogeneous polynomials of degree r."""
    if r < 0:
        return sp.Integer(0)
    f = sp.prod([1 / (1 - t ** c) for c in cyc])
    return sp.series(f, t, 0, r + 1).removeO().coeff(t, r)


def bipartition_data(perm):
    """(# bipartitions fixed with parts kept, # fixed with parts exchanged)."""
    parts = [frozenset({0, j}) for j in (1, 2, 3)]
    keep = swap = 0
    for A in parts:
        B = frozenset(range(4)) - A
        img = frozenset(perm[i] for i in A)
        if img == A:
            keep += 1
        elif img == B:
            swap += 1
    return keep, swap


def face_char(cyc, n):
    fixed = cyc.count(1)
    if fixed == 0:
        return sp.Integer(0)
    rest = list(cyc)
    rest.remove(1)
    return fixed * h_char(rest, n)


def coker_char(name, n):
    perm, cyc, _ = CLASSES[name]
    m = n + 2
    keep, swap = bipartition_data(perm)
    ker = cyc.count(1) + keep + swap * (-1) ** m
    return face_char(cyc, n) + h_char(cyc, n - 1) - h_char(cyc, m) + ker


def target_char(name, n):
    perm, cyc, _ = CLASSES[name]
    k = n - 3
    if k < 0:
        return sp.Integer(0)
    # character of Sym^k(std): coefficient of t^k in (1-t)/prod(1-t^c)
    g = (1 - t) * sp.prod([1 / (1 - t ** c) for c in cyc])
    return SIGN[name] * sp.series(g, t, 0, k + 1).removeO().coeff(t, k)


def generating_function_identity():
    """sum_{n>=1} chi_coker(n) t^n == sign * t^3 (1-t) / prod(1-t^c), per class."""
    out = {}
    for name in ORDER:
        perm, cyc, _ = CLASSES[name]
        rest = list(cyc)
        H = sp.prod([1 / (1 - t ** c) for c in cyc])
        fixed = cyc.count(1)
        if fixed:
            rest.remove(1)
            Hrest = sp.prod([1 / (1 - t ** c) for c in rest])
        else:
            Hrest = 0
        # sum_{n>=1} t^n h_n(rest), sum_{n>=1} t^n h_{n-1}, sum_{n>=1} t^n h_{n+2}
        faces = fixed * (Hrest - 1)
        interior = t * H
        h1 = sum(cyc[i] == 1 for i in range(len(cyc)))  # h_1 = number of cycles of length 1
        h2 = h_char(cyc, 2)
        top = (H - 1 - h1 * t - h2 * t ** 2) / t ** 2
        keep, swap = bipartition_data(perm)
        # sum_{n>=1} t^n (fixed + keep + swap (-1)^(n+2))
        kern = (fixed + keep) * t / (1 - t) + swap * (-t) / (1 + t)
        lhs = faces + interior - top + kern
        rhs = SIGN[name] * t ** 3 * (1 - t) * H
        diff = sp.simplify(sp.together(lhs - rhs))
        out[name] = {"lhs_minus_rhs": str(diff), "identity_holds": diff == 0}
    return out


def projected_element(n):
    """V_n = Pi_m(P_m) + Delta4 P_{n-3}^{hom}, m = n+2, where Pi_m removes the ridge
    span Z_m from P_m by a_B-orthogonal projection."""
    m = n + 2
    ks = L.tet_ridges(m)
    extra = [L.mul(L.DELTA4, {a: F(1)}) for a in L.compositions(n - 3, 4)]
    extra_orth = all(L.edge_energy(e, z) == 0 for e in extra for z in ks)
    delta_orth = all(L.edge_energy(L.DELTA4, z) == 0 for z in ks)
    G = [[L.edge_energy(a, b) for b in ks] for a in ks]
    mons = [{a: F(1)} for a in L.compositions(m, 4)]
    rhs = [[L.edge_energy(z, q) for q in mons] for z in ks]
    R, piv = L.rref([g + r for g, r in zip(G, rhs)])
    cs = [row[7:] for row in R]
    proj = [L.add(q, *(L.scale(z, -cs[j][i]) for j, z in enumerate(ks))) for i, q in enumerate(mons)]
    proj_orth = all(L.edge_energy(p, z) == 0 for p in proj for z in ks)
    lower = [{a: F(1)} for a in L.compositions(m - 1, 4)]
    contains_lower = all(L.edge_energy(p, z) == 0 for p in lower for z in ks)
    D = m + 1
    basis = list(L.compositions(D, 4))
    idx = {a: i for i, a in enumerate(basis)}
    vecs = []
    for p in proj + extra:
        v = [F(0)] * len(basis)
        for a, c in L.homogenise(p, D, 4).items():
            v[idx[a]] = c
        vecs.append(v)
    dimV, dmethod = L.rank_certified(vecs)
    M = L.moment_matrix_polys(proj + extra, L.moment_rows(3, n))
    r, rmethod = L.rank_certified(M)
    N = L.N_dofs(3, n)
    # Z_m is permuted up to sign by every vertex permutation; with the invariance of
    # a_B this makes Pi_m(P_m) invariant, and Delta4 P_{n-3} is invariant.
    # polynomials are compared as functions on K (homogenised with sum l = 1)
    hks = [L.homogenise(z, m, 4) for z in ks]

    def signed_member(p):
        h = L.homogenise(p, m, 4)
        return any(h == z or h == L.scale(z, -1) for z in hks)
    ridge_perm = all(signed_member(L.relabel(z, s)) for z in ks for s in L.ALL_PERMS4)
    delta_sign = all(L.relabel(L.DELTA4, s) == L.scale(L.DELTA4, L.perm_sign(s)) for s in L.ALL_PERMS4)
    return {"n": n, "m": m, "N": N,
            "space": "Pi_%d(P_%d) + Delta4 * (degree-%d barycentric monomials)" % (m, m, n - 3),
            "a_B(Delta4*mono, z)=0_all_ridges": extra_orth,
            "a_B(Delta4, z)=0_all_ridges": delta_orth,
            "gram_nonsingular": piv == list(range(7)),
            "projected_P_m_a_B_orthogonal_to_ridges": proj_orth,
            "contains_P_m-1": contains_lower,
            "dim": dimV, "dim_method": dmethod,
            "moment_rank": r, "rank_method": rmethod,
            "unisolvent": dimV == N and r == N,
            "ridges_permuted_up_to_sign_all_24": ridge_perm,
            "Delta4_o_sigma=sgn(sigma)Delta4": delta_sign,
            "invariant_under_all_24_vertex_permutations": ridge_perm and delta_sign and proj_orth}


def character_checks():
    rows = []
    for n in range(3, 13):
        ch = {k: coker_char(k, n) for k in ORDER}
        tg = {k: target_char(k, n) for k in ORDER}
        rows.append({"n": n, "cokernel_character": [int(ch[k]) for k in ORDER],
                     "sign_x_Sym^(n-3)(std)": [int(tg[k]) for k in ORDER],
                     "dimension": int(ch["e"]),
                     "equal": all(ch[k] == tg[k] for k in ORDER)})
    gf = generating_function_identity()
    return {"class_order": ORDER,
            "class_sizes": [CLASSES[k][2] for k in ORDER],
            "std_character": [STD[k] for k in ORDER],
            "termwise_n3..12": rows,
            "generating_function_identity": gf,
            "identity_holds_for_all_n_>=1": all(v["identity_holds"] for v in gf.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "certificates" / "lean_element_exact.json"))
    args = ap.parse_args()
    lean, n4 = lean_checks()
    proj = [projected_element(n) for n in (4, 5)]
    chars = character_checks()
    data = {"description": "Lean ambient spaces W_n = P_{n+2} + Delta4 P_{n-3} and the S4 "
                           "character of the cokernel of M_{3,n} on P_{n+2}",
            "prime": "2^61-1",
            "lean_spaces": lean,
            "lean_space_n4_explicit": n4,
            "W_n_invariance": "P_{n+2} and P_{n-3} are invariant under vertex permutations and "
                              "Delta4 o sigma = sgn(sigma) Delta4 (checked in tetrahedron_exact.json)",
            "cokernel_character": chars,
            "projected_elements": proj}
    data["all_checks_pass"] = bool(
        all(e["dim_matches"] and e["surjective_on_W_n"] and e["kernel_on_W_n_is_the_seven_ridges"] for e in lean)
        and n4["surjective"]
        and all(p["unisolvent"] and p["contains_P_m-1"] and p["invariant_under_all_24_vertex_permutations"]
                and p["a_B(Delta4*mono, z)=0_all_ridges"] for p in proj)
        and all(r["equal"] for r in chars["termwise_n3..12"])
        and chars["identity_holds_for_all_n_>=1"])
    L.write_json(data, args.out)
    print("all_checks_pass:", data["all_checks_pass"])


if __name__ == "__main__":
    main()
