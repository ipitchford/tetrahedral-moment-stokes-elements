#!/usr/bin/env python3
"""Numerical Stokes convergence test for the fourth-order nonconforming
tetrahedral element with degrees of freedom M_3 (n = 3).

Local space (barycentric description, identical on every affine tetrahedron):
    V = { p in P_5 : a_B(p, z) = 0 for all z in Z_5 }  (+)  span{Delta4},
    Delta4 = prod_{i<j} (l_j - l_i),
    a_B(u, v) = (1/|K|) int_K sum_{i<j} D_ij u D_ij v,  D_ij = d/dl_i - d/dl_j,
    Z_5 = span{ l_i P_4^{(1,1)}(2 l_i - 1), P_5(2(l_0 + l_j) - 1) }.
Degrees of freedom: normalised face moments against P_3(F) (10 per face) and
normalised interior moments against P_2(K) (10).

Face test functions are an L2(F)-orthonormal basis of P_3 written in the
three face barycentrics ordered by increasing global vertex index.  Every
tetrahedron stores its vertices in increasing global order, so local face i
(opposite local vertex i) lists its vertices in global order and both
neighbouring elements use identical face tests.  Since all functionals are
normalised and the space is barycentric, the moment matrix is the same on
every affine element; it is assembled and inverted on the reference element.

Discretisation of -nu Lap u + grad p = f, div u = 0 on (0,1)^3 with Kuhn
meshes: nonconforming velocity (shared face moments, zero boundary face
moments), discontinuous P_3 pressure, mean-zero pressure by a Lagrange
multiplier.  All results are floating-point computations.
"""
import argparse
import hashlib
import itertools
import json
import math

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy as sy
from scipy.special import eval_jacobi, eval_legendre, roots_jacobi, roots_legendre

# ---------------------------------------------------------------------------
# quadrature (collapsed Gauss-Jacobi / Stroud), normalised weights (sum 1)
# ---------------------------------------------------------------------------


def tet_rule(n):
    """Collapsed rule on {x >= 0, x1+x2+x3 <= 1}, exact to degree 2n-1."""
    a, wa = roots_jacobi(n, 2, 0)
    b, wb = roots_jacobi(n, 1, 0)
    c, wc = roots_legendre(n)
    t1, t2, t3 = (1 + a) / 2, (1 + b) / 2, (1 + c) / 2
    w1, w2, w3 = wa / 8, wb / 4, wc / 2
    T1, T2, T3 = np.meshgrid(t1, t2, t3, indexing="ij")
    W = np.einsum("i,j,k->ijk", w1, w2, w3)
    x1 = T1
    x2 = (1 - T1) * T2
    x3 = (1 - T1) * (1 - T2) * T3
    pts = np.stack([x1.ravel(), x2.ravel(), x3.ravel()], axis=1)
    w = 6.0 * W.ravel()
    return pts, w


def tri_rule(n):
    """Collapsed rule on {s >= 0, s1+s2 <= 1}, exact to degree 2n-1."""
    a, wa = roots_jacobi(n, 1, 0)
    c, wc = roots_legendre(n)
    t1, t2 = (1 + a) / 2, (1 + c) / 2
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")
    W = np.einsum("i,j->ij", wa / 4, wc / 2)
    pts = np.stack([T1.ravel(), ((1 - T1) * T2).ravel()], axis=1)
    return pts, 2.0 * W.ravel()


# ---------------------------------------------------------------------------
# polynomial representation
# ---------------------------------------------------------------------------


def monomials(dim, deg):
    out = []
    for t in range(deg + 1):
        for e in itertools.product(range(t + 1), repeat=dim):
            if sum(e) == t:
                out.append(tuple(e))
    # within a degree, sort lexicographically descending for a fixed order
    out.sort(key=lambda e: (sum(e), tuple(-k for k in e)))
    return out


def mono_eval(pts, exps):
    E = np.ones((pts.shape[0], len(exps)))
    for m, e in enumerate(exps):
        for k, ek in enumerate(e):
            if ek:
                E[:, m] *= pts[:, k] ** ek
    return E


def mono_grad(pts, exps):
    d = pts.shape[1]
    G = np.zeros((d, pts.shape[0], len(exps)))
    for m, e in enumerate(exps):
        for k in range(d):
            if e[k] == 0:
                continue
            v = e[k] * np.ones(pts.shape[0])
            for j, ej in enumerate(e):
                p = ej - 1 if j == k else ej
                if p:
                    v = v * pts[:, j] ** p
            G[k, :, m] = v
    return G


class OrthoBasis:
    """L2-orthonormal (normalised measure) basis of P_deg on a simplex,
    obtained by weighted QR of the degree-ordered monomial Vandermonde;
    the first dim(P_k) functions span P_k."""

    def __init__(self, dim, deg, pts, w):
        self.exps = monomials(dim, deg)
        E = mono_eval(pts, self.exps)
        _, R = np.linalg.qr(np.sqrt(w)[:, None] * E)
        sgn = np.sign(np.diag(R))
        R = R * sgn[:, None]
        self.C = sla.solve_triangular(R, np.eye(R.shape[0]))  # mono coeffs
        self.n = len(self.exps)

    def val(self, pts):
        return mono_eval(pts, self.exps) @ self.C

    def grad(self, pts):
        return np.einsum("kpm,mn->kpn", mono_grad(pts, self.exps), self.C)


def dimP(dim, k):
    return math.comb(k + dim, dim)


# ---------------------------------------------------------------------------
# the local element
# ---------------------------------------------------------------------------

FACE_VERTS = [tuple(j for j in range(4) if j != i) for i in range(4)]


def bary(x):
    return np.column_stack([1 - x.sum(axis=1), x])


def face_points(i, s):
    """Tetrahedron coordinates of points on face i; face barycentrics
    (mu_a, mu_b, mu_c) = (1-s1-s2, s1, s2) for the face vertices in
    increasing (local = global) order."""
    a, b, c = FACE_VERTS[i]
    L = np.zeros((s.shape[0], 4))
    L[:, a] = 1 - s[:, 0] - s[:, 1]
    L[:, b] = s[:, 0]
    L[:, c] = s[:, 1]
    return L[:, 1:]


def ridge_values(x):
    L = bary(x)
    cols = [L[:, i] * eval_jacobi(4, 1, 1, 2 * L[:, i] - 1) for i in range(4)]
    cols += [eval_legendre(5, 2 * (L[:, 0] + L[:, j]) - 1) for j in (1, 2, 3)]
    return np.column_stack(cols)


def delta4_values(x):
    L = bary(x)
    v = np.ones(x.shape[0])
    for i in range(4):
        for j in range(i + 1, 4):
            v = v * (L[:, j] - L[:, i])
    return v


class Element:
    def __init__(self, n_vol=8, n_face=5):
        self.xq, self.wq = tet_rule(n_vol)          # exact to degree 15
        self.sq, self.ws = tri_rule(n_face)         # exact to degree 9
        self.P6 = OrthoBasis(3, 6, self.xq, self.wq)
        self.Pf = OrthoBasis(2, 3, *tri_rule(4))    # P_3(F), orthonormal
        self.Phi = self.P6.val(self.xq)
        self.G = self.P6.grad(self.xq)
        self.n5, self.n4 = dimP(3, 5), dimP(3, 4)
        self.n3, self.n2 = dimP(3, 3), dimP(3, 2)
        wq = self.wq
        self.ortho_err = np.abs((self.Phi * wq[:, None]).T @ self.Phi - np.eye(84)).max()

        # a_B Gram matrix on P_6
        D = []
        for i in range(4):
            for j in range(i + 1, 4):
                # in coordinates x = (l1, l2, l3): D_0j = -d/dx_j, D_ij = d/dx_i - d/dx_j
                D.append(-self.G[j - 1] if i == 0 else self.G[i - 1] - self.G[j - 1])
        self.AB = sum((d * wq[:, None]).T @ d for d in D)

        # ridges and Delta4 as coefficient vectors in the orthonormal P_6 basis
        self.Z = self.project(ridge_values(self.xq))
        self.d4 = self.project(delta4_values(self.xq)[:, None])[:, 0]

        # moment functionals acting on P_6 coefficient vectors (50 x 84)
        self.Qf = self.Pf.val(self.sq)                  # face tests
        rows = []
        for i in range(4):
            Pfi = self.P6.val(face_points(i, self.sq))
            rows.append((self.Qf * self.ws[:, None]).T @ Pfi)
        rows.append(np.eye(84)[: self.n2])              # interior P_2 tests
        self.Mfun = np.vstack(rows)

        # V = a_B-orthogonal complement of Z_5 in P_5, plus Delta4
        Gc = self.Z.T @ self.AB[:, : self.n5]
        _, s, Vt = np.linalg.svd(Gc)
        null = Vt[7:].T
        W = np.zeros((84, null.shape[1]))
        W[: self.n5] = null
        Vraw = np.column_stack([W, self.d4])
        Q, _ = np.linalg.qr(Vraw)
        self.Vb = Q                                     # L2-orthonormal basis of V
        self.Mloc = self.Mfun @ self.Vb
        self.Cn = self.Vb @ np.linalg.inv(self.Mloc)    # nodal (dual) basis
        self.Phin = self.Phi @ self.Cn
        self.Gn = np.einsum("kpm,mn->kpn", self.G, self.Cn)

    def project(self, vals):
        return (self.Phi * self.wq[:, None]).T @ vals

    def sanity(self):
        M5 = self.Mfun[:, : self.n5]
        sv = np.linalg.svd(M5, compute_uv=False)
        sv_aug = np.linalg.svd(np.column_stack([self.Vb, np.eye(84)[:, : self.n4]]),
                               compute_uv=False)
        return {
            "orthonormality_error_P6": self.ortho_err,
            "rank_M3_on_P5": int((sv > 1e-10 * sv[0]).sum()),
            "ridges_in_P5_residual": np.abs(self.Z[self.n5:]).max(),
            "ridge_moments_max": np.abs(self.Mfun @ self.Z).max(),
            "dim_V": self.Vb.shape[1],
            "rank_of_V_plus_P4": int((sv_aug > 1e-10 * sv_aug[0]).sum()),
            "aB_Delta4_P5_max": np.abs(self.AB[: self.n5] @ self.d4).max(),
            "M3_on_V_smallest_singular_value": np.linalg.svd(self.Mloc, compute_uv=False)[-1],
        }


# ---------------------------------------------------------------------------
# conditioning and interpolation checks
# ---------------------------------------------------------------------------

QUINTIC_EXPS = [(0, 1, 4), (0, 2, 3), (0, 3, 2), (1, 0, 4), (1, 1, 3), (1, 2, 2),
                (1, 3, 1), (1, 4, 0), (2, 0, 3), (2, 1, 2), (2, 2, 1), (2, 3, 0),
                (3, 0, 2), (3, 2, 0)]


def naive_moment_matrix(el, Jmap, v0):
    """Naive basis: monomials of P_4 and the 14 listed quintics in Cartesian
    coordinates X = v0 + J x (quintics projected a_B-orthogonally to Z_5),
    plus Delta4; monomial face tests (face barycentrics) and Cartesian
    monomial interior tests."""
    X = v0 + el.xq @ Jmap.T
    cols = [mono_eval(X, monomials(3, 4))]
    q = el.project(mono_eval(X, QUINTIC_EXPS))
    Zab = el.Z.T @ el.AB @ el.Z
    q = q - el.Z @ np.linalg.solve(Zab, el.Z.T @ el.AB @ q)
    basis = np.column_stack([el.project(cols[0]), q, el.d4])
    fe = monomials(2, 3)
    rows = []
    for i in range(4):
        Pfi = el.P6.val(face_points(i, el.sq))
        rows.append((mono_eval(el.sq, fe) * el.ws[:, None]).T @ Pfi)
    rows.append((mono_eval(X, monomials(3, 2)) * el.wq[:, None]).T @ el.Phi)
    M = np.vstack(rows) @ basis
    # residual of basis outside V (should vanish)
    proj = el.Vb @ (el.Vb.T @ basis)
    return M, np.abs(basis - proj).max()


def reference_stiffness(el):
    K = np.einsum("kpi,p,kpj->ij", el.Gn, el.wq, el.Gn) / 6.0
    ev = np.linalg.eigvalsh(0.5 * (K + K.T))
    return ev[-1] / ev[1], ev[0] / ev[-1]


def interpolate(el, f, v0, h):
    """Moment interpolant on the element v0 + h * reference; returns
    nodal coefficients."""
    dofs = []
    for i in range(4):
        X = v0 + h * face_points(i, el.sq)
        dofs.append((el.Qf * el.ws[:, None]).T @ f(X))
    X = v0 + h * el.xq
    dofs.append((el.Phi[:, : el.n2] * el.wq[:, None]).T @ f(X))
    return np.concatenate(dofs)


def interpolation_checks(el, rng):
    c = rng.standard_normal(dimP(3, 4))
    ex4 = monomials(3, 4)
    v0 = np.array([0.1, -0.2, 0.3])
    quart = lambda X: mono_eval(X, ex4) @ c
    d = interpolate(el, quart, v0, 1.0)
    err4 = np.abs(el.Phin @ d - quart(v0 + el.xq)).max() / np.abs(c).max()

    f = lambda X: np.exp(X[:, 0] + 0.5 * X[:, 1] - 0.7 * X[:, 2]) * np.sin(2 * X[:, 0] + X[:, 2])

    def gradf(X):
        e = np.exp(X[:, 0] + 0.5 * X[:, 1] - 0.7 * X[:, 2])
        s, co = np.sin(2 * X[:, 0] + X[:, 2]), np.cos(2 * X[:, 0] + X[:, 2])
        return np.stack([e * s + 2 * e * co, 0.5 * e * s, -0.7 * e * s + e * co])

    hs = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
    rows = []
    for h in hs:
        d = interpolate(el, f, v0, h)
        X = v0 + h * el.xq
        e0 = el.Phin @ d - f(X)
        e1 = np.einsum("kpn,n->kp", el.Gn, d) / h - gradf(X)
        rows.append((h, math.sqrt(el.wq @ e0 ** 2), math.sqrt(el.wq @ (e1 ** 2).sum(0))))
    return err4, rows


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


def kuhn_mesh(N):
    g = np.linspace(0.0, 1.0, N + 1)
    idx = lambda i, j, k: i + (N + 1) * (j + (N + 1) * k)
    I, J, K = np.meshgrid(g, g, g, indexing="ij")
    verts = np.zeros(((N + 1) ** 3, 3))
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                verts[idx(i, j, k)] = (g[i], g[j], g[k])
    tets = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for perm in itertools.permutations(range(3)):
                    p = [i, j, k]
                    t = [idx(*p)]
                    for ax in perm:
                        p[ax] += 1
                        t.append(idx(*p))
                    tets.append(sorted(t))
    return verts, np.array(tets, dtype=np.int64)


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------


def exact_solution():
    x, y, z = sy.symbols("x y z")
    phi = sy.sin(sy.pi * x) ** 2 * sy.sin(sy.pi * y) ** 2 * sy.sin(sy.pi * z) ** 2
    u = [sy.diff(phi, y) - sy.diff(phi, z), sy.diff(phi, z) - sy.diff(phi, x),
         sy.diff(phi, x) - sy.diff(phi, y)]
    p = sy.cos(sy.pi * x) * sy.cos(sy.pi * y) * sy.cos(sy.pi * z)
    X = (x, y, z)
    f = [-sum(sy.diff(u[c], v, 2) for v in X) + sy.diff(p, X[c]) for c in range(3)]
    gu = [[sy.diff(u[c], v) for v in X] for c in range(3)]
    div = sy.simplify(sum(sy.diff(u[c], X[c]) for c in range(3)))
    assert div == 0

    def lam(expr):
        fn = sy.lambdify(X, expr, "numpy")
        return lambda P: np.broadcast_to(fn(P[..., 0], P[..., 1], P[..., 2]), P.shape[:-1]) * 1.0

    return ([lam(e) for e in u], [[lam(e) for e in r] for r in gu], lam(p), [lam(e) for e in f])


# ---------------------------------------------------------------------------
# Stokes solver
# ---------------------------------------------------------------------------


class StokesProblem:
    def __init__(self, el, N):
        self.el, self.N = el, N
        verts, tets = kuhn_mesh(N)
        self.verts, self.tets = verts, tets
        ne = len(tets)
        self.ne = ne
        V0 = verts[tets[:, 0]]
        J = np.stack([verts[tets[:, k]] - V0 for k in (1, 2, 3)], axis=2)  # columns
        self.V0, self.J = V0, J
        detJ = np.linalg.det(J)
        self.vol = np.abs(detJ) / 6.0
        self.Jinv = np.linalg.inv(J)
        edges = [verts[tets[:, a]] - verts[tets[:, b]] for a in range(4) for b in range(a)]
        self.h = max(np.linalg.norm(e, axis=1).max() for e in edges)

        # faces
        fdict = {}
        lf = np.zeros((ne, 4), dtype=np.int64)
        count = []
        for e in range(ne):
            for i in range(4):
                key = tuple(tets[e, list(FACE_VERTS[i])])
                if key not in fdict:
                    fdict[key] = len(fdict)
                    count.append(0)
                lf[e, i] = fdict[key]
                count[lf[e, i]] += 1
        count = np.array(count)
        interior = count == 2
        fnum = -np.ones(len(count), dtype=np.int64)
        fnum[interior] = np.arange(interior.sum())
        nfi = int(interior.sum())
        self.nfaces_interior, self.nfaces_boundary = nfi, int((~interior).sum())
        L = -np.ones((ne, 50), dtype=np.int64)
        for i in range(4):
            fi = fnum[lf[:, i]]
            for k in range(10):
                L[:, 10 * i + k] = np.where(fi >= 0, 10 * fi + k, -1)
        L[:, 40:] = 10 * nfi + 10 * np.arange(ne)[:, None] + np.arange(10)[None, :]
        self.L = L
        self.n1 = 10 * (nfi + ne)
        self.npr = 20 * ne

        # reference tensors
        wq = el.wq
        S = np.einsum("api,p,bpj->abij", el.Gn, wq, el.Gn)
        T = np.einsum("pq,p,apj->aqj", el.Phi[:, :20], wq, el.Gn)
        Cm = np.einsum("eak,ebk->eab", self.Jinv, self.Jinv)
        Ke = self.vol[:, None, None] * np.einsum("eab,abij->eij", Cm, S)
        R = np.broadcast_to(L[:, :, None], Ke.shape)
        Cc = np.broadcast_to(L[:, None, :], Ke.shape)
        m = (R >= 0) & (Cc >= 0)
        self.A1 = sp.csr_matrix((Ke[m], (R[m], Cc[m])), shape=(self.n1, self.n1))
        P = 20 * np.arange(ne)[:, None] + np.arange(20)[None, :]
        self.B = []
        for c in range(3):
            Be = -self.vol[:, None, None] * np.einsum("ea,aqj->eqj", self.Jinv[:, :, c], T)
            R = np.broadcast_to(P[:, :, None], Be.shape)
            Cc = np.broadcast_to(L[:, None, :], Be.shape)
            m = Cc >= 0
            self.B.append(sp.csr_matrix((Be[m], (R[m], Cc[m])), shape=(self.npr, self.n1)))

    def phys_points(self):
        return self.V0[:, None, :] + np.einsum("eij,pj->epi", self.J, self.el.xq)

    def solve(self, sol, nu=1.0):
        el = self.el
        u_ex, gu_ex, p_ex, f_ex = sol
        X = self.phys_points()
        n1, npr, ne = self.n1, self.npr, self.ne
        Lm = self.L >= 0
        F = np.zeros(3 * n1)
        for c in range(3):
            fv = f_ex[c](X)                                   # (ne, nq)
            Fe = self.vol[:, None] * np.einsum("ep,p,pj->ej", fv, el.wq, el.Phin)
            np.add.at(F, c * n1 + self.L[Lm], Fe[Lm])
        A = sp.block_diag([nu * self.A1] * 3, format="csr")
        Bf = sp.hstack(self.B, format="csr")
        mrow = sp.csr_matrix((self.vol, (np.zeros(ne, dtype=np.int64), 20 * np.arange(ne))),
                             shape=(1, npr))
        K = sp.bmat([[A, Bf.T, None], [Bf, None, mrow.T], [None, mrow, None]], format="csc")
        rhs = np.concatenate([F, np.zeros(npr + 1)])
        x = spla.splu(K).solve(rhs)
        uh = x[: 3 * n1].reshape(3, n1)
        ph = x[3 * n1: 3 * n1 + npr].reshape(ne, 20)

        eH1 = eL2 = eP = 0.0
        for c in range(3):
            loc = np.where(Lm, uh[c][np.where(Lm, self.L, 0)], 0.0)    # (ne,50)
            val = loc @ el.Phin.T                                      # (ne,nq)
            gref = np.einsum("apj,ej->eap", el.Gn, loc)
            gph = np.einsum("eab,eap->ebp", self.Jinv, gref)
            eL2 += np.einsum("e,p,ep->", self.vol, el.wq, (val - u_ex[c](X)) ** 2)
            for b in range(3):
                eH1 += np.einsum("e,p,ep->", self.vol, el.wq, (gph[:, b] - gu_ex[c][b](X)) ** 2)
        pv = ph @ el.Phi[:, :20].T
        eP = np.einsum("e,p,ep->", self.vol, el.wq, (pv - p_ex(X)) ** 2)
        return math.sqrt(eH1), math.sqrt(eL2), math.sqrt(eP), x[-1]

    def inf_sup(self, chunk=1000):
        lu = spla.splu(self.A1.tocsc())
        S = np.zeros((self.npr, self.npr))
        for c in range(3):
            BT = self.B[c].T.tocsc()
            for s in range(0, self.npr, chunk):
                Y = lu.solve(BT[:, s: s + chunk].toarray())
                S[:, s: s + chunk] += self.B[c] @ Y
        # pressure mass matrix is diag(|K|) in the orthonormal P_3 basis
        dm = np.repeat(self.vol, 20)
        Ssc = S / np.sqrt(dm)[:, None] / np.sqrt(dm)[None, :]
        ev = sla.eigvalsh(0.5 * (Ssc + Ssc.T))
        return ev


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def r4(x):
    return float(f"{float(x):.4g}")


def bound(x):
    x = abs(float(x))
    return "0" if x == 0 else f"<=1e{int(math.ceil(math.log10(x)))}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-N", type=int, default=5)
    ap.add_argument("--max-N-infsup", type=int, default=4)
    args = ap.parse_args()

    el = Element()
    out = {"label": "numerical (floating point); not a proof",
           "parameters": {
               "domain": "unit cube, N^3 subcubes, 6 Kuhn tetrahedra each",
               "N_values": list(range(1, args.max_N + 1)),
               "inf_sup_N_values": list(range(1, min(args.max_N, args.max_N_infsup) + 1)),
               "nu": 1,
               "velocity": "3 components in the nonconforming M_3 space, boundary face moments zero",
               "pressure": "discontinuous P_3, mean zero via Lagrange multiplier",
               "exact_solution": "u = curl(phi,phi,phi), phi = sin^2(pi x) sin^2(pi y) sin^2(pi z); "
                                 "p = cos(pi x) cos(pi y) cos(pi z)",
               "volume_quadrature": "collapsed Gauss-Jacobi, 8 points per direction (degree 15)",
               "face_quadrature_basis": "collapsed Gauss-Jacobi, 5 points per direction (degree 9)",
               "local_basis": "L2-orthonormal basis of V inside an orthonormal P_6 basis; "
                              "orthonormal P_3(F) and P_2(K) tests",
               "face_test_orientation": "face barycentrics ordered by increasing global vertex index",
           }}
    s = el.sanity()
    out["local_sanity"] = {k: (v if isinstance(v, int) else bound(v)) for k, v in s.items()}
    out["local_sanity"]["M3_on_V_smallest_singular_value"] = r4(s["M3_on_V_smallest_singular_value"])

    # conditioning
    Jref, v0ref = np.eye(3), np.zeros(3)
    Jreg = np.array([[1.0, 0.5, 0.5], [0.0, math.sqrt(3) / 2, math.sqrt(3) / 6],
                     [0.0, 0.0, math.sqrt(2.0 / 3.0)]])
    cond = {}
    for name, Jm in (("reference_tet", Jref), ("regular_tet_edge1", Jreg)):
        Mn, res = naive_moment_matrix(el, Jm, v0ref)
        cond[name] = {"naive_basis_monomial_tests": r4(np.linalg.cond(Mn)),
                      "naive_basis_outside_V_residual": bound(res),
                      "orthonormal_basis_orthonormal_tests": r4(np.linalg.cond(el.Mloc))}
    kc, k0 = reference_stiffness(el)
    cond["reference_stiffness_nodal_basis_nonconstant_subspace"] = r4(kc)
    cond["note"] = ("the orthonormal construction is barycentric and the functionals are normalised, "
                    "so its moment matrix is identical on every affine tetrahedron; the naive basis "
                    "uses Cartesian monomials of the given element (origin at vertex 0)")
    out["conditioning"] = cond

    rng = np.random.default_rng(20261022)
    err4, irows = interpolation_checks(el, rng)
    interp = {"random_quartic_max_relative_error": bound(err4), "smooth_function_shrinking_element": []}
    for k, (h, e0, e1) in enumerate(irows):
        row = {"h_factor": r4(h), "L2_error_normalised": r4(e0), "H1_seminorm_error_normalised": r4(e1)}
        if k:
            hp, a0, a1 = irows[k - 1]
            row["rate_L2"] = r4(math.log(a0 / e0) / math.log(hp / h))
            row["rate_H1"] = r4(math.log(a1 / e1) / math.log(hp / h))
        interp["smooth_function_shrinking_element"].append(row)
    out["interpolation"] = interp

    sol = exact_solution()
    conv, infsup = [], []
    prev = None
    for N in range(1, args.max_N + 1):
        prob = StokesProblem(el, N)
        eH1, eL2, eP, lam = prob.solve(sol)
        row = {"N": N, "h": r4(prob.h), "tets": prob.ne, "velocity_dofs": 3 * prob.n1,
               "pressure_dofs": prob.npr, "interior_faces": prob.nfaces_interior,
               "velocity_H1_broken_seminorm_error": r4(eH1), "velocity_L2_error": r4(eL2),
               "pressure_L2_error": r4(eP)}
        if prev is not None:
            lh = math.log(prev[0] / prob.h)
            row["rate_velocity_H1"] = r4(math.log(prev[1] / eH1) / lh)
            row["rate_velocity_L2"] = r4(math.log(prev[2] / eL2) / lh)
            row["rate_pressure_L2"] = r4(math.log(prev[3] / eP) / lh)
        conv.append(row)
        prev = (prob.h, eH1, eL2, eP)
        print(json.dumps(row), flush=True)
        if N <= args.max_N_infsup:
            ev = prob.inf_sup()
            nz = int((ev < 1e-8 * ev[-1]).sum())
            infsup.append({"N": N, "near_zero_eigenvalues": nz,
                           "beta_h": r4(math.sqrt(ev[nz])),
                           "largest_eigenvalue": r4(ev[-1])})
            print(json.dumps(infsup[-1]), flush=True)
    out["convergence"] = conv
    out["inf_sup"] = {"definition": "beta_h^2 = smallest eigenvalue of B A^{-1} B^T q = beta^2 M q "
                                    "on mean-zero pressures (the single near-zero eigenvalue is the "
                                    "constant pressure)", "values": infsup}
    text = json.dumps(out, sort_keys=True, indent=1) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print("sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
