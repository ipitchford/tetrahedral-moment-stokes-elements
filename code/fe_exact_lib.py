"""Exact polynomial, moment and rank utilities for simplicial moment elements.

Polynomials on the d-simplex are sparse dictionaries mapping exponent tuples
(a_0, ..., a_d) of the barycentric coordinates l_0, ..., l_d to Fractions.
Only the Python standard library is used.

Conventions
-----------
* Facet F_i is {l_i = 0}.  Facet moments are normalised integrals
  avg_{F_i}(v q) for q in P_n(F_i); interior moments are avg_K(v r) for
  r in P_{n-1}(K).  Test functions are barycentric monomials, homogeneous of
  degree n (facet, in the d coordinates other than l_i) and n-1 (interior).
* Normalised monomial integrals:
      avg over a k-simplex of prod l^a = k! prod a_i! / (|a| + k)!.
* P_D(K) is spanned by the homogeneous barycentric monomials of degree D.
"""
import json
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, permutations
from math import comb, factorial, prod

PRIME = 2 ** 61 - 1


# ----------------------------------------------------------------------------
# combinatorics
# ----------------------------------------------------------------------------
def compositions(n, k):
    """Exponent tuples of length k and total degree n (lexicographic)."""
    if k == 1:
        yield (n,)
        return
    for j in range(n + 1):
        for rest in compositions(n - j, k - 1):
            yield (j,) + rest


def monomials_upto(n, k):
    return [a for t in range(n + 1) for a in compositions(t, k)]


def dim_P(D, d):
    return comb(D + d, d)


def N_dofs(d, n):
    return (d + 1) * comb(n + d - 1, d - 1) + comb(n + d - 1, d)


def kernel_dim(d, m):
    return sum(comb(d, r) for r in range(1, min(d, m) + 1))


def perm_sign(p):
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


# ----------------------------------------------------------------------------
# sparse polynomials
# ----------------------------------------------------------------------------
def clean(p):
    return {a: F(c) for a, c in p.items() if c}


def add(*ps):
    out = {}
    for p in ps:
        for a, c in p.items():
            out[a] = out.get(a, F(0)) + c
    return clean(out)


def scale(p, c):
    return clean({a: c * x for a, x in p.items()})


def mul(p, q):
    out = {}
    for a, x in p.items():
        for b, y in q.items():
            e = tuple(i + j for i, j in zip(a, b))
            out[e] = out.get(e, F(0)) + x * y
    return clean(out)


def one(nv):
    return {(0,) * nv: F(1)}


def var(i, nv):
    return {tuple(int(j == i) for j in range(nv)): F(1)}


def power(p, k, nv):
    out = one(nv)
    for _ in range(k):
        out = mul(out, p)
    return out


def deriv(p, i):
    out = {}
    for a, c in p.items():
        if a[i]:
            b = list(a)
            b[i] -= 1
            out[tuple(b)] = c * a[i]
    return clean(out)


def edge_deriv(p, i, j):
    return add(deriv(p, i), scale(deriv(p, j), -1))


def relabel(p, sigma):
    """(p o sigma): the variable l_i is replaced by l_sigma(i)."""
    out = {}
    for a, c in p.items():
        b = [0] * len(a)
        for i, e in enumerate(a):
            b[sigma[i]] += e
        b = tuple(b)
        out[b] = out.get(b, F(0)) + c
    return clean(out)


def substitute_univariate(coeffs, t, nv):
    return add(*(scale(power(t, k, nv), c) for k, c in enumerate(coeffs)))


def serialise_poly(p):
    return [[list(a), str(c)] for a, c in sorted(p.items())]


def homogenise(p, D, nv):
    """Rewrite p (degree <= D) as a homogeneous form of degree D using sum l = 1."""
    s = add(*(var(i, nv) for i in range(nv)))
    out = {}
    for a, c in p.items():
        out = add(out, scale(mul({a: F(1)}, power(s, D - sum(a), nv)), c))
    return out


# ----------------------------------------------------------------------------
# integrals and moments
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def simplex_avg_monomial(a):
    """avg over the k-simplex with barycentrics given by the k+1 entries of a."""
    k = len(a) - 1
    return F(factorial(k) * prod(factorial(x) for x in a), factorial(sum(a) + k))


@lru_cache(maxsize=None)
def monomial_moment(a, face):
    """avg_K(l^a) if face < 0, avg_{F_face}(l^a) otherwise (zero if a[face] > 0)."""
    if face < 0:
        return simplex_avg_monomial(a)
    if a[face]:
        return F(0)
    return simplex_avg_monomial(a[:face] + a[face + 1:])


def avg(p, face=-1):
    return sum((c * monomial_moment(a, face) for a, c in p.items()), F(0))


def moment_rows(d, n):
    """List of (face, test exponent).  face = -1 marks interior moments."""
    rows = []
    for f in range(d + 1):
        for q in compositions(n, d):
            rows.append((f, q[:f] + (0,) + q[f:]))
    for q in compositions(n - 1, d + 1):
        rows.append((-1, q))
    return rows


def monomial_moment_vector(a, rows):
    return [monomial_moment(tuple(x + y for x, y in zip(a, q)), f) for f, q in rows]


def poly_moment_vector(p, rows):
    out = [F(0)] * len(rows)
    for a, c in p.items():
        for i, v in enumerate(monomial_moment_vector(a, rows)):
            if v:
                out[i] += c * v
    return out


def moment_matrix_monomials(d, n, D):
    """Rows: moments of M_{d,n}; columns: homogeneous monomials of degree D."""
    rows = moment_rows(d, n)
    cols = list(compositions(D, d + 1))
    colvecs = [monomial_moment_vector(a, rows) for a in cols]
    return [list(r) for r in zip(*colvecs)], rows, cols


def moment_matrix_polys(polys, rows):
    colvecs = [poly_moment_vector(p, rows) for p in polys]
    return [list(r) for r in zip(*colvecs)]


# ----------------------------------------------------------------------------
# linear algebra
# ----------------------------------------------------------------------------
def to_mod(M, p=PRIME):
    out = []
    for row in M:
        r = []
        for x in row:
            x = F(x)
            if x.denominator % p == 0:
                raise ValueError("denominator not invertible modulo p")
            r.append(x.numerator * pow(x.denominator, -1, p) % p)
        out.append(r)
    return out


def rank_mod(M, p=PRIME):
    """Rank over GF(p).  For a rational matrix this is a lower bound for the rank over Q."""
    A = to_mod(M, p)
    if not A:
        return 0
    m, ncol = len(A), len(A[0])
    r = 0
    for c in range(ncol):
        piv = None
        for i in range(r, m):
            if A[i][c]:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = pow(A[r][c], -1, p)
        pr = [x * inv % p for x in A[r]]
        A[r] = pr
        for i in range(r + 1, m):
            v = A[i][c]
            if v:
                Ai = A[i]
                A[i] = [(x - v * y) % p for x, y in zip(Ai, pr)]
        r += 1
        if r == m:
            break
    return r


def rref(M):
    """Exact reduced row echelon form over Q; returns (R, pivot columns)."""
    A = [[F(x) for x in row] for row in M]
    m = len(A)
    ncol = len(A[0]) if m else 0
    piv = []
    r = 0
    for c in range(ncol):
        k = next((i for i in range(r, m) if A[i][c]), None)
        if k is None:
            continue
        A[r], A[k] = A[k], A[r]
        inv = 1 / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(m):
            if i != r and A[i][c]:
                v = A[i][c]
                A[i] = [x - v * y for x, y in zip(A[i], A[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    return A, piv


def rank_exact(M):
    return len(rref(M)[1]) if M else 0


def rank_certified(M, exact_limit=12000):
    """Exact rank over Q when the matrix is small, otherwise rank modulo PRIME."""
    if len(M) * len(M[0]) <= exact_limit:
        return rank_exact(M), "exact_rational_elimination"
    return rank_mod(M), "modular_rank_p_2^61-1_lower_bound"


def det_bareiss(M):
    A = [list(r) for r in M]
    n = len(A)
    den = 1
    sgn = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            j = next((j for j in range(k + 1, n) if A[j][k]), None)
            if j is None:
                return 0
            A[k], A[j] = A[j], A[k]
            sgn = -sgn
        pv = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                z = A[i][j] * pv - A[i][k] * A[k][j]
                assert z % den == 0
                A[i][j] = z // den
            A[i][k] = 0
        den = pv
    return sgn * A[-1][-1]


def det_rational(M):
    """Exact determinant of a rational matrix via row scaling and Bareiss."""
    total = F(1)
    IM = []
    for row in M:
        den = 1
        for x in row:
            den = den * F(x).denominator // __import__("math").gcd(den, F(x).denominator)
        IM.append([int(F(x) * den) for x in row])
        total /= den
    return det_bareiss(IM) * total


def factor_small(n):
    n = abs(n)
    out = {}
    p = 2
    while p * p <= n and p < 10 ** 6:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return {str(k): v for k, v in sorted(out.items())}


# ----------------------------------------------------------------------------
# tetrahedral special functions
# ----------------------------------------------------------------------------
def shifted_legendre(m):
    """Coefficients of P_m(2t-1) in powers of t."""
    return [F((-1) ** (m - k) * comb(m, k) * comb(m + k, k)) for k in range(m + 1)]


def vertex_ridge_coeffs(m):
    """Coefficients of t P^{(1,1)}_{m-1}(2t-1) = t (d/dt P_m(2t-1)) / (m+1)."""
    p = shifted_legendre(m)
    return [F(0)] + [k * p[k] / (m + 1) for k in range(1, m + 1)]


def tet_ridges(m):
    """U_0..U_3 and V_01, V_02, V_03 on the tetrahedron (degree m)."""
    lam = [var(i, 4) for i in range(4)]
    u = [substitute_univariate(vertex_ridge_coeffs(m), t, 4) for t in lam]
    v = [substitute_univariate(shifted_legendre(m), add(lam[0], lam[j]), 4) for j in (1, 2, 3)]
    return u + v


def vandermonde(idx, nv):
    """prod_{j<k in idx} (l_k - l_j)."""
    out = one(nv)
    for j, k in combinations(idx, 2):
        out = mul(out, add(var(k, nv), scale(var(j, nv), -1)))
    return out


DELTA4 = vandermonde(range(4), 4)
FACE_DELTA = [vandermonde([j for j in range(4) if j != i], 4) for i in range(4)]


def alternating_functional(v):
    """A(v) = sum_i (-1)^i avg_{F_i}(v delta_i)."""
    return sum(((-1) ** i * avg(mul(v, FACE_DELTA[i]), i) for i in range(4)), F(0))


def edge_energy(u, v, nv=4):
    """a_B(u, v) = avg_K sum_{i<j} D_ij u D_ij v."""
    return sum((avg(mul(edge_deriv(u, i, j), edge_deriv(v, i, j)))
                for i, j in combinations(range(nv), 2)), F(0))


ALL_PERMS4 = list(permutations(range(4)))


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------
def write_json(data, path):
    text = json.dumps(data, sort_keys=True, indent=1) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def frac_matrix(M):
    return [[str(F(x)) for x in row] for row in M]
