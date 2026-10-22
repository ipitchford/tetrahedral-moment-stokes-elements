# Agent navigation

## Identity

- Release slug: `tetrahedral-moment-stokes-elements`; version `1.0.0-candidate`; DOI 10.5281/zenodo.23565002 (concept DOI 10.5281/zenodo.23565001)
- Scholarly creator: Anonymous. Status: unrefereed candidate ([STATUS.md](STATUS.md))
- Manuscript: [PDF](tetrahedral_moment_kernels.pdf), [LaTeX source](tetrahedral_moment_kernels.tex); human overview [README.md](README.md); structured target [target.yaml](target.yaml)
- Machine-readable claims: [CLAIMS.json](CLAIMS.json)
- Assurance limits: [ASSURANCE.md](ASSURANCE.md); internal review summary: [REVIEW_SUMMARY.md](REVIEW_SUMMARY.md)
- Provenance: [PROVENANCE.md](PROVENANCE.md); prior art and limitation triage: [RESEARCH_GATES.json](RESEARCH_GATES.json)
- Citation: [CITATION.cff](CITATION.cff), [Zenodo metadata](.zenodo.json); licences: [LICENSES.md](LICENSES.md)

## Scientific map

Object: tetrahedron K, barycentrics lambda_0..lambda_3; moment map M_n = normalised face moments against P_n on the four faces plus cell moments against P_(n-1); N_n = 4 C(n+2,2) + C(n+2,3) (13, 28, 50, 80, 119, 168 for n=1..6). Goal: V_n(K) with P_(n+1) in V_n, dim N_n, M_n bijective. Source: Balazi, Allaire, Jolivet and Omnes (BAJO), doi:10.1002/num.70057, Appendix A question for n=3.

- **Theorem 2.1 (proof):** injective on P_(n+1); kernel on P_(n+2) = 7 for every n (vertex ridges lambda_i J^(1,1)_(n+1)(2 lambda_i - 1), pair ridges J_(n+2)(2(lambda_0+lambda_j) - 1)); deficit (n-1)(n-2)/2 = 0,0,1,3,6,10; surjective on P_(n+3); minimum degree n+2 (n<=2), n+3 (n>=3).
- **Proposition 3.2 (proof; exact):** A(v) = sum_i avg_(F_i) v eps_i, eps_i = (-1)^i Vandermonde of the other three coordinates; A = 0 on P_5, A(Delta_4) = 1/34650. Answers BAJO in the negative.
- **Section 5 (proof):** minimal count (Corollary 5.2), existence of vertex-invariant lean elements (Theorem 5.4), 50-dimensional V_3^* with one sextic (Corollary 5.8), 14-monomial basis (Proposition 5.9). Exact: sgn x Sym^(n-3)(std) (Proposition 5.5), Delta_4 P_(n-3) for n=3..8. Conjecture 5.6: all n.
- **Section 6:** d-simplex kernel and deficit (proof), triangles never deficient (Corollary 6.3), n+3 checked for d=4,5,6 (exact, modulo 2^61-1); Conjecture 6.4 for d>=4.
- **Section 8 (numerical):** rates consistent with fourth order; inf-sup values 0.40-0.37, N=1..4.

Do not read: "minimum degree" as covering split-cell, nonpolynomial or virtual element spaces; exact checks as proofs of the all-n theorems; the numerics as proof of mesh independence; Conjectures 5.6 and 6.4 as proved.

## Replay

Python 3.12+; exact scripts need the standard library and SymPy; the numerical script needs NumPy and SciPy (about 7 GB RAM):

```sh
shasum -a 256 -c SHA256SUMS
python3 -B -O negative_controls.py              # standard library: A on P_5 and Delta_4, rank-table closed forms, corruptions rejected
python3 -B code/verify_all.py --skip-numerical  # exact certificates byte for byte (about 1 minute)
python3 -B code/verify_all.py                   # everything (about 5 minutes)
```

Programs: [code/](code/); certificates: [certificates/](certificates/).

## Useful next research question

Prove Conjecture 5.6 (Delta_4 P_(n-3) is a lean enrichment for every n) or Conjecture 6.4 (degree n+3 suffices on every d-simplex).
