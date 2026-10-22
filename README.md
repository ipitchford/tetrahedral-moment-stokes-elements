# Minimal-degree polynomial nonconforming Stokes elements on tetrahedra

## In plain English

Engineers who simulate slow viscous flow often build their finite elements from
*moments*: averages of the velocity against polynomials on each face and inside
each cell of a tetrahedral mesh. These moments make the velocity and pressure
fit together stably. Each element needs a polynomial space on which the moments
determine the function uniquely, and the space must contain enough polynomials
for the target accuracy.

A 2025 paper by Balazi, Allaire, Jolivet and Omnes built such elements of second
and third order. It asked whether a fourth-order element could be made by adding
fifteen degree-five polynomials. This paper shows that no choice of degree-five
polynomials works, explains why, and gives the fix. One degree-six polynomial
(the product of the six differences of barycentric coordinates) supplies the one
missing direction. More generally, for every order the paper finds the exact set
of polynomials that the moments cannot see. From this it determines the smallest
polynomial degree that works and builds elements that do not depend on how the
vertices are numbered. A numerical Stokes test shows rates consistent with fourth-order
convergence; computed inf-sup values are 0.40–0.37 for N=1–4.

## What is proved, computed and observed

| Claim | Status |
|---|---|
| The kernel of the moments on P_{n+2} is 7-dimensional with an explicit basis, for every n. Deficit (n-1)(n-2)/2. Surjective on P_{n+3}. Minimum degree n+2 (n<=2) or n+3 (n>=3) | proof |
| The fourth-order obstruction (an alternating functional with value 1/34650 on the Vandermonde sextic) | proof |
| A minimal number of top-degree directions; vertex-invariant lean elements exist for every n | proof |
| The missing directions form sgn ⊗ Sym^{n-3}(std) | exact symbolic computation (a generating-function identity) |
| The lean element P_{n+2} + Δ4·P_{n-3} works for n = 3..8; explicit n=4 and n=5 elements | exact computation |
| The same for all n | conjecture |
| d-simplex kernel dimension and closed form for the deficit | proof |
| Surjectivity on P_{n+3} for d=4 (n<=5), d=5 (n<=4), d=6 (n=3) | exact computation (full rank mod a prime certifies rank over Q) |
| Sharp degree n+3 in every dimension | conjecture |
| Stokes convergence rates of about 4 and a discrete inf-sup constant of about 0.37; conditioning | numerical |

## Contents

- `tetrahedral_moment_kernels.tex` and `.pdf`: the paper.
- `code/fe_exact_lib.py`: exact polynomial, moment and rank utilities.
- `code/verify_tetrahedron.py` → `certificates/tetrahedron_exact.json`: ranks for
  n=1..6, the ridges, A(Δ4)=1/34650, the sign rule, the Gram matrix, the
  14-monomial basis and its determinant, and invariance.
- `code/verify_dimensions.py` → `certificates/dimensions_exact.json`: the deficit
  table and closed forms, and ranks for d=2,4,5,6.
- `code/verify_lean_element.py` → `certificates/lean_element_exact.json`: the
  lean spaces for n=3..8, the explicit n=4 and n=5 elements, and the S4
  character identity.
- `code/independent_check.py` → `certificates/independent_check.json`: a
  separate reconstruction of the main ranks, ridges and the value 1/34650,
  with its own polynomial code.
- `code/stokes_n3_numerical.py` → `certificates/stokes_n3_numerical.json`: the
  numerical Stokes test (floating point; values rounded to 4 significant
  digits).
- `code/verify_all.py`: replay driver. `SHA256SUMS`: checksums of all
  shipped files.

## Replay

Requirements: Python 3.12 or later. The exact scripts need only the standard
library and SymPy; the numerical script needs NumPy, SciPy and SymPy. From the
package root:

    python3 code/verify_all.py                   # everything (about 5 minutes, about 7 GB RAM)
    python3 code/verify_all.py --skip-numerical  # exact certificates only (about 1 minute)
    python3 -B -O negative_controls.py           # standard-library semantic negative controls
    sha256sum -c SHA256SUMS                      # or: shasum -a 256 -c SHA256SUMS

`verify_all.py` regenerates every certificate in a temporary directory and
compares it byte for byte with the shipped file and with `SHA256SUMS`. The
outputs contain no timings, dates or paths. The numerical JSON is rounded to
4 significant digits. On a different BLAS or platform, the last digit of a
floating-point entry could in principle differ; the exact certificates do
not depend on the platform.

## Citation and licence

Author: Anonymous. Version 1.0.0-candidate; DOI [10.5281/zenodo.23565002](https://doi.org/10.5281/zenodo.23565002). Unrefereed AI-assisted candidate: internal AI-assisted review, AI-assisted adversarial review (OpenAI gpt-5.6-sol) and a verification re-review; not peer review, independent reproduction or formal verification. Prose, manuscript and certificates CC0-1.0; code MIT ([LICENSES.md](LICENSES.md)).

Release records: [STATUS.md](STATUS.md), [ASSURANCE.md](ASSURANCE.md), [PROVENANCE.md](PROVENANCE.md), [AI_INDEX.md](AI_INDEX.md), [CLAIMS.json](CLAIMS.json), [RESEARCH_GATES.json](RESEARCH_GATES.json), [REVIEW_SUMMARY.md](REVIEW_SUMMARY.md), [CITATION.cff](CITATION.cff), `.zenodo.json`, [LICENSES.md](LICENSES.md).
