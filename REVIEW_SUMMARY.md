# Review summary (public, rating-free)

The manuscript was checked by internal AI-assisted review in several roles (editor, methodology, domain, perspective and devil's-advocate), an AI-assisted adversarial review by a different model (OpenAI gpt-5.6-sol) and a verification re-review. Original reports are kept privately. This summary keeps the substantive objections and responses. It is not peer review.

| Objection | Response in the published version |
|---|---|
| Positioning against two-dimensional constructions, other three-dimensional moment work and alternative routes to stability was thin | New Section 1.5 compares Fortin–Soulié, Matthies–Tobiska, Baran–Stoyan, Crouzeix–Raviart, Ciarlet–Dunkl–Sauter, Sauter–Torres, Chen–Hu–Zhang, Ye–Zhang, Bohne–Ciarlet–Sauter, conforming and split-mesh elements, finite element exterior calculus and virtual elements; Corollary 6.3 explains why triangles never need an extra degree |
| Source quotations and the status of the source's statements | The Appendix A sentence and Remarks 3.2 and 3.6 are quoted verbatim; Remark 3.2 is sharpened from n<=5 to exactly n<=2; the source's Appendix A is described as a question; priority language was removed |
| Terminology: unisolvence versus surjectivity; scope of "minimum degree" | "Surjective" is used for the rank statements and "unisolvent" only for bijectivity; the single-cell polynomial scope is stated in the abstract, after Theorem 2.1 and in the exclusions; the one-sextic element is called a minimal-rank repair with no classification claimed |
| Gaps in proofs: sign rule, integration identity, Gram entries, 14-monomial basis | Lemma 3.1 (sign rule via the cofactor identity), Lemma 4.4 (Dirichlet aggregation), the Andreief identity cited, Gram entries derived exactly, Proposition 5.9 proved analytically by a triangular leading-form argument |
| Readability for non-specialists | Plain-language abstract and Section 1.1, the answer in words, labelled contributions, roadmap, notation table, an overview figure and a worked example; symbol clashes removed |
| Implementation guidance and conditioning; the tiny exact determinant | Section 9: global-vertex-ordered face bases, quadrature degrees, DOF counts and condensation, conditioning of monomial versus orthonormal bases; the determinant is explained as a scaling artefact |
| Extensions requested: other dimensions, minimal symmetric elements, a numerical test | Proved closed form and bounds in d dimensions with exact checks and a conjecture; proved minimal count and existence of symmetric lean elements with explicit cases; a numerical Stokes test (rates consistent with fourth order; inf-sup values 0.40–0.37 for N=1–4) |
| Reproducibility | Deterministic certificates without timings; `verify_all.py` replays byte for byte against SHA256SUMS |

Remaining open items: Conjectures 5.6 and 6.4; no dedicated search of the Sobolev-orthogonal-polynomial literature for a prior kernel statement (no priority is claimed); the numerical study covers one mesh family and did not compute the inf-sup value at N=5.
