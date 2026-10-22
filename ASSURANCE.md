# Assurance

| Dimension | State | Note |
|---|---|---|
| Written proofs | present | Complete proofs of every result labelled proof; not machine-checked |
| Exact checks | passed | Rational and modular programs in `code/` (standard library and SymPy); they confirm instances and constants, not the general theorems |
| Producer replay | passed | `code/verify_all.py` regenerates every certificate in a temporary directory and compares it byte for byte with the shipped file and `SHA256SUMS`. A clean full replay of the final research package ran in a separate copy; the released code and certificates were bound to it by SHA-256 (byte-identical), and the checks affected by publication changes were rerun |
| Negative controls | passed | `negative_controls.py` (standard library, exact): recomputes A on all 56 quintic monomials and on Delta_4 with its own polynomial code, checks the rank table against the closed forms, and confirms that a wrong obstruction value, unsigned face weights and a corrupted deficit formula are rejected |
| Independent rerun | not assessed | No unaffiliated rerun |
| Independent reimplementation | partial | `code/independent_check.py` and `negative_controls.py` are separately written producer-side checks, not external reproduction |
| Formal verification | not assessed | No proof assistant |
| Specialist review | not assessed | AI-assisted internal and adversarial review is not external specialist review |
| Editorial peer review | not assessed | None |
| Prior art | bounded | Dated bounded searches summarised in [RESEARCH_GATES.json](RESEARCH_GATES.json); no priority claim |

Trust boundary: the general statements rest on the written proofs. Exact programs check ranks and constants for listed cases. Values labelled numerical are floating-point results rounded to four significant digits.
