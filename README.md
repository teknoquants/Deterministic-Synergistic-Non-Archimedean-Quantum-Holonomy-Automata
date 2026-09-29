# Python validation suite for DS-NAQHA

Source: **Deterministic Synergistic Non-Archimedean Quantum Holonomy Automata (DS-NAQHA): An Architectural Blueprint for Gauge Field Regularization on Bruhat–Tits Trees**, September 2026, supplied as `1788940434090.pdf` (13 PDF pages).

This suite covers all ten numbered theorems, all three algorithmic subroutines, and the mathematical formulas/claims in Sections II–V and VIII–IX. It checks valid finite models, provides exact arithmetic where practical, and exposes incorrect or underspecified claims. **It does not certify all the paper's theorems. Finite numerical experiments cannot prove unrestricted infinite-dimensional assertions.**

## Run

Requires Python 3.10 or later. No quantum SDK, GPU, internet access during execution, or copy of the paper is needed.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The default reproducible run uses 20,000 independent SU(3) samples and seed 20260929. It writes `results/validation_report.json` and `results/validation_report.md`.

```bash
python run_all.py --samples 100000 --seed 42 --output results_high_precision
python run_all.py --section theorems
python run_all.py --section foundations
python run_all.py --section subroutines
```

Run these commands from the extracted `ds_naqha_validation` directory. Changing the section changes random-number consumption; results remain reproducible for the same full command. Runtime and memory grow with the sample count. The default run is small enough for an ordinary laptop. Requirements specify compatible minimums; the included report records the exact versions used here.

## Files

| File | Purpose |
|---|---|
| `run_all.py` | CLI, reproducible seeds, reports and exit status |
| `common.py` | SU(3) Haar sampler, Lie generators, singlet projectors, exact Weyl integrals, result recorder |
| `validate_foundations.py` | Haar identities, gauge transformations, geometry and frequency arithmetic |
| `validate_theorems.py` | Tests and explicit scope assessments for Theorems 1–10 |
| `validate_subroutines.py` | QPE, Hadamard-test and polynomial/block-encoding simulations |
| `requirements.txt` | NumPy and SciPy dependencies |
| `results/validation_report.md` | Saved detailed execution results |
| `results/validation_report.json` | Same results in machine-readable form |

## Result meanings

- **PASS**: the specified exact or numerical check reproduced the stated result; this is not a blanket approval of its parent theorem.
- **COUNTEREXAMPLE**: a contradiction to a printed formula, or a failure of an unqualified algorithmic claim, was successfully reproduced. These are expected findings, not software failures.
- **QUALIFICATION**: a necessary hypothesis, interpretation or missing specification is recorded. These are not presented as computational proofs.
- **FAIL**: a validation check failed to reproduce its expected outcome; investigate the report. The program exits with code 1 for any such failure, and code 0 otherwise. It still prints all mathematical objections even when the exit code is 0.

`kind` distinguishes exact arithmetic, finite linear algebra, floating-point checks, Monte Carlo checks and scope notes. Monte Carlo uses seven estimated standard errors per complex coordinate; this is a generous diagnostic threshold, **not** a rigorous simultaneous confidence guarantee. Matrix identities use absolute tolerances, normally 1e-10. Exact class integrals use integer Laurent-polynomial coefficients and `fractions.Fraction`, not numerical quadrature.

## Complete coverage map

Page numbers below are physical PDF pages.

| Paper item | Pages | Check IDs | What is validated; limits |
|---|---:|---|---|
| Section II: gauge action, product Haar measure, normalization, second and third moments | 2–3 | F01–F10; T1; T7 | SU(3) QR sampling, all 81 second moments and all 729 epsilon third moments; exact character moments; endpoint covariance and inverse edges. Only one orientation per undirected link is independent. |
| Section II: p-adic norm bound and ultraviolet assertion | 3 | F11–F13 | `x=1/3` has norm 3 at prime 3. The unit-ball bound is for Z_p. Casimir growth remains unbounded. Compact projective boundary does not make the bulk tree compact or remove electric cutoffs. |
| Section III: electric Hamiltonian, vacuum and source/leaf energies | 3–4 | T1; T2; T6.open | Generator-derived Casimirs, vacuum, one-link 2g²/3 and two-link 4g²/3 costs; physical-sector and boundary-condition qualifications. |
| Sections IV–V: hardware numerical statements | 4–5 | F14–F16 | Wavelength/frequency conversion and optical/microwave energy ratio. No Hamiltonian, loss model or measured data is supplied for validating transduction, fidelity, blockade constraints or coherence. |
| Theorem 1: gauge projector | 5 | T1.* | Haar average vs. independently constructed Gauss nullspace; Hermiticity, idempotence and all eight generators in 3, 3⊗bar3 and 3⊗3⊗3 sectors. Infinite-dimensional statement still needs the Haar averaging proof. |
| Theorem 2: spectrum and gap | 6 | T2.* | Fundamental/adjoint/sextet Casimirs, exact rational energies and finite-tree constraints. Source-free and static-source sectors cannot be silently combined. Infinite-volume discreteness is not established. |
| Theorem 3: open Wilson lines | 6–7 | T3.* | Independent paths of lengths 1–4 and exact nontrivial character means. The path is nonempty; the open trace is gauge variant. Correct factorization retains the outer trace. |
| Theorem 4: quotient cycles and magnetic term | 7 | T4.* | Incidence-matrix cycle ranks for finite cores; nonzero operator matrix element. The printed bare Haar integral is zero. Backtracking loops have trace 3. No explicit arithmetic Schottky quotient is supplied. |
| Theorem 5: bulk Green kernel and boundary scaling | 7–8 | T5.* | Radial recurrence, normalization, mass-dimension relation, depth-distance formula and renormalized boundary scaling over several primes and dimensions. Sign conventions and resolvent branch matter. No general SU(3) bulk/boundary duality is demonstrated. |
| Theorem 6: first-order strong coupling | 8 | T6.* | Exact fusion integrals and zero diagonal matrix elements, g^-4 scaling, trivial tree case. Explicit degenerate mixing refutes the proposed general extension to quotient cycles. |
| Theorem 7: automorphism invariance | 8–9 | T7.* | Exact product-cylinder invariance under permutations and inversion, plus numerical reversed-edge moments. Infinite extension follows from the cylinder-measure argument, not a finite sample. |
| Theorem 8: complexity | 9 | T8.* | Exact finite vertex counts, binary index widths and gauge storage. One-hot is not binary indexing. Register size alone gives no oracle or total gate complexity bound. |
| Theorem 9: partition convergence | 9 | T9.* | SU(3) dimensions, multiplicity squared, positive-temperature heat trace and explicit tail bound. Full product trace differs from gauge-projected trace; beta>0 is required; no infinite-volume finite-Z conclusion. |
| Theorem 10: entanglement bound | 10 | T10.* | Actual finite pure TTN states and saturation of S≤log d. Requires a Schmidt/bond rank bound; arbitrary states on a tree need not obey it. Entropy uses natural logarithms. |
| Subroutine 1: QPE projection | 10 | A1.* | Exact statevector simulation, conditional projector and success probability; phase-aliasing counterexample and finite-resolution leakage. |
| Subroutine 2: Hadamard test | 10–11 | A2.* | Valid color-reference unitary realization with Tr(U)/3 and mean p(0)=1/2. A controlled traced Wilson operator is generally nonunitary. |
| Subroutine 3: QSVT | 11 | A3.*; T8 | Finite unitary block encoding; explicit Chebyshev QSP sequence with d queries; bounded filter and gap-dependent degree. No target polynomial/phase schedule/oracle is given for a full paper-specific implementation. |
| Sections VIII–IX and abstract: repeated global claims | 1, 11–12 | F11–F16; T2; T8; T9; A3 | Same geometry, spectral, storage, truncation and error qualifications. QCD relevance, novelty and device feasibility are not mathematical results established by these scripts. Bibliographic priority is not audited. |

## Important mathematical findings

### Spectrum and sectors (Theorem 2)

With Gauss law at every vertex, including leaves, a finite source-free tree has only the trivial representation on every link: start at a leaf and remove it inductively. Its physical space has only the vacuum. Relaxing leaf constraints admits boundary flux paths and can give the stated two-link energy when sibling leaves exist. Specifying static charges defines another sector; its minimal flux energy is not an excitation above a vacuum within that same sector.

For finitely many unconstrained links the electric operator has compact resolvent. Infinitely many links can produce infinitely many independent excitations at the same energy; discrete finite-volume spectra do not establish discrete spectrum with finite multiplicities in the infinite-volume theory. A pure-point spectrum and compact-resolvent discreteness are different claims. The report's infinite-multiplicity illustration is explicitly in the unconstrained reference-vacuum space, not a fabricated gauge-invariant state.

### Cycles and perturbation (Theorems 4 and 6)

For a simple cycle of independent Haar links, `integral Tr(U_c) dU = 0`. The magnetic multiplication operator can nevertheless be nonzero, since `integral conjugate(chi_3) chi_3 dU = 1`.

On a one-cycle model, the gauge-invariant character states `chi_3` and `chi_bar3` are degenerate electric eigenstates. At g=1 the perturbation restricted to these states is

```text
[[ 0,   -1/2],
 [-1/2,  0  ]]
```

Its eigenvalues are -1/2 and +1/2. Thus zero diagonal first-order expectations do not imply zero first-order spectral shift. These matrix elements are calculated by exact SU(3) Weyl integration. The open-tree version remains trivially true because its magnetic sum is empty.

### Green function (Theorem 5)

For `L = degree - adjacency`, the displayed positive kernel satisfies

```text
(L + m²)G = delta
m² = p^Delta + p^(1-Delta) - (p+1)
G(d) = p^(-Delta*d)/(p^Delta - p^(-Delta)).
```

For `Box = -L`, the paper's `(Box-m²)G` instead has source `-delta`; either the operator equation or the sign of G must change. Since the paper does not define its Laplacian sign or its mass relation, the suite states the conventions explicitly. The local recurrence works for Delta>0, while an l2 resolvent column uses Delta>1/2. The boundary formula is checked in a Z_p coordinate patch at distinct regulated endpoints; it is not a global norm formula at the projective point at infinity.

### Partition function (Theorem 9)

For `t=beta*g²/2>0`, `partition_link(t, K)` sums irreps with `a+b<=K`. Write `n=a+b`. Then

```text
C2(a,b) >= n²/4 + n
dim(a,b) <= (n+2)^3/8.
```

There are n+1 irreps in shell n, so its contribution is bounded by
`b_n=(n+2)^7 exp(-t*n²/4)/64`. The ratios `b_(n+1)/b_n` decrease. If the first omitted ratio r is below one, the entire tail is at most `b_(K+1)/(1-r)`. This supplies a real error bound instead of treating apparent numerical stabilization as a proof of convergence. Its evaluation is float64, not certified outward-rounded interval arithmetic. The Gaussian `r^7` comparison in the paper is a growth estimate, not an exact integral identity. A projected physical trace is no greater than the full finite-volume trace, so finiteness survives projection although factorization generally does not.

### Quantum subroutines

The QPE test simulates controlled powers and an inverse QFT on a finite constraint operator. Projection is probabilistic and requires spectral resolution without phase wraparound. The Hadamard test uses an actual unitary holonomy and a color reference. The QSP test uses a conventional signal rotation to reproduce Chebyshev polynomials; it is a validated example, not invented phases for the unspecified paper algorithm. A simple bounded filter `(1-x²)^k` illustrates explicit gap dependence without claiming optimal QSVT complexity. The full, untruncated electric Hamiltonian is unbounded even on one link, so `H/||H||` needs a truncation or a different bounded-operator construction.

## Method references

The attachment is the primary specification. All numerical checks operate on equations transcribed from it or on explicitly labeled corrections/counterexamples. General methodological context:

- Gubser et al., *p-adic AdS/CFT*: https://arxiv.org/abs/1605.01061 — scalar bulk/boundary setting; the script independently checks the recurrence and normalization.
- Gilyén et al., *Quantum singular value transformation and beyond*: https://arxiv.org/abs/1806.01838 — polynomial transformations of block encodings; the suite provides a small conventional QSP example and does not claim a full synthesized application circuit.

The programs are classical validation scripts. They do not establish 4D Yang–Mills confinement, hardware feasibility, an arithmetic Schottky uniformization, a full holographic duality, or a formally machine-checked proof of every infinite-dimensional theorem.

## License

MIT License. Copyright (c) September 2026 Roger Cooperman; A.E. Petrunina V.V.

The full license is included in `LICENSE` and in every Python source file.
