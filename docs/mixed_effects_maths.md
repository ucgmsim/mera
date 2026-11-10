# The mathematics of MERA, step by step

This guide explains what `mera.run_mera` computes and why, for readers with basic
mathematics: means and standard deviations, and what a vector and a matrix are. The few
matrix operations it uses are listed where they are first needed
([section 3](#3-the-model-in-matrix-form)). It needs no statistics beyond that, and no
Rust.

It follows one fit from the model to the numbers that MERA reports. Every step is also in
[`mera/example/maths_walkthrough.py`](../mera/example/maths_walkthrough.py), which does the
same calculation in plain numpy and prints the intermediate results:

```
python -m mera.example.maths_walkthrough
```

Reading the two side by side is the quickest way in: start at the script's `main`
function, and look at `evaluate`, `find_theta` and `estimates` for the mathematics. The
section numbers below are the ones in that script's comments. [Section 9](#9-where-each-step-is-in-the-code) says where
each step is in the real code, and [section 10](#10-symbols) lists the symbols.

Contents:

1. [The model](#1-the-model)
2. [Why not just average? Shrinkage](#2-why-not-just-average-shrinkage)
3. [The model in matrix form](#3-the-model-in-matrix-form)
4. [What "best fit" means](#4-what-best-fit-means)
5. [Profiling: from four unknowns to two](#5-profiling-from-four-unknowns-to-two)
6. [Computing the criterion without the big matrix](#6-computing-the-criterion-without-the-big-matrix)
7. [Searching for the best θ](#7-searching-for-the-best-θ)
8. [The results at the best θ](#8-the-results-at-the-best-θ)
9. [Where each step is in the code](#9-where-each-step-is-in-the-code)
10. [Symbols](#10-symbols)

## 1. The model

A *residual* is the difference between an observed and a predicted ground-motion intensity
measure (IM), usually in natural-log units: ln(observed) − ln(predicted). There is one per
record, where a record is one earthquake recorded at one site.

MERA splits each residual into four parts:

```
residual = bias + event term + site term + remaining residual
```

- The **bias** is one number, the same for every record: how far off the predictions are
  on average.
- The **event term** is one number per earthquake, shared by all its records. It says
  whether that earthquake was stronger or weaker everywhere than predicted. The literature
  calls it the between-event residual, δB.
- The **site term** is one number per site, shared by all records at that site. It says
  whether that site shakes more or less than predicted in every earthquake. It is the
  site-to-site residual, δS2S.
- The **remaining residual** is what is left for the individual record. With a site
  term in the model, the literature writes it δWS, the single-site within-event residual.

The event terms, the site terms and the remaining residuals are each assumed to be
independent draws from a normal distribution with a mean of zero. Their standard
deviations are what a ground-motion study usually wants:

| Part | Standard deviation | Name in MERA's outputs |
|---|---|---|
| event terms | τ | `tau` |
| site terms | φ<sub>S2S</sub> | `phi_S2S` |
| remaining residuals | φ<sub>w</sub> | `phi_w` |
| all three together | σ = √(τ² + φ<sub>S2S</sub>² + φ<sub>w</sub>²) | `sigma` |

A note on `phi_w` for readers who know the notation of Al Atik et al. (2010). With a site
term fitted, `phi_w` is their single-station within-event standard deviation
φ<sub>SS</sub>. Without one (`compute_site_term=False`), the remaining residual is their
whole within-event residual δW, and `phi_w` is their φ.

So a fit has to find four numbers (the bias, τ, φ<sub>S2S</sub> and φ<sub>w</sub>) and
then the event and site terms themselves.

Two words from statistics appear everywhere in this subject:

- A **fixed effect** is an unknown constant that is estimated directly. Here, the bias.
- A **random effect** is a quantity treated as a draw from a distribution whose standard
  deviation is estimated. Here, the event terms and the site terms.

A model with both is a *mixed*-effects model. `run_mera` can also leave out the bias
(`assume_biased=False`) or the site term (`compute_site_term=False`).

## 2. Why not just average? Shrinkage

The obvious estimate of an event term is the average residual of that event's records.
The trouble is an event with only two records: its average owes a good deal to chance,
because the remaining residuals of two records do not average out. Taking it at face
value would overstate how much events differ.

The mixed model has a precise answer. Take the simplest case, with events only and a known
bias. The best estimate of the term of an event with n<sub>e</sub> records is

```math
\text{event term} = \frac{n_e\tau^2}{n_e\tau^2 + \phi_w^2} \times (\text{mean residual of the event} - \text{bias})
```

The fraction is between 0 and 1. It is called the *shrinkage*, because it pulls the
average towards zero. With τ = 0.5 and φ<sub>w</sub> = 0.6:

| Records of the event, n<sub>e</sub> | Shrinkage | Uncertainty left in the event term |
|---|---|---|
| 1 | 0.41 | 0.38 |
| 3 | 0.68 | 0.28 |
| 10 | 0.87 | 0.18 |
| 100 | 0.99 | 0.06 |

An event with a hundred records keeps almost all of its average. An event with one record
keeps less than half, because one record cannot tell an unusual earthquake from an unusual
recording. The last column is the *conditional standard deviation* of the event term,
which MERA also reports ([section 8](#8-the-results-at-the-best-θ)):

```math
\text{conditional standard deviation} = \sqrt{\frac{\tau^2\,\phi_w^2}{n_e\tau^2 + \phi_w^2}}
```

With sites in the model the idea is the same, but each kind of term depends on the other:

- an event term is the shrunk average of (residual − bias − site term) over its records;
- a site term is the shrunk average of (residual − bias − event term) over its records.

All the terms are therefore found together, by solving one set of simultaneous equations.
The walkthrough script checks the first statement on its data: the event term it prints
equals the shrinkage times that average, to every digit.

There is a catch. The shrinkage needs τ and φ<sub>w</sub>, which are not known in advance
and have to be estimated from the same data. Sections 4 to 7 are about how.

## 3. The model in matrix form

To handle thousands of records at once, the model is written with vectors and matrices.
Say there are n records, and stack their residuals into a vector **y**. Then

```math
y = X\beta + Zb + \varepsilon
```

- **Xβ** is the fixed-effects part. X has a row per record and a column per fixed effect.
  MERA has one fixed effect, the bias, so X is a single column of ones and β holds one
  number. Multiplying them gives every record the same value.
- **Zb** is the random-effects part. **b** holds all the event terms followed by all the
  site terms. Z has a row per record and a column per term, with a 1 in the column of the
  record's event, a 1 in the column of its site, and zeros elsewhere. Multiplying them
  gives each record the sum of its own event term and site term.
- **ε** holds the remaining residuals.

For example, take five records of three events (0, 1, 2) at two sites (0, 1):

```
record   event   site
  0        0      0
  1        0      1
  2        1      0
  3        1      1
  4        2      1

        events    sites
       0  1  2    0  1
 Z = [ 1  0  0    1  0 ]   record 0: event 0, site 0
     [ 1  0  0    0  1 ]   record 1: event 0, site 1
     [ 0  1  0    1  0 ]   record 2: event 1, site 0
     [ 0  1  0    0  1 ]   record 3: event 1, site 1
     [ 0  0  1    0  1 ]   record 4: event 2, site 1
```

Each event and each site is called a *level*, and the events and the sites are the two
*grouping factors*. The sizes used below are n records, p fixed effects (1 with a bias, 0
without) and q levels (events plus sites; 5 in the example).

### The matrix operations used below

The rest of this guide uses five things from matrix algebra. This is all that needs to
be known about each:

| Written | Name | What it is | In numpy |
|---|---|---|---|
| Aᵀ | transpose | A with its rows and columns swapped | `a.T` |
| I | identity matrix | ones on the diagonal and zeros elsewhere; multiplying by it changes nothing, like the number 1 | `np.eye(n)` |
| A<sup>−1</sup> | inverse | the matrix that undoes A: A<sup>−1</sup>A = I. It plays the part of division | `np.linalg.inv(a)` |
| \|A\| | determinant | one number computed from a square matrix. For a covariance matrix it measures the overall spread of the distribution, as s² does for a single quantity | `np.linalg.det(a)` |
| | covariance matrix | for a vector of random quantities: their variances on the diagonal and, off the diagonal, how strongly each pair varies together (zero if they are independent) | `np.cov` |

A *diagonal* matrix is zero everywhere except on its diagonal.

## 4. What "best fit" means

### Likelihood

For any values of the bias and the three standard deviations, the model says how probable
it would be to observe exactly the residuals in the data. That probability, seen as a
function of those values, is called the *likelihood*. *Maximum likelihood* picks the
values that make the observed data most probable.

The calculation has to allow for the fact that the records are not independent: two
records of the same earthquake share an event term, so they tend to be high or low
together. The model implies exactly how strongly. Measure the standard deviations of the
random effects relative to φ<sub>w</sub>:

```math
\theta_E = \frac{\tau}{\phi_w}, \qquad \theta_S = \frac{\phi_{S2S}}{\phi_w}
```

Then the covariance matrix of **y** is φ<sub>w</sub>² V, where V has a row and a column
for every record, and its entry for records i and j is

```
V[i, j] =   1          if i and j are the same record
          + θ_E²       if they have the same event
          + θ_S²       if they have the same site
```

In matrix form, V = I + ZΛΛᵀZᵀ. Here I is the identity matrix, ᵀ means transpose, and Λ
is the diagonal matrix that holds θ<sub>E</sub> for each event followed by
θ<sub>S</sub> for each site. These two numbers, written together as **θ**, are what the
rest of this guide is about.

### REML

Plain maximum likelihood has a known flaw: it underestimates standard deviations, because
it treats the estimated bias as if it were the true one. It is the same flaw as dividing
by n instead of n − 1 when computing a sample variance. *Restricted* maximum likelihood
(REML) corrects for it, and is the default of most mixed-model software, including the R
package lme4, whose method MERA follows.

To see where its formula comes from, start with a single number x drawn from a normal
distribution with mean μ and standard deviation s. The likelihood is the height of the
bell curve at x, and −2 times its logarithm is

```math
\log(2\pi s^2) + \frac{(x-\mu)^2}{s^2}
```

The second term is the squared misfit, measured in units of s. The first term grows with
s: it is the price of a wide distribution, which spreads its probability thinly. The best
s balances the two.

The formula for all n records together has the same two kinds of term. REML chooses the
values that minimise this quantity, which is −2 times the logarithm of the restricted
likelihood:

```math
(n-p)\log(2\pi\phi_w^2) \;+\; \log|V| \;+\; \log|X^T V^{-1} X| \;+\; \frac{r^2}{\phi_w^2}
```

The last term is the squared misfit, and the first two are the price of the spread:
φ<sub>w</sub>² and V together take the place of s². In the last term,

```math
\hat\beta = (X^T V^{-1} X)^{-1} X^T V^{-1} y, \qquad r^2 = (y - X\hat\beta)^T\, V^{-1}\, (y - X\hat\beta)
```

- β̂ is the best bias for a given θ. For MERA it is a weighted average of the residuals,
  in which records that share events and sites count for less than independent ones would.
  This is called generalised least squares.
- r² is a sum of squares of what the bias leaves unexplained, again allowing for the
  correlations. It is called the *penalised residual sum of squares*
  ([section 8](#8-the-results-at-the-best-θ) says why).

The price terms stop the fit from cheating. Making θ very large would let the event and
site terms absorb as much as they can and shrink r² as far as it will go, but log|V|
grows with θ and charges for it. The term with X is REML's correction.

## 5. Profiling: from four unknowns to two

Searching for four numbers at once is not necessary. For any given θ:

- the best bias has a formula, β̂ above;
- the best φ<sub>w</sub> has one too: φ̂<sub>w</sub>² = r² / (n − p).

Putting these into the expression of section 4 leaves a function of θ alone:

```math
d(\theta) = \log|V| \;+\; \log|X^T V^{-1} X| \;+\; (n-p)\left[1 + \log\frac{2\pi\, r^2}{n-p}\right]
```

This is the *profiled REML criterion*, which lme4 and MERA's code also call the
*deviance*. Smaller is better. The fit is now a search for the two numbers
θ<sub>E</sub> and θ<sub>S</sub> that make d(θ) smallest, and everything else follows from
them by formulas.

`criterion_from_definition` in the walkthrough script computes d(θ) exactly as written
here, by building V.

## 6. Computing the criterion without the big matrix

V has n rows and n columns. For the 5,218 records of MERA's example data that is 27
million entries, and for a million records it could not be stored at all. Two identities
replace V with a much smaller matrix,

```math
A = \Lambda^T Z^T Z \Lambda + I
```

which has a row and a column per *level* (691 for the example data) rather than per
record:

- log|V| = log|A| (Sylvester's determinant identity);
- V<sup>−1</sup> = I − ZΛA<sup>−1</sup>ΛᵀZᵀ (the Woodbury identity).

### What is in A

ZᵀZ is a table of counts. On its diagonal is the number of records of each level, and off
the diagonal the number of records that two levels share. For the five records of
section 3:

```
         [ 2  0  0  1  1 ]   event 0: 2 records, 1 at site 0 and 1 at site 1
         [ 0  2  0  1  1 ]   event 1: the same
 ZᵀZ =   [ 0  0  1  0  1 ]   event 2: 1 record, at site 1
         [ 1  1  0  2  0 ]   site 0: 2 records
         [ 1  1  1  0  3 ]   site 1: 3 records
```

A is this table with each entry multiplied by the θ of its row and the θ of its column,
plus 1 on the diagonal. With θ<sub>E</sub> = 2 and θ<sub>S</sub> = 3:

```
       [ 9  0  0   6   6 ]      2·2·2 + 1 = 9       2·3·1 = 6
       [ 0  9  0   6   6 ]
 A =   [ 0  0  5   0   6 ]      2·2·1 + 1 = 5
       [ 6  6  0  19   0 ]      3·3·2 + 1 = 19
       [ 6  6  6   0  28 ]      3·3·3 + 1 = 28
```

The counts do not depend on θ, so they are made once. In real data most entries are zero,
because most earthquakes are not recorded at most sites, and MERA only stores the others
(a *sparse* matrix). That is what lets it handle very large datasets.

### The Cholesky factorisation

Neither the determinant nor the inverse of A is computed directly. Instead A is written as

```math
A = L L^T
```

where L is *lower triangular*: all its entries above the diagonal are zero. This is the
Cholesky factorisation, a kind of square root of A. Only a *positive definite* matrix
has one. That is the matrix version of a positive number, and A always is one, thanks to
the "+ I" in its definition. The factorisation gives both things cheaply:

- log|A| is twice the sum of the logarithms of the diagonal entries of L;
- an equation A**g** = **h** is solved for the unknown **g** in two passes, one with L and
  one with Lᵀ, because equations with a triangular matrix can be solved one unknown at a
  time. This replaces multiplying by A<sup>−1</sup>.

### The recipe

The data enter only through three sets of sums, which are also made once: ZᵀZ, the sums of
**y** and of the columns of X within each level (Zᵀ[y X]), and their sums of products over
all records ([y X]ᵀ[y X]). Here [y X] is the matrix with **y** as its first column,
followed by the columns of X.

For each θ that the search tries:

1. Fill in A and factorise it. This gives log|A|.
2. Form R = ΛᵀZᵀ[y X]: the sums within each level, times the θ of the level.
3. Solve AW = R for W.
4. Form the small matrix M = [y X]ᵀ[y X] − RᵀW. By the Woodbury identity this is
   [y X]ᵀV<sup>−1</sup>[y X]. With a bias it has two rows and two columns:

   ```
   M = [ yᵀV⁻¹y   yᵀV⁻¹X ]
       [ XᵀV⁻¹y   XᵀV⁻¹X ]
   ```

5. Read off β̂ = (XᵀV<sup>−1</sup>X)<sup>−1</sup> XᵀV<sup>−1</sup>y, and
   r² = yᵀV<sup>−1</sup>y − (XᵀV<sup>−1</sup>y)ᵀβ̂. This is the r² of section 4,
   multiplied out.
6. Put log|A|, log|XᵀV<sup>−1</sup>X| and r² into the formula for d(θ).

`evaluate` in the walkthrough script is exactly these six steps, and the script prints
that it gives the same number as the definition in sections 4 and 5.

## 7. Searching for the best θ

d(θ) has no formula for its minimum, so it is found by trying values of θ in a sensible
order. MERA uses COBYQA, an optimiser in scipy that needs only the value of the function,
not its slope. Roughly, it works like this:

1. Evaluate d at a few points around the current best θ.
2. Fit a simple curved surface (a quadratic) through those values.
3. Step to the lowest point of that surface, but no further than a set distance, called
   the *trust-region radius*, because the surface is only trusted near the points it was
   fitted to.
4. If d improved, keep the new point. Shrink the radius as progress slows.
5. Stop when the radius falls below 10<sup>−8</sup>.

A fit usually takes 20 to 100 evaluations of d. Three details matter.

**The starting point** is θ = 1 for each factor, as in lme4.

**The bound.** A standard deviation cannot be negative, so θ is kept at or above zero. If
the best θ for a factor is exactly zero, the fit is called *singular*. It means the data
show no more variation between events (or between sites) than the remaining residuals
would produce by chance. MERA then reports a standard deviation of zero for that factor,
and all its terms are zero.

**The search variable.** COBYQA does not search over θ itself but over

```math
\omega = \log(1 + \theta^2)
```

for each factor, which is converted back afterwards. There are two reasons:

- Near zero, ω is θ². d depends on θ only through θ², which makes d flat at θ = 0, and an
  optimiser can stop there by mistake when the true optimum is a small positive value. In
  θ² there is no such flat spot.
- For large θ, ω is 2 log θ. Whether θ is 10 or 1,000, a step of a given size in ω then
  changes d by a similar amount, so the same step sizes work across the whole range.

If COBYQA runs out of evaluations before it converges, `run_mera` warns and names the IM.

## 8. The results at the best θ

With the best θ in hand, steps 1 to 5 of section 6 are done once more, and everything
that MERA reports comes from their results.

**The standard deviations.** φ̂<sub>w</sub> = √(r² / (n − p)) is `phi_w`. Since θ is
relative to it, `tau` = θ<sub>E</sub> φ̂<sub>w</sub> and `phi_S2S` = θ<sub>S</sub>
φ̂<sub>w</sub>. The total, `sigma`, is the square root of the sum of the three squares.

Note that `phi_w` is *not* the root-mean-square of the remaining residuals in
`rem_res_df`, which is smaller. r² is the sum of the squares of the remaining residuals
*plus* the sum of the squares of **u**, defined below (the "penalty" in its name). The
extra part allows for the scatter that the event and site terms have absorbed.

**The bias** is β̂, and its standard error (`bias_std_err`) is
φ̂<sub>w</sub> √((XᵀV<sup>−1</sup>X)<sup>−1</sup>).

**The event and site terms.** Solve

```math
A\,u = \Lambda^T Z^T (y - X\hat\beta)
```

for **u**, and then **b** = Λ**u** holds the event terms followed by the site terms.
(**u** is **b** with each term divided by the θ of its factor: the terms on a common
scale, on which they all have the standard deviation φ<sub>w</sub>.)
These are the simultaneous equations of section 2: the row for one event says that its
term is the shrunk average of (residual − bias − site term) over its records. Statistics
calls these estimates the *conditional modes* of the random effects, or BLUPs (best linear
unbiased predictors). They are `event_res_df` and `site_res_df`.

**Fitted values and remaining residuals.** The fitted value of a record is the bias plus
its event term plus its site term (Xβ̂ + Z**b**, `fit_df`), and its remaining residual is
the residual less the fitted value (`rem_res_df`).

**The conditional standard deviations** say how uncertain each event or site term is,
given the data. They are the square roots of the diagonal of

```math
\phi_w^2\; \Lambda A^{-1} \Lambda^T
```

so the one for a level is φ̂<sub>w</sub> × (the θ of its factor) × √(its diagonal entry
of A<sup>−1</sup>). With events only, this is the formula of section 2. They are
`event_cond_std_df` and `site_cond_std_df`. Two things to know about them:

- They treat the estimated bias and standard deviations as exact, as lme4 does, so they
  slightly understate the full uncertainty.
- Only the diagonal of A<sup>−1</sup> is needed. The inverse of a sparse matrix is not
  sparse, so MERA never forms it: a method called *selected inversion* computes just the
  entries that are needed from the Cholesky factor
  (`Factor::inverse_diagonal` in `crates/mera_lmm/src/factor.rs`).

## 9. Where each step is in the code

| Step | Walkthrough script (numpy) | MERA (Python) | MERA (Rust, `crates/mera_lmm/src/`) |
|---|---|---|---|
| Labels to level numbers; Z | `indicator_matrix` | `fit_lmm` in `mera/lmm.py`, step 1 | Z is never built |
| The counts and sums (section 6) | inside `evaluate` | | `Design::new` in `design.rs` |
| A for a given θ | inside `evaluate` | | `Design::fill_a` in `design.rs` |
| Cholesky factorisation, log-determinant, solve | `np.linalg` | | `Factor` in `factor.rs` |
| The criterion d(θ) (sections 5 and 6) | `evaluate` | `model.deviance(...)` | `CrossedModel::profile` in `model.rs` |
| The search (section 7) | `find_theta` | `fit_lmm`, step 3 | not in Rust |
| The results (section 8) | `estimates` | `fit_lmm`, step 4 | `CrossedModel::fit_at` in `model.rs` |
| Diagonal of A<sup>−1</sup> | `np.linalg.inv` | | `Factor::inverse_diagonal` in `factor.rs` |
| One fit per IM, masks, output tables | | `run_mera` in `mera/analysis.py` | |

The Rust code is commented for readers who do not know Rust, and
[`crates/mera_lmm/README.md`](../crates/mera_lmm/README.md) explains the Rust syntax it
uses.

## 10. Symbols

| Symbol | Meaning | In the code | In MERA's outputs |
|---|---|---|---|
| n | number of records in the fit | `n` | |
| p | number of fixed effects (1 with a bias, 0 without) | `p` | |
| q | number of levels: events plus sites | `q` | |
| **y** | the residuals of one IM | `y` | the input |
| X | fixed-effects matrix: a column of ones | `x` | |
| β | fixed effects: the bias | `beta` | `bias` |
| Z | indicator matrix of the levels of each record | `z` (walkthrough only) | |
| **b** | random effects: event terms, then site terms | `ranef` | `event_res_df`, `site_res_df` |
| ε | remaining residuals | `resid` | `rem_res_df` |
| τ | standard deviation of the event terms | `sd[0]` | `tau` |
| φ<sub>S2S</sub> | standard deviation of the site terms | `sd[1]` | `phi_S2S` |
| φ<sub>w</sub> | standard deviation of the remaining residuals | `sigma` | `phi_w` |
| σ | total standard deviation | | `sigma` |
| θ | τ / φ<sub>w</sub> and φ<sub>S2S</sub> / φ<sub>w</sub> | `theta` | |
| Λ | diagonal matrix with the θ of each level's factor | `lambda`, `lam` | |
| V | I + ZΛΛᵀZᵀ: the covariance of **y** divided by φ<sub>w</sub>² | built only by the walkthrough's `criterion_from_definition` | |
| A | ΛᵀZᵀZΛ + I | `values` (Rust), `a` | |
| L | Cholesky factor of A | `l_values` | |
| R | ΛᵀZᵀ[y X]: the sums within each level, times its θ | `r` | |
| W | A<sup>−1</sup>R | `w` | |
| M | [y X]ᵀV<sup>−1</sup>[y X], a small matrix | `m` | |
| **u** | the random effects on a common scale: **b** = Λ**u** | `u` (walkthrough) | |
| r² | penalised residual sum of squares | `r2` | |
| d(θ) | profiled REML criterion | `deviance` | |
| ω | log(1 + θ²), the search variable | `omega` | |

Mind two clashes of names.

- **sigma.** In the code, as in lme4, `sigma` is the standard deviation of the remaining
  residuals, φ<sub>w</sub>. In MERA's output table `bias_std_df`, the column `sigma` is
  the total standard deviation and `phi_w` is the code's `sigma`.
- **residual.** In ground-motion work, the residual is the input **y**. In statistics,
  and in `mera/lmm.py` and the Rust code, the "residual" is what the model leaves over:
  MERA's remaining residual. So the "residual standard deviation" is φ<sub>w</sub>.

## References

- Bates, D., Mächler, M., Bolker, B. and Walker, S. (2015). Fitting linear mixed-effects
  models using lme4. *Journal of Statistical Software* 67(1). The method that MERA
  implements; sections 2 to 5 of the paper cover sections 4 to 8 here in full generality.
- Bates, D. *lme4: Mixed-effects modeling with R*
  ([draft book](https://people.math.ethz.ch/~maechler/MEMo-pages/lMMwR.pdf)). Chapter 1 is
  a gentle introduction to random effects and their interpretation.
- Al Atik, L., Abrahamson, N., Bommer, J. J., Scherbaum, F., Cotton, F. and Kuehn, N.
  (2010). The variability of ground-motion prediction models and its components.
  *Seismological Research Letters* 81(5). The notation for the components of ground-motion
  variability.
- Takahashi, K., Fagan, J. and Chin, M.-S. (1973). Formation of a sparse bus impedance
  matrix and its application to short circuit study. *Proceedings of the 8th PICA
  Conference*. The selected inversion of section 8.
