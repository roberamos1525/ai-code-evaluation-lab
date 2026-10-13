# Statistics used for comparisons

All implementations are pure Python (no SciPy) and deterministic given a seed.

## pass@k

With `n` samples per task and `c` correct, the unbiased estimator of "at least one of k samples
passes" is `1 - C(n-c, k) / C(n, k)` (Chen et al., 2021). It is computed as
`1 - Π_{i=n-c+1}^{n} (1 - k/i)`, which avoids huge binomials; tests check it against the exact
form for a grid of `(n, c, k)`. Using `n > k` samples gives a lower-variance estimate than drawing
exactly `k`. The dataset-level value is the mean over tasks.

## Wilson interval

The run summary reports a Wilson score interval for the sample pass rate. Unlike the normal
approximation it stays inside [0, 1] and behaves sensibly at 0 or n successes.

## Paired bootstrap

Two runs over the same tasks are *paired*: per-task differences `candidate - baseline` (pass rate
and mean rubric score) are resampled with replacement (default 2000 resamples) and the percentile
interval of the mean difference is reported. An interval excluding 0 indicates a difference that
is unlikely to be explained by which tasks happen to be in the dataset. It does not account for
sampling noise *within* a task beyond what the per-task pass rate captures.

## McNemar's test

On the first sample of each task, `b` counts tasks that passed in the baseline and failed in the
candidate and `c` the reverse. Concordant tasks carry no information. With `b + c < 25` the
exact two-sided binomial test at p = 0.5 is used; otherwise the chi-square statistic with
continuity correction `(|b - c| - 1)^2 / (b + c)` (1 degree of freedom, p via `erfc`).

## Regression gate

Checks, each reported with observed value and threshold:

1. `dataset_version` — both runs evaluated the same task content (waivable).
2. `task_coverage` — the candidate covers every baseline task (waivable).
3. `pass@k` drop ≤ `max_pass_drop`. With `require_significance`, a larger drop is tolerated
   unless McNemar p < `alpha` or the bootstrap CI upper bound is below 0.
4. mean rubric score drop ≤ `max_score_drop`.
5. number of tasks that went from passing to failing (first sample) ≤ `max_new_failures`.

The verdict is `fail` if any check fails; the CLI then exits with status 1.
