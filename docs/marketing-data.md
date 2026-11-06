# Marketing data and methods

Source: MOOT `optimize/sales_data/Marketing_Analytics.csv`, inspected at MOOT commit
`90803be51b00f881305db45aa0cf6a3b5340804f`.
SHA-256: `e3bfe25dc6f7a3d7e48a7ba993744705dd454ba48c1e5449522f2a48122ca7ec`.
No rows are filtered and the original is preserved. Observed load: 2,205 rows,
15.102% response prevalence. No redistribution or original provenance claim is made.

The in-memory prepared view uses the explicit INPUTS tuple in marketing.py and
`Response!` as its sole target. Header matching strips `+`, `-`, `!`, normalizes
case, and does **not** strip `X`: an ignored source feature cannot silently become
an input. All selected source fields are numeric, including binary indicators.
`Z_COSTCONTACT` and `Z_REVENUE` are excluded constants; unselected extra columns
are ignored. Source CSV line number is `source_rowX`, ignored by EZR, used only
for deterministic tie-breaking and evidence lookup; it is not a customer identifier.
The original marketing file has no customer ID.

Blank, ?, NA and NaN input cells become EZR `?`, whose distance gap is 1.
Missing/bad targets, nonfinite numeric values, malformed rows, duplicate normalized
headers and absent required columns fail clearly. Negative spending fails except
MNTREGULARPRODS, which has negative values in this prepared source and is retained
as supplied. Validation is structural, not an anomaly detector or a full audit of
plausible ranges. No learned imputation, feature selection or derived transformations.
Summaries report means and missing counts for all inputs.

Groups use EZR recursive halving (`halve`, `project`, `xdist`, `clone`) into about
four balanced partitions. Profiles use input means and readable income, wine and
recency descriptions. Response never enters their distances or profiles. These
are descriptive full-data groups; they are not fed into held-out prediction.

Prediction shuffles a copy with a local seeded RNG (591), takes floor(80%) for
training, clamped to leave at least one row in each split, and requires >=2 rows.
Only training rows enter Tbl and its mean/SD logistic normalization. Five nearest
training rows (or all if fewer) vote by unweighted response mean. Distances use
only x columns. All inputs are assumed available before the contact; the dataset
does not establish this timing, so deployment requires a timing audit. Past campaign
fields are historical predictors, never the current Response. There is no future
retention target. Correlated spending totals remain separate weighted dimensions.

Ranking sorts descending score then ascending source line. Neighbor ties sort by
ascending source line. Precision is measured on the top min(50,test size) rows.
The random baseline is the exact expected precision of a uniform sample without
replacement of that same size from the test set: test response prevalence. There
are no random-baseline draws or uncertainty estimates. Overall prevalence is also
reported, separately. With fewer than 50 test rows all are selected.

Explain uses the lowest-source-line held-out customer, blanks their true response,
and returns the actual neighbors, distances, observed training labels and their
mean. It does not reveal the held-out outcome. Scores are not calibrated probabilities
or causal explanations. The full-data groups and summary are not training transforms.

Verified seed-591 demonstration: 1,764 training / 441 test, precision@50 0.62,
expected random precision 0.1609977324. These are observed demonstration results,
not pre-registered experimental evidence or statistically certified superiority.
