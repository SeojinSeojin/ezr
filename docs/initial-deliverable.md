# Initial deliverable: marketing ladder

Sam, a grocery and wine retailer’s marketing manager, wants to spend a limited contact budget on customers likely to respond. The marketing requirements contain 20 asks, organized into ten skills:

| Skill | Customer tasks |
|---|---|
| Remember | Identify loyal, frequent buyers who do not complain; group customers into useful types. |
| Guess | Predict campaign response and new customers’ purchases across shopping channels. |
| Certify | Check whether teenagers affect shopping patterns and wine buyers respond more often. |
| Flow | Recommend coupons for new sign-ups and detect when advice stabilizes. |
| Blame | Find patterns among nonresponders and frequent website visitors who rarely buy. |
| Justify | Explain contact decisions and show similar customers’ responses. |
| Choose | Find likely responders with few response lookups, including families with children. |
| Fix | Find small differences between inactive customers and responders, and between discount and full-price buyers. |
| Spot | Detect suspicious records and spending unusual for similar customers. |
| Race | Compare recommendations with contacting the richest customers or those with high spending and past acceptance. |

## Running the initial ladder

Python 3.12+; default seed: 591. Dataset: `../moot/optimize/sales_data/Marketing_Analytics.csv` (override with `--data PATH`). Results below are output excerpts.

### 1. Summarize

**Command**

```sh
python3.12 marketing.py summarize
```

**Result**

```text
Customers: 2,205
Responded to the observed campaign: 15.1%
```

**Interpretation:** About 15 in 100 customers responded to the recorded campaign.

### 2. Group customers — ask 2

**Command**

```sh
python3.12 marketing.py groups
```

**Result**

```text
Group   Customers       Income   Wine spend  Days inactive
1             551     29,102.5         30.6           53.9
2             551     42,504.4        102.1           40.9
3             551     75,236.3        678.7           48.3
4             552     59,630.7        413.0           52.9
```

**Interpretation:** Group 3 has the highest average income and wine spending; group numbers are not rankings.

### 3. Predict response — ask 3

**Command**

```sh
python3.12 marketing.py predict
```

**Result**

```text
Learned from 1,764 customers; evaluated on 441 held-out customers (seed 591).
If we contact the 50 highest-ranked test customers:
  31 of 50 actually responded in the recorded campaign (62.0% precision).
  Randomly contacting 50 test customers would find 8.0 responders on average (16.1%).
```

**Interpretation:** The selected 50 contain 31 recorded responders, versus about 8 expected randomly; this single historical split does not certify superiority.

### 4. Explain scores — parts of asks 11–12

Three examples illustrate different levels of neighbor support. `--row` identifies an original CSV row.

**Command**

```sh
python3.12 marketing.py explain --row 403
```

**Result**

```text
Neighbor response score: 5/5 = 100.0%
```

**Interpretation:** All five similar training customers responded, giving this customer high recommendation support.

**Command**

```sh
python3.12 marketing.py explain --row 19
```

**Result**

```text
Neighbor response score: 2/5 = 40.0%
```

**Interpretation:** Two similar training customers responded and three did not, giving mixed support.

**Command**

```sh
python3.12 marketing.py explain --row 8
```

**Result**

```text
Neighbor response score: 0/5 = 0.0%
```

**Interpretation:** None of the five similar training customers responded, giving low recommendation support.

Different customer features retrieve different neighbors, whose responses determine the score. The customer's own response is excluded; 0% and 100% are neighbor averages, not guaranteed outcomes or causal explanations.

### 5. Verify

**Command**

```sh
python3.12 -m unittest discover -s tests -v
```

**Result**

```text
Ran 8 tests
OK
```

**Interpretation:** Checks for data handling, leakage prevention, reproducibility and explanation consistency passed.

Remaining: other asks, two domain lists, lecturer approval, team review of the evaluation claim and the final one-page hand-in.
