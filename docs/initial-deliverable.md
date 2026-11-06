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

## Working initial ladder

The prototype uses customer demographics, spending, shopping activity and past campaigns from `Marketing_Analytics.csv`. Four commands run successive capabilities:

1. `python3.12 marketing.py summarize` validates records and reports missing values, customer means and response prevalence.
2. `python3.12 marketing.py groups` reuses EZR summaries and distances to produce four customer groups with readable profiles (ask 2).
3. `python3.12 marketing.py predict` predicts response from five similar training customers using a reproducible 80/20 split (ask 3).
4. `python3.12 marketing.py explain` shows the actual neighbors and responses behind a held-out prediction (parts of asks 11–12).

Response is excluded from similarity calculations, normalization uses training data only, and explanations do not consult the held-out outcome. The original CSV is preserved. Python 3.12+ is required; use `--data PATH` if the dataset is outside the default sibling MOOT checkout.

Seven tests passed. On 2,205 customers, the seed-591 demonstration achieved 62% precision among 50 recommended test customers, versus 16.1% expected for random contacts. This does not certify superiority, causation, retention or coupon effectiveness.

The remaining asks are future work. Two other domain lists, lecturer approval, team review of the evaluation claim and the final one-page hand-in remain outstanding.
