#!/usr/bin/env python3
"""Marketing adapter for EZR. Python 3.12+, standard library only."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random

from ezr import Tbl, mid, xdist
from ezr_eg import halve

# Explicit contact-time schema. Response is never an input.
INPUTS = (
    'INCOME',
    'KIDHOME',
    'TEENHOME',
    'RECENCY',
    'MNTWINES',
    'MNTFRUITS',
    'MNTMEATPRODUCTS',
    'MNTFISHPRODUCTS',
    'MNTSWEETPRODUCTS',
    'MNTGOLDPRODS',
    'NUMDEALSPURCHASES',
    'NUMWEBPURCHASES',
    'NUMCATALOGPURCHASES',
    'NUMSTOREPURCHASES',
    'NUMWEBVISITSMONTH',
    'ACCEPTEDCMP3',
    'ACCEPTEDCMP4',
    'ACCEPTEDCMP5',
    'ACCEPTEDCMP1',
    'ACCEPTEDCMP2',
    'COMPLAIN',
    'AGE',
    'CUSTOMER_DAYS',
    'MARITAL_DIVORCED',
    'MARITAL_MARRIED',
    'MARITAL_SINGLE',
    'MARITAL_TOGETHER',
    'MARITAL_WIDOW',
    'EDUCATION_2N CYCLE',
    'EDUCATION_BASIC',
    'EDUCATION_GRADUATION',
    'EDUCATION_MASTER',
    'EDUCATION_PHD',
    'MNTTOTAL',
    'MNTREGULARPRODS',
    'ACCEPTEDCMPOVERALL',
)
TARGET = 'RESPONSE'
NAMES = ['source_rowX', *INPUTS, 'Response!']


def data_path():
    """Override via --data or MARKETING_DATA; MOOT can name its root."""
    if os.getenv('MARKETING_DATA'):
        return Path(os.environ['MARKETING_DATA']).expanduser()
    return Path(os.getenv('MOOT', '../moot')).expanduser() / 'optimize/sales_data/Marketing_Analytics.csv'


def load(path):
    """Validate without fitting any statistics or modifying the source CSV."""
    path = Path(path).expanduser()
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            # Strip only goal/class suffixes, never EZR's ignored X suffix.
            header = [h.strip().rstrip('+-!').upper() for h in header]
            if len(set(header)) != len(header):
                raise ValueError('duplicate normalized column names')
            required = (*INPUTS, TARGET)
            missing = set(required) - set(header)
            if missing:
                raise ValueError(f'missing required columns: {sorted(missing)}')
            positions = [header.index(h) for h in required]
            rows = []
            for line, cells in enumerate(reader, 2):
                if len(cells) != len(header):
                    raise ValueError(f'row {line}: expected {len(header)} cells')
                row = [line]
                for name, at in zip(required, positions):
                    text = cells[at].strip()
                    if text in ('', '?', 'NA', 'NaN') and name != TARGET:
                        row.append('?')
                        continue
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise ValueError(f'row {line}: {name} must be numeric') from exc
                    if not math.isfinite(value):
                        raise ValueError(f'row {line}: {name} must be finite')
                    if name == TARGET and value not in (0, 1):
                        raise ValueError(f'row {line}: RESPONSE must be 0 or 1')
                    if name.startswith('MNT') and value < 0 and name != 'MNTREGULARPRODS':
                        raise ValueError(f'row {line}: {name} must be nonnegative')
                    row.append(value)
                rows.append(row)
    except OSError as exc:
        raise ValueError(f'cannot read dataset {path}: {exc.strerror}; use --data or MARKETING_DATA') from exc
    if not rows:
        raise ValueError('dataset contains no customer rows')
    return rows


def summarize(rows):
    table = Tbl([NAMES, *rows])
    return dict(rows=len(rows), response_prevalence=sum(r[-1] for r in rows)/len(rows),
                inputs=list(INPUTS), target=TARGET,
                missing={name: sum(r[i] == '?' for r in rows)
                         for i, name in enumerate(NAMES) if i},
                means={c.txt: mid(c) if c.n else None for c in table.x})


def groups(rows):
    """Four balanced descriptive groups; target excluded by EZR's x roles."""
    table = Tbl([NAMES, *rows])
    clusters = halve(table, stop=max(1, math.ceil(len(rows)/4)))
    profiles = []
    for i, cluster in enumerate(clusters, 1):
        means = {c.txt: mid(c) if c.n else None for c in cluster.x}
        profiles.append(dict(group=i, rows=len(cluster.rows),
                             profile=means,
                             description='Mean income {INCOME}; wine spend {MNTWINES}; '
                             'days since purchase {RECENCY}'.format(**means)))
    return profiles


def split(rows, seed=591):
    if len(rows) < 2:
        raise ValueError('prediction requires at least two customers')
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = min(len(rows)-1, max(1, int(0.8*len(rows))))
    return shuffled[:cut], shuffled[cut:]


class Predictor:
    """Unweighted k-nearest-neighbor response mean using train-only EZR scales."""
    def __init__(self, train, k=5):
        if not train or k < 1:
            raise ValueError('nonempty training rows and positive k required')
        self.table = Tbl([NAMES, *train])
        self.k = min(k, len(train))

    def neighbors(self, row):
        # Source row breaks distance ties, never a response label.
        return sorted(((xdist(self.table, row, r), r) for r in self.table.rows),
                      key=lambda pair: (pair[0], pair[1][0]))[:self.k]

    def score(self, row):
        return sum(r[-1] for _, r in self.neighbors(row))/self.k


def predict(rows, seed=591):
    train, test = split(rows, seed)
    model = Predictor(train)
    ranked = sorted(((model.score(r), r) for r in test),
                    key=lambda pair: (-pair[0], pair[1][0]))
    budget = min(50, len(test))
    return dict(seed=seed, train_size=len(train), test_size=len(test),
                contacts=budget,
                precision_at_budget=sum(r[-1] for _, r in ranked[:budget])/budget,
                overall_response_prevalence=sum(r[-1] for r in rows)/len(rows),
                test_response_prevalence=sum(r[-1] for r in test)/len(test),
                random_contact_expected_precision=sum(r[-1] for r in test)/len(test),
                random_baseline='Exact expected precision of a uniform sample without '
                                'replacement of the same contact budget from the test set; '
                                'equals test response prevalence, no Monte Carlo draws.',
                recommended_source_rows=[r[0] for _, r in ranked[:budget]],
                note='Working demonstration; no statistically certified superiority.')


def explain(model, row):
    # Blank target defensively: evidence cannot consult the query outcome.
    query = row[:-1] + ['?']
    neighbors = model.neighbors(query)
    score = model.score(query)
    evidence = [dict(source_row=r[0], distance=d, observed_response=r[-1],
                     income=r[1], wine_spend=r[1+INPUTS.index('MNTWINES')])
                for d, r in neighbors]
    yes = sum(r[-1] for _, r in neighbors)
    return dict(source_row=row[0], predicted_response_score=score,
                similar_training_customers=evidence, inputs=list(INPUTS),
                explanation=f'{int(yes)} of these {model.k} nearest training customers '
                            f'responded. Their unweighted response mean is {score:.3f}. '
                            'Similarity uses the listed contact-time inputs and '
                            'EZR distances scaled using training data only. '
                            'This is neighbor evidence, not a causal reason or a '
                            'calibrated probability.')


def explain_holdout(rows, seed=591, source_row=None):
    train, test = split(rows, seed)
    model = Predictor(train)
    if source_row is None:
        # Same ranking as predict; selection never consults test responses.
        row = min(test, key=lambda r: (-model.score(r), r[0]))
    else:
        row = next((r for r in test if r[0] == source_row), None)
        if row is None:
            raise ValueError(f'CSV row {source_row} is not in the held-out test set for seed {seed}')
    result = explain(model, row)
    result['selection'] = ('Highest-ranked held-out customer (ties use CSV row number).'
                           if source_row is None else 'Held-out customer selected with --row.')
    return result


def readable(command, result):
    """Present the same calculations in customer-facing language."""
    def number(value):
        return 'unknown' if value in (None, '?') else f'{value:,.1f}'

    if command == 'summarize':
        means = result['means']
        lines = ['CUSTOMER OVERVIEW',
                 f"Customers: {result['rows']:,}",
                 f"Responded to the observed campaign: {result['response_prevalence']:.1%}",
                 '', 'Typical customer (averages across known values):']
        for key, label in [('INCOME', 'Income'), ('AGE', 'Age (years)'),
                           ('RECENCY', 'Days since last purchase'),
                           ('MNTWINES', 'Wine spending'), ('MNTTOTAL', 'Total spending'),
                           ('NUMSTOREPURCHASES', 'Store purchases'),
                           ('NUMWEBPURCHASES', 'Web purchases'),
                           ('NUMCATALOGPURCHASES', 'Catalog purchases')]:
            lines.append(f'  {label}: {number(means[key])}')
        missing = {k: v for k, v in result['missing'].items() if v}
        lines += ['', 'Missing values: '+(', '.join(f'{k}: {v}' for k, v in missing.items())
                                         if missing else 'none in selected columns.'),
                  f"Prediction uses {len(result['inputs'])} customer features; only campaign response is the target.",
                  'Spending and income use the original dataset units.']
    elif command == 'groups':
        lines = ['CUSTOMER GROUPS',
                 'Each row describes the average customer in one similarity group.', '',
                 f"{'Group':<7}{'Customers':>10}{'Income':>13}{'Wine spend':>13}{'Days inactive':>15}"]
        for group in result:
            profile = group['profile']
            lines.append(f"{group['group']:<7}{group['rows']:>10,}"
                         f"{number(profile['INCOME']):>13}"
                         f"{number(profile['MNTWINES']):>13}"
                         f"{number(profile['RECENCY']):>15}")
        lines += ['', 'Compare income, wine spending and time since purchase to plan different messages.',
                  'Groups use customer features only; campaign responses do not determine membership.',
                  'These are descriptive groups, not proven marketing segments. Amounts use dataset units.']
    elif command == 'predict':
        n = result['contacts']
        responders = round(result['precision_at_budget'] * n)
        expected = result['random_contact_expected_precision'] * n
        lines = ['CAMPAIGN CONTACT DEMO',
                 f"Learned from {result['train_size']:,} customers; evaluated on {result['test_size']:,} held-out customers (seed {result['seed']}).",
                 '', f'If we contact the {n} highest-ranked test customers:',
                 f"  {responders} of {n} actually responded in the recorded campaign ({result['precision_at_budget']:.1%} precision).",
                 f'  Randomly contacting {n} test customers would find {expected:.1f} responders on average '
                 f"({result['random_contact_expected_precision']:.1%}).",
                 f"  Across the full dataset, {result['overall_response_prevalence']:.1%} responded.",
                 '', 'Random baseline = the exact expected result of uniform sampling without replacement;',
                 'it equals the response rate in the held-out test set.',
                 '', 'Recommended CSV row numbers, in ranked order:',
                 '\n'.join('  '+', '.join(map(str, result['recommended_source_rows'][i:i+10]))
                           for i in range(0, len(result['recommended_source_rows']), 10)),
                 'Row numbers locate records in the original CSV; they are not customer IDs.',
                 '', 'This is a historical demonstration, not a guarantee of future results or certified superiority.']
    else:
        evidence = result['similar_training_customers']
        yes = sum(e['observed_response'] for e in evidence)
        lines = [f"WHY THIS SCORE? — CSV row {result['source_row']}",
                 result.get('selection', 'Held-out customer.'),
                 f"Neighbor response score: {int(yes)}/{len(evidence)} = {result['predicted_response_score']:.1%}",
                 f'{int(yes)} of the {len(evidence)} most similar training customers responded.',
                 'This is a vote from similar past customers, not this customer’s chance of responding.',
                 'With five neighbors, scores change in 20-percentage-point steps.',
                 'Even 0% or 100% is not a guarantee.', '',
                 f"{'CSV row':<10}{'Distance':>10}{'Income':>13}{'Wine spend':>13}  Campaign response"]
        for e in evidence:
            lines.append(f"{e['source_row']:<10}{e['distance']:>10.3f}"
                         f"{number(e['income']):>13}{number(e['wine_spend']):>13}  "
                         + ('Responded' if e['observed_response'] else 'Did not respond'))
        lines += ['', 'Smaller distance means more similar across all customer input features.',
                  'Income and wine spending are shown as examples; they are not the only inputs.',
                  'The held-out customer’s own response was not used to explain or calculate this score.',
                  'This shows the evidence behind the score, not a causal reason to contact someone.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['summarize', 'groups', 'predict', 'explain'])
    parser.add_argument('--data', type=Path, default=data_path())
    parser.add_argument('--seed', type=int, default=591)
    parser.add_argument('--row', type=int, help='explain a specific held-out CSV row (default: highest-ranked)')
    parser.add_argument('--json', action='store_true', help='print full structured output instead of the readable report')
    args = parser.parse_args()
    if args.row is not None and args.command != 'explain':
        parser.error('--row is only available for explain')
    try:
        rows = load(args.data)
        result = (explain_holdout(rows, args.seed, args.row) if args.command == 'explain' else
                  predict(rows, args.seed) if args.command == 'predict' else
                  summarize(rows) if args.command == 'summarize' else groups(rows))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2) if args.json else readable(args.command, result))


if __name__ == '__main__':
    main()
