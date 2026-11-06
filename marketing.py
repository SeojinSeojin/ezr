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


def explain_holdout(rows, seed=591):
    train, test = split(rows, seed)
    return explain(Predictor(train), min(test, key=lambda r: r[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['summarize', 'groups', 'predict', 'explain'])
    parser.add_argument('--data', type=Path, default=data_path())
    parser.add_argument('--seed', type=int, default=591)
    args = parser.parse_args()
    try:
        rows = load(args.data)
        result = (explain_holdout(rows, args.seed) if args.command == 'explain' else
                  predict(rows, args.seed) if args.command == 'predict' else
                  summarize(rows) if args.command == 'summarize' else groups(rows))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
