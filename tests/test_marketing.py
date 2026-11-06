import csv
from pathlib import Path
import tempfile
import unittest

import marketing as m
from ezr import Tbl


def fixture(n=20):
    return [[i+2, *[float(i % 5) for _ in m.INPUTS], float(i % 2)] for i in range(n)]


class AdapterTests(unittest.TestCase):
    def test_roles(self):
        t = Tbl([m.NAMES, *fixture()])
        self.assertEqual([c.txt for c in t.x], list(m.INPUTS))
        self.assertEqual([c.txt for c in t.y], ['Response!'])
        self.assertNotIn(0, t.cols)

    def test_groups_ignore_response(self):
        rows = fixture()
        changed = [r[:-1]+[1-r[-1]] for r in rows]
        self.assertEqual(m.groups(rows), m.groups(changed))
        self.assertEqual(sum(g['rows'] for g in m.groups(rows)), len(rows))

    def test_split_and_reproducibility(self):
        rows = fixture()
        train, test = m.split(rows)
        self.assertFalse({r[0] for r in train} & {r[0] for r in test})
        self.assertEqual(len(train)+len(test), len(rows))
        self.assertEqual(m.predict(rows), m.predict(rows))
        self.assertNotEqual(m.split(rows, 1), m.split(rows, 2))
        self.assertEqual(m.predict(fixture(2))['contacts'], 1)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            m.predict(fixture(1))

    def test_train_only_and_no_target_distance(self):
        train, test = m.split(fixture())
        model = m.Predictor(train)
        income = model.table.x[0]
        self.assertAlmostEqual(income.mu, sum(r[1] for r in train)/len(train))
        flipped = m.Predictor([r[:-1]+[1-r[-1]] for r in train])
        self.assertEqual([d for d, _ in model.neighbors(test[0])],
                         [d for d, _ in flipped.neighbors(test[0])])
        query = test[0][:]
        before = model.neighbors(query)
        query[-1] = '?'
        self.assertEqual(before, model.neighbors(query))
        query[1] = 1e20
        model.score(query)
        self.assertAlmostEqual(income.mu, sum(r[1] for r in train)/len(train))
        self.assertEqual({r[0] for r in model.table.rows}, {r[0] for r in train})
        report = m.predict(fixture())
        self.assertEqual(report['random_contact_expected_precision'],
                         report['test_response_prevalence'])

    def test_explanation_matches_prediction(self):
        train, test = m.split(fixture())
        model = m.Predictor(train)
        row = test[0]
        report = m.explain(model, row)
        evidence = report['similar_training_customers']
        self.assertEqual(report['predicted_response_score'], model.score(row))
        self.assertEqual(report['predicted_response_score'],
                         sum(e['observed_response'] for e in evidence)/len(evidence))
        self.assertEqual(report, m.explain(model, row[:-1]+['?']))
        self.assertTrue({e['source_row'] for e in evidence} <= {r[0] for r in train})

    def test_invalid_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'data.csv'
            for value, message in [('2', 'RESPONSE'), ('nan', 'finite'), ('', 'numeric')]:
                with p.open('w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([*m.INPUTS, 'RESPONSE+'])
                    w.writerow([*[0 for _ in m.INPUTS], value])
                with self.assertRaisesRegex(ValueError, message):
                    m.load(p)

    def test_load_and_failures(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'data.csv'
            with p.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow([*(h+'+' for h in m.INPUTS), 'RESPONSE+'])
                w.writerows(r[1:] for r in fixture())
            self.assertEqual(m.load(p), fixture())
            with p.open('a') as f:
                f.write('bad\n')
            with self.assertRaisesRegex(ValueError, 'expected'):
                m.load(p)
            p.write_text('INCOMEX,RESPONSE+\n3,1\n')
            with self.assertRaisesRegex(ValueError, 'missing required'):
                m.load(p)
            with self.assertRaisesRegex(ValueError, 'cannot read dataset'):
                m.load(Path(d)/'absent')


if __name__ == '__main__':
    unittest.main()
