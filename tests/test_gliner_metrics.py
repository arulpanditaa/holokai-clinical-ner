"""Pure-Python smoke checks of the preserved GLiNER evaluation logic."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'gliner'))
from evaluate_gliner import compute_metrics
from label_schemas import get_labels, source_label_map

class TestNERMetrics(unittest.TestCase):
    def test_exact_span_micro_f1(self):
        gold = [[(0, 0, 'medication')], [(1, 1, 'procedure')]]
        predicted = [[(0, 0, 'medication')], [(1, 1, 'diagnosis')]]
        summary = compute_metrics(gold, predicted)
        self.assertEqual(summary['overall']['tp'], 1)
        self.assertEqual(summary['overall']['fp'], 1)
        self.assertEqual(summary['overall']['fn'], 1)
        self.assertAlmostEqual(summary['overall']['f1'], 0.5)

    def test_11_labels_and_recommended_mapping(self):
        self.assertEqual(len(get_labels('original')), 11)
        self.assertEqual(len(get_labels('recommended')), 11)
        self.assertEqual(source_label_map('recommended')['MEDICATION'],
                         'medication, drug, or prescription')

if __name__ == '__main__':
    unittest.main()
