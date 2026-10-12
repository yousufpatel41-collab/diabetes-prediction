import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train_all import load_data


class SourceTests(unittest.TestCase):
    def test_pinned_source_and_conversion(self):
        manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
        for name,key in [('diabetes.arff','raw_sha256'),('diabetes.csv','csv_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest(),manifest[key])
        X,y=load_data()
        self.assertEqual(len(X),768)
        self.assertEqual(int(y.sum()),268)
        self.assertEqual(X.shape[1],8)

    def test_physiologically_missing_zeros_preserve_valid_zeros(self):
        X,y=load_data()
        self.assertGreater(int((X.Pregnancies==0).sum()),0)
        for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
            self.assertFalse((X[col]==0).any())
            self.assertGreater(int(X[col].isna().sum()),0)


if __name__=='__main__': unittest.main()
