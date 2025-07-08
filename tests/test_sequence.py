import os
import unittest
import tempfile
import numpy as np

from proteinshake.datasets import RCSBDataset
from proteinshake.transforms import IdentityTransform


class TestSequenceRepresentation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.dataset = RCSBDataset(root=cls.tmpdir.name, verbosity=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_sequence_strings(self):
        seq_ds = self.dataset.to_sequence(transform=IdentityTransform())
        strings = seq_ds.strings()

        # Check type and content
        self.assertIsInstance(strings, list)
        self.assertGreater(len(strings), 0)
        self.assertTrue(all(isinstance(s, str) for s in strings))
        self.assertTrue(all(len(s) > 0 for s in strings))

    def test_sequence_tokens(self):
        seq_ds = self.dataset.to_sequence(transform=IdentityTransform())
        tokens = seq_ds.numpy()

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, np.ndarray) for t in tokens))
        self.assertTrue(all(t.ndim == 1 for t in tokens))
        self.assertTrue(all(np.issubdtype(t.dtype, np.integer) for t in tokens))

    def test_string_token_length_match(self):
        seq_ds = self.dataset.to_sequence(transform=IdentityTransform())
        strings = seq_ds.strings()
        tokens = seq_ds.numpy()

        for s, t in zip(strings, tokens):
            self.assertEqual(len(s), len(t))


if __name__ == '__main__':
    unittest.main()