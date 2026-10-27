import unittest
import torch
from torch import nn
from sft_experiment import AdapterHead, batch, mask_labels, splits


class SFTTests(unittest.TestCase):
    def test_zero_adapter_matches_base(self):
        base = nn.Linear(6, 10, bias=False)
        head = AdapterHead(base)
        x = torch.randn(2, 3, 6)
        torch.testing.assert_close(head(x), base(x))

    def test_base_frozen(self):
        head = AdapterHead(nn.Linear(6, 10, bias=False))
        head(torch.randn(2, 3, 6)).sum().backward()
        self.assertIsNone(head.base.weight.grad)
        self.assertIsNotNone(head.up.weight.grad)

    def test_mask(self):
        self.assertEqual(mask_labels([1,2,3,4], [1,2]), [-100,-100,3,4])

    def test_invalid_boundary(self):
        with self.assertRaises(ValueError):
            mask_labels([1,2,3], [9])

    def test_padding_ignored(self):
        encoded = batch([([1,2], [-100,2]), ([3], [3])], 0)
        self.assertEqual(encoded['labels'][1,1], -100)
        self.assertEqual(encoded['attention_mask'][1,1], 0)

    def test_disjoint_splits(self):
        sets = [{tuple(row.values()) for row in part} for part in splits()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual([len(part) for part in splits()], [48,12,12])


if __name__ == '__main__':
    unittest.main()
