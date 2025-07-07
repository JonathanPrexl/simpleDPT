import unittest
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vit_dpt import TokenReshape
from modules.vit_tokens_to_spatial_features import TokenReshapeReverse


class TestReshapingOperations(unittest.TestCase):

    def test_flatten_and_unflatten_16x16(self):
        
        # Test case with image size 128 and patch size 16 hence
        # a 8x8 grid of patches

        tokens = torch.zeros(1, 768, 8, 8)
        for i in range(8):
            for j in range(8):
                tokens[:, :, i, j] = 10 * i + j
        
        reshaped = TokenReshape()(tokens)
        spatial_layout = TokenReshapeReverse(image_size=128,patch_size=16)(reshaped)

        self.assertTrue(torch.equal(tokens, spatial_layout))

    def test_flatten_and_unflatten_8x8(self):
        
        # Test case with image size 128 and patch size 16 hence
        # a 8x8 grid of patches

        tokens = torch.zeros(1, 768, 16, 16)
        for i in range(16):
            for j in range(16):
                tokens[:, :, i, j] = 100 * i + j
        
        reshaped = TokenReshape()(tokens)
        spatial_layout = TokenReshapeReverse(image_size=128,patch_size=8)(reshaped)
        self.assertTrue(torch.equal(tokens, spatial_layout))


if __name__ == '__main__':
    unittest.main()
