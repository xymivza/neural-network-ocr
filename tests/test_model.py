import unittest
import torch
from ocr_network import OCRNet, preprocess_image
import os

class TestOCRModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OCRNet().to(self.device)
    
    def test_model_output_shape(self):
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(self.device)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_model_forward_pass(self):
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        output = self.model(dummy_input)
        self.assertTrue(torch.is_tensor(output))

if __name__ == '__main__':
    unittest.main() 