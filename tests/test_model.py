import unittest
import torch
from ocr_network import OCRNet, preprocess_image
import os

class TestOCRModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OCRNet().to(self.device)
        # Set model to evaluation mode to avoid BatchNorm issues
        self.model.eval()
    
    def test_model_output_shape(self):
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(self.device)
        with torch.no_grad():  # Add no_grad context
            output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_model_forward_pass(self):
        # Use batch size > 1 to avoid BatchNorm issues
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(self.device)
        with torch.no_grad():  # Add no_grad context
            output = self.model(dummy_input)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (batch_size, 10))
    
    def test_single_image_inference(self):
        # Test single image inference in eval mode
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (1, 10))

if __name__ == '__main__':
    unittest.main() 