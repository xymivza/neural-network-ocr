import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_network import OCRNet

def export_to_onnx(model_path='ocr_model.pth', output_path='ocr_model.onnx'):
    model = OCRNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")

if __name__ == '__main__':
    export_to_onnx() 