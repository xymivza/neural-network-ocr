import torch
import torch.quantization

class ModelQuantizer:
    def __init__(self, model):
        self.model = model
        
    def quantize_dynamic(self):
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized_model
        
    def quantize_static(self, calibration_data):
        self.model.eval()
        
        # Prepare for static quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate with data
        with torch.no_grad():
            for data, _ in calibration_data:
                self.model(data)
                
        # Convert to quantized model
        torch.quantization.convert(self.model, inplace=True)
        return self.model 