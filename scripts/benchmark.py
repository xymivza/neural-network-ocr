import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_network import OCRNet

def benchmark_inference(model, input_size=(1, 1, 28, 28), num_iterations=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        model(dummy_input)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time * 1000  # Convert to milliseconds

if __name__ == '__main__':
    model = OCRNet()
    avg_inference_time = benchmark_inference(model)
    print(f"Average inference time: {avg_inference_time:.2f}ms") 