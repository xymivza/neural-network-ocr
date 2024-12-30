from ocr_network import OCRNet, predict
import torch
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <path_to_image>")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = OCRNet().to(device)
    try:
        model.load_state_dict(torch.load('ocr_model.pth'))
        model.eval()
    except:
        print("Error: Could not find trained model (ocr_model.pth)")
        return

    image_path = sys.argv[1]
    try:
        result = predict(model, image_path, device)
        print(f"Predicted digit: {result}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 