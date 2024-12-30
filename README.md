# Neural Network OCR Project

A sophisticated Optical Character Recognition (OCR) system built with PyTorch, featuring advanced model optimization, data augmentation, and deployment capabilities.

## Features

### Core Functionality
- CNN-based neural network for digit recognition
- MNIST dataset training integration
- Custom image testing support
- Test image generation utility

### Advanced Features
- Model optimization through pruning and quantization
- Data augmentation pipeline
- Model performance benchmarking
- ONNX model export support
- Comprehensive visualization tools
- Docker containerization
- GPU acceleration support

## Project Structure

```
project_root/
├── data/                  # MNIST dataset storage
├── test_images/          # Directory for test images
├── ocr_network.py        # Main OCR implementation
├── test_ocr.py          # Testing script
├── generate_test_digit.py # Test image generator
├── requirements.txt      # Dependencies
└── ocr_model.pth        # Trained model (created after training)
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/avstiix/neural-network-ocr.git
cd neural-network-ocr
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main OCR network script to train the model:
```bash
python ocr_network.py
```

This will:
- Download the MNIST dataset (first run only)
- Train the neural network
- Save the trained model as 'ocr_model.pth'

### Generating Test Images

Generate sample digit images for testing:
```bash
python generate_test_digit.py
```

This creates test images (0-9) in the `test_images` directory.

### Testing the Model

Test the model with your own images:
```bash
python test_ocr.py test_images/digit_5.png
```

## Model Architecture

The OCR neural network consists of:
- 2 Convolutional layers
- Max pooling layers
- Dropout for regularization
- 2 Fully connected layers
- Output layer for 10 digits (0-9)

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- NumPy

See `requirements.txt` for specific versions.

## Limitations

- Currently only supports single digit recognition
- Best results with clear, centered digits
- Trained on MNIST dataset (handwritten digits only)
- Images should be similar to MNIST style for optimal results

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License

## Acknowledgments

- MNIST Dataset
- PyTorch Documentation