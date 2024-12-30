import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_network import OCRNet, train_model
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train OCR Model')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OCRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Reference to the training setup from ocr_network.py
    """python:ocr_network.py
    startLine: 140
    endLine: 153
    """
    
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

if __name__ == '__main__':
    main() 