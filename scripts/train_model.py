import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_network import OCRNet, train_model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='../data', 
        train=True,
        transform=transform,
        download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Initialize model and training components
    model = OCRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'best_ocr_model.pth')

if __name__ == '__main__':
    main() 