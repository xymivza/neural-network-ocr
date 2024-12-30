import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class OCRNet(nn.Module):
    def __init__(self):
        super(OCRNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(-1, 128 * 7 * 7)
        
        x = self.dropout(torch.relu(self.fc_bn(self.fc1(x))))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_ocr_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def predict(model, image_path, device):
    model.eval()
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = OCRNet().to(device)
    
    try:
        checkpoint = torch.load('best_ocr_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pre-trained model")
    except:
        print("Training new model...")
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True,
            transform=get_train_transforms(),
            download=True
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        train_model(model, train_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()