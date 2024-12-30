import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class OCRNet(nn.Module):
    def __init__(self):
        super(OCRNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the output
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers with dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

def preprocess_image(image_path):
    # Load and preprocess a single image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image_path, device):
    model.eval()
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    model = OCRNet().to(device)
    
    # Check if we're training or loading a pre-trained model
    try:
        model.load_state_dict(torch.load('ocr_model.pth'))
        print("Loaded pre-trained model")
    except:
        print("Training new model...")
        
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True,
            transform=transform,
            download=True
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=64,
            shuffle=True
        )
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        train_model(model, train_loader, criterion, optimizer, device)
        
        # Save the model
        torch.save(model.state_dict(), 'ocr_model.pth')
        print("Model trained and saved")

    # Example prediction
    # You can comment this out or modify the path to test your own images
    try:
        test_image_path = "test_digit.png"  # Replace with your image path
        prediction = predict(model, test_image_path, device)
        print(f"Predicted digit: {prediction}")
    except Exception as e:
        print(f"Couldn't make prediction: {str(e)}")

if __name__ == '__main__':
    main()