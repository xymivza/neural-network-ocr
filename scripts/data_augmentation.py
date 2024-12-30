import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import os

class DataAugmenter:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
        ])
    
    def augment_dataset(self, input_dir, output_dir, samples_per_image=5):
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                image = Image.open(image_path).convert('L')
                
                for i in range(samples_per_image):
                    augmented = self.transforms(image)
                    output_path = os.path.join(
                        output_dir, 
                        f"{os.path.splitext(filename)[0]}_aug_{i}.png"
                    )
                    transforms.ToPILImage()(augmented).save(output_path) 