import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import os
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class AugmentationConfig:
    rotation_degrees: int = 10
    translate: Tuple[float, float] = (0.1, 0.1)
    perspective_distortion: float = 0.2
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    gaussian_noise_prob: float = 0.2
    gaussian_blur_prob: float = 0.2

class DataAugmenter:
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.logger = self._setup_logger()
        self.transforms = self._create_transform_pipeline()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DataAugmenter')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _create_transform_pipeline(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomRotation(self.config.rotation_degrees),
            transforms.RandomAffine(
                degrees=0, 
                translate=self.config.translate,
                fill=255  # White background for MNIST-like images
            ),
            transforms.RandomPerspective(
                distortion_scale=self.config.perspective_distortion,
                fill=255
            ),
            transforms.ColorJitter(
                brightness=self.config.brightness_range,
                contrast=self.config.contrast_range
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(3)
            ], p=self.config.gaussian_blur_prob),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self._add_gaussian_noise(x)),
            transforms.ToPILImage()
        ])
    
    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.config.gaussian_noise_prob:
            noise = torch.randn_like(tensor) * 0.05
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    def augment_dataset(
        self, 
        input_dir: str, 
        output_dir: str, 
        samples_per_image: int = 5,
        supported_formats: List[str] = ['.png', '.jpg', '.jpeg']
    ) -> None:
        """
        Augment images from input directory and save to output directory.
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
            samples_per_image: Number of augmented versions per image
            supported_formats: List of supported image file extensions
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Processing images from {input_dir}")
            
            image_files = [
                f for f in os.listdir(input_dir) 
                if any(f.lower().endswith(fmt) for fmt in supported_formats)
            ]
            
            if not image_files:
                self.logger.warning(f"No supported images found in {input_dir}")
                return
            
            for filename in image_files:
                self._process_single_image(filename, input_dir, output_dir, samples_per_image)
                
            self.logger.info(f"Successfully augmented {len(image_files)} images")
            
        except Exception as e:
            self.logger.error(f"Error during dataset augmentation: {str(e)}")
            raise
    
    def _process_single_image(
        self, 
        filename: str, 
        input_dir: str, 
        output_dir: str, 
        samples_per_image: int
    ) -> None:
        try:
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('L')
            
            for i in range(samples_per_image):
                augmented = self.transforms(image)
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(filename)[0]}_aug_{i}.png"
                )
                augmented.save(output_path)
                
        except Exception as e:
            self.logger.error(f"Error processing {filename}: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    config = AugmentationConfig(
        rotation_degrees=15,
        translate=(0.15, 0.15),
        perspective_distortion=0.25
    )
    augmenter = DataAugmenter(config)
    augmenter.augment_dataset("input_images", "augmented_images") 