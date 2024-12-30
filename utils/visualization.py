import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE

class VisualizationTools:
    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        
    @staticmethod
    def visualize_feature_maps(model, input_image, layer_name):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        model.eval()
        handle = getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
        output = model(input_image)
        handle.remove()
        
        return activation[layer_name] 