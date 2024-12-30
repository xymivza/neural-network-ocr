import torch
import torch.nn.utils.prune as prune

class ModelPruner:
    def __init__(self, model):
        self.model = model
        
    def structured_pruning(self, amount=0.3):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=2,
                    dim=0
                )
                
    def magnitude_pruning(self, amount=0.3):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=amount
                )
                
    def remove_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.remove(module, 'weight') 