import os
import torch
from fusion_service.neural_corrector import NeuralCorrector

os.makedirs("fusion_service/models", exist_ok=True)
model = NeuralCorrector()
torch.save(model.state_dict(), "fusion_service/models/neural_corrector.pth")
print("Saved default neural corrector model weights.")
