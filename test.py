import torch
from transformers import ClapModel
import pytorch_lightning as pl

# Étape 1 — Charger le modèle Hugging Face
hf_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

# Étape 2 — Wrapping dans Lightning
class CLAPLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

pl_model = CLAPLightningModule(hf_model)

# Étape 3 — Sauvegarde en .ckpt
torch.save({'state_dict': pl_model.state_dict()}, "clap.ckpt")

print("✅ Fichier 'clap.ckpt' exporté avec succès.")
