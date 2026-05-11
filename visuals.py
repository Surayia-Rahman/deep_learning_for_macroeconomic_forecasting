import pytorch_lightning as pl
import torch

class TFT_Wrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x).prediction
        loss = self.model.loss(out, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x).prediction
        loss = self.model.loss(out, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
