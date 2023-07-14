from pytorch_lightning import LightningModule, Trainer
from torch import optim, nn
from torchvision.models import resnet18, ResNet18_Weights

from utils import cross_entropy_for_onehot

NUM_CLASSES = 4
DATA_DIR = '' # Insert path to your dataset here


class ResNetClassifier(LightningModule):
    def __init__(self, pretrained: bool=True):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, NUM_CLASSES)
        self.loss_module = cross_entropy_for_onehot

    def forward(self, images):
        return self.resnet(images)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.resnet(images)
        loss = self.loss_module(preds, labels)
        self.log("train_loss", loss)
        return loss
    
    def _evaluate(self, batch, stage: str=None) -> None:
        images, labels = batch
        preds = self.resnet(images)
        loss = self.loss_module(preds, labels)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "test")
    
    def validation_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from data_module import DrivingDataModule
    data_module = DrivingDataModule(data_dir=DATA_DIR, batch_size=16)
    data_module.setup()
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    model = ResNetClassifier(pretrained=True)
    trainer = Trainer(max_epochs=3, fast_dev_run=True, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model=model, dataloaders=test_loader)
