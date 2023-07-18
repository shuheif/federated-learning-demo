from pytorch_lightning import LightningModule, Trainer
from torch import optim, nn
from torchvision.models import resnet18, ResNet18_Weights

from utils import cross_entropy_for_onehot, accuracy

NUM_CLASSES = 4


class ResNetClassifier(LightningModule):
    def __init__(self, weights=None):
        super().__init__()
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, NUM_CLASSES)
        self.loss_module = cross_entropy_for_onehot

    def forward(self, images):
        return self.resnet(images)
    
    def _evaluate(self, batch, stage: str=None):
        images, labels = batch
        preds = self.resnet(images)
        loss = self.loss_module(preds, labels)
        acc = accuracy(preds, labels)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_accuracy", acc[0], prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._evaluate(batch, "train")
    
    def test_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "test")
    
    def validation_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    import argparse
    from data_module import DrivingDataModule
    parser = argparse.ArgumentParser(description="Train ResNetClassifier")
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    data_module = DrivingDataModule(data_dir=args.data_dir, batch_size=16)
    data_module.setup()
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    model = ResNetClassifier(weights=ResNet18_Weights.DEFAULT)
    trainer = Trainer(max_epochs=1, fast_dev_run=True, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model=model, dataloaders=test_loader)
