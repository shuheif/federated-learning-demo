from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from dataset import CarlaDrivingDataset

TRANSFORM_CARLA_IMAGES = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
]

class DrivingDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            test_ratio: float=.1,
            val_ratio: float=.2,
            num_workers: int=4,
            seed: int=1234,
            client_id: int=0,
            num_clients: int=2,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.client_id = int(client_id)
        self.num_clients = num_clients

    def setup(self):
        full_dataset = CarlaDrivingDataset(data_dir=self.data_dir, transform=TRANSFORM_CARLA_IMAGES)
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_ratio)
        train_dataset, self.test_dataset = random_split(full_dataset, [total_size - test_size, test_size])
        train_size_per_client = (total_size - test_size) // self.num_clients
        train_start_idx = self.client_id * train_size_per_client
        self.train_dataset = Subset(train_dataset, range(train_start_idx, train_start_idx + train_size_per_client))
        val_size = int(train_size_per_client * self.val_ratio)
        _, self.val_dataset = random_split(self.train_dataset, [train_size_per_client - val_size, val_size])

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)
    
    def get_data_loaders(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
