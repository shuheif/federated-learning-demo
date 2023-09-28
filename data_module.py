from pytorch_lightning import LightningDataModule
from torch import manual_seed
from torch.utils.data import DataLoader, random_split

from dataset import DrivingDataset


class DrivingDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            city: str=["PIT", "ATX"],
            test_ratio: float=.1,
            val_ratio: float=.2,
            num_workers: int=4,
            seed: int=1234,
            client_id: int=0,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.cities = [city]
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.client_id = int(client_id)
        manual_seed(seed)

    def setup(self):
        full_dataset = DrivingDataset(
            train_or_test="train",
            av2_cities=self.cities,
            av2_root=self.data_dir,
        )
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_ratio)
        train_size = total_size - test_size
        self.train_dataset, self.test_dataset = random_split(full_dataset, [train_size, test_size])
        val_size = int(train_size * self.val_ratio)
        self.val_dataset, _ = random_split(self.train_dataset, [val_size, train_size - val_size])

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
