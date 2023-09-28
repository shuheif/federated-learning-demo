import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append('/home/shuhei/Desktop/federated-learning-demo/datasets')
from datasets.federated_dataset_jt.federated_dataset_jt import FederatedDataset


class DrivingDataset(Dataset):
    def __init__(
            self,
            train_or_test: str,
            av2_cities: list = None,
            av2_root: str = None,
            ns_cities: list = None,
            wm_cities: list = None,
            train_portion: float = 0.8,
        ) -> None:
        super().__init__()
        self.dataset = FederatedDataset(
            av2_cities=av2_cities,
            av2_root=av2_root,
            ns_cities=ns_cities,
            ns_train_test=train_or_test.upper(),
            wm_cities=wm_cities,
            waymo_train_test=train_or_test.upper(),
            waymo_p=train_portion,
            ns_train_proportion=train_portion,
        )
        self.transform = transforms.Compose([
            transforms.Resize((32, 32), antialias=True),
            # transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        bgr_image, rgb_image_path_list, waypoints, command, speed, city_id = self.dataset[idx]
        rgb_image = bgr_image[[2, 1, 0], :, :]
        rgb_image = self.transform(rgb_image)
        one_hot_command = self._get_one_hot_command(command)
        return rgb_image.float(), one_hot_command.float()
    
    def _get_one_hot_command(self, command):
        if command == 1:
            return torch.tensor([1, 0, 0, 0]) # left
        elif command == 2:
            return torch.tensor([0, 1, 0, 0]) # forward
        else:
            return torch.tensor([0, 0, 1, 0]) # right


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser(description="Driving dataset")
    parser.add_argument("--data", type=str)
    args = parser.parse_args()
    dataset = DrivingDataset(
        train_or_test="train",
        av2_cities=["PIT", "ATX"],
        av2_root=args.data,
        ns_cities=None,
    )
    print('len(dataset): ', len(dataset))
    dataloader = DataLoader(dataset, batch_size=1)
    iterator = iter(dataloader)
    X, y = next(iterator)
    print('X: ', X.shape, ', y: ', y.shape)
    plt.imshow(X[0].permute(1, 2, 0))
    plt.show()
