FEDREATED_DATASET_JT_VERSION = "v0.1 beta"
from torch.utils.data import DataLoader
# from nuscenes_dataset_jt import NuScenesDataset
from .nuscenes_dataset_jt import NuScenesDataset
from .av2_dataset_jt import AV2Dataset
from .crop_and_resize import crop_car_and_resize, crop_central_and_resize
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
from pathlib import Path
from itertools import compress
import cv2
import numpy as np

class YoutubewithPL(Dataset):
    def __init__(self,
                 dataset_path='/data/shared/images_jpeg_resize/pseudo_geoinput/',
                 cities=['PIT', 'WDC', 'MIA', 'ATX', 'PAO', 'DTW', 'BOS', 'SGP', "location_phx", "location_sf",
                         "location_other"], score_th=1e-2
                 ):
        self.cmd_bank = [1, 2, 3]
        self.rgb_transform = transforms.ToTensor()
        self.dataset_path = dataset_path
        self.image_path_list = []
        self.cmd_list = []
        self.speed_list = []
        self.city_list = []
        self.wps_list = []
        self.score_list = []
        self.city_map = {}
        for i in range(len(cities)):
            self.city_map[cities[i]] = i
        # videos_folder = os.listdir(self.dataset_path)
        for cmd in self.cmd_bank:
            cmd_root = dataset_path + str(cmd)
            for city in cities:
                city_cmd_root = cmd_root + '/' + city
                data_dir = Path(city_cmd_root).expanduser()
                for file in data_dir.glob('*'):
                    pickle_file = city_cmd_root + '/' + file.name
                    with open(pickle_file, 'rb') as handle:
                        data_dict = pickle.load(handle)
                    self.image_path_list.append(data_dict['img'])
                    self.cmd_list.append(data_dict['cmd'])
                    self.speed_list.append(data_dict['speed'])
                    self.wps_list.append(data_dict['wps'])
                    self.city_list.append(self.city_map[data_dict['city']])
                    self.score_list.append(data_dict['score'])
                    # if len(self.city_list)==100:
                    #     break
                print('command ', str(cmd), city, ' Done.')
                # break
            print('command ', str(cmd), ' Done.')
            # break



        print('load %d images in total' % (len(self.image_path_list)))

        scorenp = np.array(self.score_list)
        mask = list(scorenp <= score_th)
        self.image_path_list = list(compress(self.image_path_list, mask))
        self.cmd_list = list(compress(self.cmd_list, mask))
        self.speed_list = list(compress(self.speed_list, mask))
        self.wps_list = list(compress(self.wps_list, mask))
        self.city_list = list(compress(self.city_list, mask))

        print('After filtering %d images in total' % (len(self.image_path_list)))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        rgb_image = cv2.imread(self.image_path_list[idx])
        dim = (400, 225)
        rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)
        rgb_image = self.rgb_transform(rgb_image)

        rgb_image_name = self.image_path_list[idx]
        wps = self.wps_list[idx]
        cmd = self.cmd_list[idx]
        speed = self.speed_list[idx][0]
        city = self.city_list[idx]

        return rgb_image, rgb_image_name, wps, cmd, speed, city

class FederatedDatasetPL():
    def __init__(self, av2_cities=None, ns_cities=None, wm_cities=None, av2_root="/data/shared/av2_all/train",
                 ns_train_proportion=0.8, ns_train_test="TRAIN", waymo_train_test="TRAIN", waymo_p=1, youtube_root='/data/shared/images_jpeg_resize/pseudo_geoinput/'):
        self.PLdataset = YoutubewithPL(dataset_path=youtube_root)
        # self.PLdataset = None
        print("FederatedDataset: There are", len(self.PLdataset), "PL samples overall. ")
        if av2_cities is not None:
            self.set_av2(cities=av2_cities, dataset_dir=av2_root)
            # self.set_av2(cities=av2_cities, dataset_dir=av2_root, proportion=0.05)
            print("FederatedDataset: There are", len(self.av2_dataset), "av2 samples overall. ")
        else:
            self.av2_dataset = None

        if ns_cities is not None:
            self.set_nuscenes(cities=ns_cities, train_or_test=ns_train_test, train_proportion=ns_train_proportion)
            # self.set_nuscenes(cities=ns_cities, train_or_test=ns_train_test, train_proportion=0.1)
            print("FederatedDataset: There are", len(self.nuscenes_dataset), " ns samples overall. ")
        else:
            self.nuscenes_dataset = None

        if wm_cities is not None:
            if waymo_train_test == 'TRAIN':
                self.set_waymo(cities=wm_cities)
                # self.set_waymo(cities=wm_cities, proportion=0.05)
            else:
                self.set_waymo(cities=wm_cities, tfrecords_dir="/data/shared/waymo_test", proportion=waymo_p,
                               images_dir="/home/data/waymo_test")
                # self.set_waymo(cities=wm_cities, tfrecords_dir="/data/shared/waymo_test", proportion=0.05,
                #                images_dir="/home/data/waymo_test")
        else:
            self.waymo_dataset = None

        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def set_av2(self, cities, crop_and_resize=None, dataset_dir="/data/shared/av2_all/train", proportion=1.0):
        """
        @param dataset_dir:
            directories supported:
            "/data/shared/av2_all/train"
            "/data/shared/av2_all/mini_train"
        @param cities:
            ['PIT','WDC','MIA','ATX','PAO','DTW']
            default: empty
        @param proportion:
            the ratio of data you'd like to load: (0, 1]

        """

        self.av2_dataset = AV2Dataset(crop_and_resize=crop_and_resize, cities=cities, dataset_dir=dataset_dir,
                                      proportion=proportion)

    def set_nuscenes(self, cities, random_seed=99, train_or_test="TRAIN", train_proportion=1.0):
        """
        @param cities:
            ['BOS','SGP']
            default: empty

        """
        self.nuscenes_dataset = NuScenesDataset(cities=cities, random_seed=random_seed,
                                                train_or_test=train_or_test,
                                                train_proportion=train_proportion)
        # print("FederatedDataset: There are",self.__len__(),"samples overall. " )

    def set_waymo(self, cities, tfrecords_dir="/data/shared/waymo_val", proportion=1.0,
                  images_dir="/home/data/waymo_val"):
        from .waymo import WaymoDataset
        '''

        @param cities: ["location_phx","location_sf","location_other"]
        @param tfrecords_dir:
            supports "/data/shared/waymo_val" 
            "/data/shared/waymo_test" 
        @param images_dir:
            supports "/home/data/waymo_val"
            "/data/shared/waymo_test" 
        @param proportion: (0,1]
            default 1
        @return:
        '''

        self.waymo_dataset = WaymoDataset(cities=cities, proportion=proportion, tfrecords_dir=tfrecords_dir,
                                          images_dir=images_dir)
        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def __len__(self):

        length = 0
        if self.PLdataset is not None:
            length += len(self.PLdataset)
        if self.av2_dataset is not None:
            length += len(self.av2_dataset)

        if self.nuscenes_dataset is not None:
            length += len(self.nuscenes_dataset)

        if self.waymo_dataset is not None:
            length += len(self.waymo_dataset)

        return length

    def __getitem__(self, index):

        if index < 0:
            index = index + self.__len__()

        if index < 0 or index >= self.__len__():
            raise Exception("Index", index, "out of range")

        if self.PLdataset is not None:
            if index < len(self.PLdataset):
                return self.PLdataset[index]
            else:
                index = index - len(self.PLdataset)

        if self.av2_dataset is not None:
            if index < len(self.av2_dataset):
                return self.av2_dataset[index]
            else:
                index = index - len(self.av2_dataset)

        if self.nuscenes_dataset is not None:
            if index < len(self.nuscenes_dataset):
                return self.nuscenes_dataset[index]
            else:
                index = index - len(self.nuscenes_dataset)

        if self.waymo_dataset is not None:
            if index < len(self.waymo_dataset):
                return self.waymo_dataset[index]
            else:
                index = index - len(self.waymo_dataset)


if __name__ == "__main__":
    datasets = FederatedDatasetPL(av2_cities=['PIT'], ns_cities=None,av2_root="/data/shared/av2_all/mini_test")
    trn_loader = DataLoader(datasets, batch_size=1, num_workers=1, shuffle=True, drop_last=False,
                            pin_memory=True)
    for imgs,_,wps, cmds, speed, city_id in trn_loader:
        print(imgs.shape)
        print(wps.shape)
        print(cmds.shape)
        print(speed.shape)
        print(city_id.shape)


    # av2_cities = None, ns_cities = None, av2_root = "/data/shared/av2_all/train"