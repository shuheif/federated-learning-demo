import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


import random
import cv2
from pathlib import Path
import os
import numpy as np
import albumentations as A

OFFSET = 6.5

# dir='/data/jimuyang/selfd_rebuttal/LearningByCheating_099_selfd/dataset/train/nocrash_Town01_regular'
# ''/data/shared/carla_data/nocrash_Town02_config7_ed7','
#''/data/shared/carla_data/town2_ed7_noisy','

class ImageDataset(Dataset):
    def __init__(self,
                 # dataset_path='/home/exx/jimuyang/data_collector/data_collection_120721/dataset_rebuttal/',
                 # dataset_path ='/data/jimuyang/selfd_rebuttal/LearningByCheating_099_selfd/dataset/',
                 # dataset_path = '/data/jimuyang/dataset_lbc_0913_020122/',
                 # dataset_path_list = ['/home/exx/jimuyang/LearningByCheating_0913_updated/dataset/train/nocrash_Town01_regular','/data/shared/carla_data/nocrash_Town02_config7_ed7'],
                 dataset_path_list=[
                     '/home/exx/jimuyang/LearningByCheating_0913_updated/dataset/train/nocrash_Town01_dense',
                     '/home/exx/jimuyang/LearningByCheating_0913_updated/dataset/train/nocrash_Town01_regular',
                     '/data/shared/carla_data/Town02_left',
                    '/data/shared/carla_data/Town02_left_turn',
                   '/data/shared/carla_data/Town02_left_turn2'],
                 # dataset_path_list=[
                 #     # '/home/exx/jimuyang/LearningByCheating_0913_updated/dataset/train/nocrash_Town01_dense',
                 #     '/home/exx/jimuyang/LearningByCheating_0913_updated/dataset/train/nocrash_Town01_regular',
                 #     '/data/shared/carla_data/Town02_left',
                 # #    '/data/shared/carla_data/Town2ed10',
                 # # '/data/shared/carla_data/town2_ed7_nonoisy'],

                 city_list = [0,0,1,1,1],
                 # dataset_path = '/home/exx/jimuyang/LearningByCheating_0913_updated/dataset/',
                 # dataset_path = '/data/jimuyang/selfd_rebuttal/dataset_lbc_0910_weather/',
                 # dataset_path = '/data/'
                 # second_town_path
                 _set='train',  # or val
                 gap=5,
                 batch_aug=1,
                 # augment_strategy=None,
                 batch_read_number=819200,

                 ):
        self.rgb_transform = transforms.ToTensor()
        self.batch_aug = batch_aug

        # self.dataset_path = dataset_path

        # print("augment with ", augment_strategy)
        # if augment_strategy is not None and augment_strategy != 'None':
        #     self.augmenter = getattr(augmenter, augment_strategy)
        # else:
        #     self.augmenter = None

        self.file_map = {}
        self.idx_map = {}
        self.townid = {}
        count = 0

        self.carla_path_list = dataset_path_list

        for i, data_path in enumerate(dataset_path_list):
            carla_path = data_path
        # self.carla_data_path = self.carla_path / 'nocrash_Town01_dense' / _set
        # self.carla_data_path = self.carla_path / _set /'nocrash_Town01_dense'
        # self.carla_data_path = self.carla_path / 'nocrash_Town01_regular'

        # self.carla_videos = os.listdir(self.carla_data_path)
            self.carla_videos = os.listdir(carla_path)



            if _set == 'train':
                self.img_transform = A.Compose([
                    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
                    A.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.1, ),
                    A.CoarseDropout(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.GaussianBlur(p=0.5, blur_limit=(3, 15)),
                    A.ISONoise(p=0.5)
                ])
            else:
                self.img_transform = None

            self.augmentation_prob = 0.5

            for carla_video in self.carla_videos:
                carla_lbs_path = data_path + '/'+carla_video + '/annotation'
                carla_lbs = os.listdir(str(carla_lbs_path))
                carla_lbs.sort()
                # for k in range(len(carla_lbs)-50):
                for k in range(len(carla_lbs) - 20):
                    carla_lb = carla_lbs[k]
                    carla_sample = np.load(str(data_path +'/'+ carla_video + '/annotation/' + carla_lb),
                                           allow_pickle=True).tolist()

                    carla_sample_rgb = str(data_path +'/'+ carla_video +'/'+ 'rgb' +'/'+ (carla_lb.replace('npy', 'png')))

                    carla_wpts = np.array(carla_sample['waypoints'])
                    # carla_wpts[:, 1] = carla_wpts[:, 1] - 20
                    carla_wpts[:, 1] = carla_wpts[:, 1] - 20
                    carla_wpts = carla_wpts / 20

                    if not ((carla_wpts <= 1).all() and (carla_wpts >= -1).all()):
                        continue
                    if carla_sample['speed']<0.00001:
                        continue

                    self.file_map[count] = [carla_sample, carla_sample_rgb]
                    # self.idx_map[count] = None
                    self.townid[count] = city_list[i]


                    count += 1

            print("Finished loading %s. Length: %d" % (data_path, count))
        self.batch_read_number = batch_read_number

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):
        sample_info, rgb_img_path = self.file_map[idx]
        rgb_image = cv2.imread(rgb_img_path)
        rgb_image = cv2.resize(rgb_image, (400, 225), interpolation=cv2.INTER_AREA)

        # I added the image augmentation
        if self.img_transform != None:
            if random.random() > self.augmentation_prob:
                transformed = self.img_transform(image=rgb_image)
                rgb_image = transformed["image"]

        # if self.augmenter:
        #     rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in
        #                   range(self.batch_aug)]
        # else:
        rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]

        locations_vehicle = sample_info['waypoints']
        cmd = sample_info['command']
        speed = sample_info['speed']

        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])

        return rgb_images, rgb_img_path, np.array(locations_vehicle), cmd, speed, self.townid[idx]

if __name__ == '__main__':
    train_data = ImageDataset()
    print('haha')