from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import math
import random

#import augmenter

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
from pathlib import Path
import os
import numpy as np

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import math

#import albumentations as A

nuscenes_ignore = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313,
                   314, 419]  # 419 is not in canbus blacklist but it's speed is not okay

# Boston_wpts = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
#     104, 105, 106, 107, 108, 109, 110, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 199, 200, 202,
#     203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 218, 219, 220, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
#     234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
#     259, 260, 261, 262, 263, 264, 321, 323, 324, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
#     517, 518, 552, 553, 554, 555, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 681, 683, 684, 685,
#     686, 687, 688, 689, 328, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 329, 330, 331, 332, 519, 520, 521, 522,
#     523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545, 546, 739, 740,
#     741, 744, 746, 747, 749, 750, 751, 752, 757, 758, 759, 760, 761, 762, 763, 764, 765, 767, 768, 769, 770, 771, 775, 777, 778,
#     780, 781, 782, 783, 784, 695, 696, 697, 698, 700, 701, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716,
#     717, 718, 719, 726, 727, 728, 730, 731, 733, 734, 735, 736, 737, 738]

# Boston_score = [283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304,
#     305, 306, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 440, 441, 442,
#     443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 471,
#     472, 474, 475, 476, 477, 478, 479, 480, 566, 568, 570, 571, 572, 573, 574, 575, 576, 577, 578, 580, 582, 583, 584, 585, 586, 587,
#     588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 625, 626, 627, 629, 630, 632, 633, 634, 635, 636, 637, 638, 639,
#     640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 803, 804, 805, 806, 808, 809, 810, 811, 812, 813, 815, 816, 817,
#     819, 820, 821, 822, 868, 869, 870, 871, 872, 873, 875, 876, 877, 878, 880, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892,
#     893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915 ]

Boston = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
          103,
          104, 105, 106, 107, 108, 109, 110, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176,
          199, 200, 202,
          203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 218, 219, 220, 222, 224, 225, 226, 227, 228, 229, 230,
          231, 232, 233,
          234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
          256, 257, 258,
          259, 260, 261, 262, 263, 264, 321, 323, 324, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512,
          513, 514, 515,
          517, 518, 552, 553, 554, 555, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 681,
          683, 684, 685,
          686, 687, 688, 689, 328, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 329, 330, 331, 332, 519,
          520, 521, 522,
          523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545,
          546, 739, 740,
          741, 744, 746, 747, 749, 750, 751, 752, 757, 758, 759, 760, 761, 762, 763, 764, 765, 767, 768, 769, 770, 771,
          775, 777, 778,
          780, 781, 782, 783, 784, 695, 696, 697, 698, 700, 701, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713,
          714, 715, 716,
          717, 718, 719, 726, 727, 728, 730, 731, 733, 734, 735, 736, 737, 738, 283, 284, 285, 286, 287, 288, 289, 290,
          291, 292, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 388, 389, 390, 391, 392, 393, 394, 395, 396,
          397, 398, 556,
          557, 558, 559, 560, 561, 562, 563, 564, 565, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452,
          453, 454, 455,
          456, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 471, 472, 474, 475, 476, 477, 478, 479, 480, 566,
          568, 570, 571,
          572, 573, 574, 575, 576, 577, 578, 580, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
          596, 597, 598,
          599, 600, 625, 626, 627, 629, 630, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
          647, 648, 649,
          650, 651, 652, 803, 804, 805, 806, 808, 809, 810, 811, 812, 813, 815, 816, 817, 819, 820, 821, 822, 868, 869,
          870, 871, 872,
          873, 875, 876, 877, 878, 880, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897,
          898, 899, 900,
          901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915]

Singapore = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32,
             33, 34, 3, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 35, 36, 38, 39,
             41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 120, 121, 122, 123, 124, 125, 126, 127, 128,
             129, 130, 131, 132, 133, 134,
             135, 138, 139, 149, 150, 151, 152, 154, 155, 157, 158, 159, 160, 190, 191, 192, 193, 194, 195, 196, 177,
             178, 179, 180, 181, 182, 183,
             184, 185, 187, 188, 315, 316, 317, 318, 221, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 360,
             361, 362, 363, 364, 365, 366,
             367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 399,
             400, 401, 402, 403, 405, 406,
             407, 408, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
             429, 430, 431, 432, 433, 434,
             435, 436, 437, 438, 439, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 858, 860, 861, 862, 863, 864,
             865, 866, 916, 917, 919, 920,
             921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 786, 787, 789, 790, 791, 792, 794, 795, 796, 797,
             798, 799, 800, 802, 945, 947,
             949, 952, 953, 955, 956, 957, 958, 959, 960, 961, 962, 963, 966, 967, 968, 969, 971, 972, 975, 976, 977,
             978, 979, 980, 981, 982, 983,
             984, 988, 989, 990, 991, 992, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
             1008, 1009, 1010, 1011,
             1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1044, 1045, 1046, 1047,
             1048, 1049, 1050, 1051,
             1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069,
             1070, 1071, 1072, 1073,
             1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091,
             1092, 1093, 1094, 1095,
             1096, 1097, 1098, 1099, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109, 1110]

Singapore_val = [17, 351, 36, 191, 188, 221, 278, 370, 401, 851, 789, 961, 981, 1010, 1098]

OFFSET = 6.5

NUSC_CAN = NuScenesCanBus(dataroot='/data/jimuyang/nuscenes/')
NUSC = NuScenes(version='v1.0-trainval', dataroot='/data/jimuyang/nuscenes/', verbose=True)


class ImageDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/jimuyang/nuscenes/',
                 dataset_version='v1.0-trainval',
                 _set='train',  # or val
                 gap=5,
                 batch_aug=1,
                 augment_strategy=None,
                 batch_read_number=819200,

                 ):
        self.rgb_transform = transforms.ToTensor()
        self.batch_aug = batch_aug

        self.dataset_path = dataset_path

        self.boston_trainset = []
        self.singapore_valset = []
        self.singapore_testset = []

        for _scene in NUSC.scene:
            _ind = int(_scene['name'].split('-')[-1].lstrip('0'))
            if _ind in Boston:
                self.boston_trainset.append(_scene)
            if _ind in Singapore:
                self.singapore_testset.append(_scene)
            if _ind in Singapore_val:
                self.singapore_valset.append(_scene)

        print("Boston Trainset: %d, Singapore Valset: %d, Singapore Testset: %d" % (
        len(self.boston_trainset), len(self.singapore_valset), len(self.singapore_testset)))

        print("augment with ", augment_strategy)
        # if augment_strategy is not None and augment_strategy != 'None':
        #     self.augmenter = getattr(augmenter, augment_strategy)
        # else:
        #     self.augmenter = None

        self.file_map = {}
        self.idx_map = {}

        dataset = {'boston_train': self.boston_trainset, 'singapore_val': self.singapore_valset,
                   'singapore_test': self.singapore_testset}

        # if _set == 'boston_train':
        #     self.img_transform = A.Compose([
        #         A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
        #         A.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.1, ),
        #         A.CoarseDropout(p=0.5),
        #         A.HueSaturationValue(p=0.5),
        #         A.GaussianBlur(p=0.5, blur_limit=(3, 15)),
        #         A.ISONoise(p=0.5)
        #     ])
        # else:
        self.img_transform = None

        self.augmentation_prob = 0.5

        count = 0
        for my_scene in dataset[_set]:

            scene_name = my_scene['name']

            if int(scene_name.split('-')[-1].lstrip('0')) in nuscenes_ignore:  # self.nusc_can.can_blacklist + 419
                continue

            first_sample_token = my_scene['first_sample_token']
            first_sample = NUSC.get('sample', first_sample_token)

            all_episode_data = []

            while True:
                cam_front_data = NUSC.get('sample_data', first_sample['data']['CAM_FRONT'])

                anno_path = dataset_path + 'annotations/' + cam_front_data['filename'].split('/')[-1].replace('jpg',
                                                                                                              'npy')

                if not os.path.exists((anno_path)):
                    break
                else:
                    anno = np.load(anno_path, allow_pickle=True).tolist()

                all_episode_data.append([first_sample, cam_front_data, anno])

                next_sample_token = first_sample['next']

                if next_sample_token == '':
                    break
                else:
                    next_sample = NUSC.get('sample', next_sample_token)
                first_sample = next_sample

            N = len(
                all_episode_data) - gap  # this makes sure I do not use last 'gap' frames as the training data (just for wpts generation)

            for _ in range(N):
                self.file_map[_ + count] = all_episode_data
                self.idx_map[_ + count] = _

            count += N

        print("Finished loading %s. Length: %d" % (dataset_path, count))
        self.batch_read_number = batch_read_number

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):

        all_episode_data = self.file_map[idx]
        index = self.idx_map[idx]

        sample, cam_front_data, anno = all_episode_data[index]

        rgb_name = self.dataset_path + cam_front_data['filename']

        rgb_image = cv2.imread(self.dataset_path + cam_front_data['filename'])

        # resize the image to half size
        # rgb_image = cv2.resize(rgb_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        rgb_image = cv2.resize(rgb_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        # print(rgb_image.shape)

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

        brake = anno['brake']
        steering = anno['steering']
        throttle = anno['throttle']
        # rad = anno['cmd_rad']
        speed = anno['speed_ms']
        veh_locations = anno['veh_locations']
        cam_locations = anno['cam_locations']
        img_locations = anno['img_locations']

        locations = []
        locations_vehicle = []

        for i in range(5):
            locations.append([img_locations[i][0] / 4., img_locations[i][1] / 4.])

            if veh_locations[i][1] < 0:
                locations_vehicle.append([veh_locations[i][0], 0.])
            else:
                locations_vehicle.append([veh_locations[i][0], veh_locations[i][1]])

        future_x, future_y = locations_vehicle[-1]

        if future_y < 0:
            future_y = 0.0

        rad = math.atan2(future_y, future_x)

        # cmd start from 1
        if rad >= 0 and rad < (math.pi * 85 / 180):
            cmd = 3
        elif rad >= (math.pi * 85 / 180) and rad < (math.pi * 95 / 180):
            cmd = 2
        elif rad >= (math.pi * 95 / 180) and rad <= (math.pi * 180 / 180):
            cmd = 1

        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])

        return rgb_images, rgb_name, np.array(locations_vehicle), cmd, speed


class Wrap(Dataset):
    def __init__(self, data, batch_size, samples):
        self.data = data
        self.batch_size = batch_size
        self.samples = samples

    def __len__(self):
        return self.batch_size * self.samples

    def __getitem__(self, i):
        return self.data[np.random.randint(len(self.data))]


def _dataloader(data, batch_size, num_workers):
    return DataLoader(
        data, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, drop_last=True, pin_memory=True)


def get_image(
        dataset_dir,
        batch_size=64, num_workers=0, shuffle=True, augment=None,
        n_step=5, gap=5, batch_aug=1):
    # import pdb; pdb.set_trace()

    def make_dataset(dir_name, is_train):
        # _dataset_dir = str(Path(dataset_dir) / dir_name)
        # _samples = 1000 if is_train else 10
        _samples = 1000 if is_train else 10
        _num_workers = num_workers if is_train else 0
        _batch_aug = batch_aug if is_train else 1
        _augment = augment if is_train else None

        data = ImageDataset(
            dataset_path=dataset_dir, _set=dir_name, gap=gap, augment_strategy=_augment, batch_aug=_batch_aug)
        data = Wrap(data, batch_size, _samples)
        data = DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=False, pin_memory=True)

        return data

    def make_testset(dir_name):
        _num_workers = 0
        _batch_aug = 1
        _augment = None

        data = ImageDataset(
            dataset_path=dataset_dir, _set=dir_name, gap=gap, augment_strategy=_augment, batch_aug=_batch_aug)
        data = DataLoader(
            data, batch_size=batch_size, num_workers=_num_workers,
            shuffle=False, drop_last=False, pin_memory=True)

        return data

    boston_train = make_dataset('boston_train', True)
    singapore_val = make_testset('singapore_val')
    singapore_test = make_testset('singapore_test')

    return boston_train, singapore_val, singapore_test


if __name__ == '__main__':
    batch_size = 256
    import tqdm

    dataset = ImageDataset('/raid0/dian/carla_0.9.6_data/train')
    loader = _dataloader(dataset, batch_size=batch_size, num_workers=16)
    mean = []
    for rgb_img, bird_view, locations, cmd, speed in tqdm.tqdm(loader):
        mean.append(rgb_img.mean(dim=(0, 2, 3)).numpy())

    print("Mean: ", np.mean(mean, axis=0))
    print("Std: ", np.std(mean, axis=0) * np.sqrt(batch_size))
