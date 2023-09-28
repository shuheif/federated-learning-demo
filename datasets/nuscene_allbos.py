from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

import math
import random

from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
from pathlib import Path
import os
import numpy as np

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import math

# import albumentations as A

# nuscenes_ignore = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313,
#                    314, 419]  # 419 is not in canbus blacklist but it's speed is not okay


# Boston = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
#           103,
#           104, 105, 106, 107, 108, 109, 110, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176,
#           199, 200, 202,
#           203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 218, 219, 220, 222, 224, 225, 226, 227, 228, 229, 230,
#           231, 232, 233,
#           234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
#           256, 257, 258,
#           259, 260, 261, 262, 263, 264, 321, 323, 324, 499, 500, 501, 502, 504, 505, 506, 507, 508, 509, 510, 511, 512,
#           513, 514, 515,
#           517, 518, 552, 553, 554, 555, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 681,
#           683, 684, 685,
#           686, 687, 688, 689, 328, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 329, 330, 331, 332, 519,
#           520, 521, 522,
#           523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545,
#           546, 739, 740,
#           741, 744, 746, 747, 749, 750, 751, 752, 757, 758, 759, 760, 761, 762, 763, 764, 765, 767, 768, 769, 770, 771,
#           775, 777, 778,
#           780, 781, 782, 783, 784, 695, 696, 697, 698, 700, 701, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713,
#           714, 715, 716,
#           717, 718, 719, 726, 727, 728, 730, 731, 733, 734, 735, 736, 737, 738, 283, 284, 285, 286, 287, 288, 289, 290,
#           291, 292, 293,
#           294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 388, 389, 390, 391, 392, 393, 394, 395, 396,
#           397, 398, 556,
#           557, 558, 559, 560, 561, 562, 563, 564, 565, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452,
#           453, 454, 455,
#           456, 457, 458, 459, 461, 462, 463, 464, 465, 467, 468, 469, 471, 472, 474, 475, 476, 477, 478, 479, 480, 566,
#           568, 570, 571,
#           572, 573, 574, 575, 576, 577, 578, 580, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
#           596, 597, 598,
#           599, 600, 625, 626, 627, 629, 630, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
#           647, 648, 649,
#           650, 651, 652, 803, 804, 805, 806, 808, 809, 810, 811, 812, 813, 815, 816, 817, 819, 820, 821, 822, 868, 869,
#           870, 871, 872,
#           873, 875, 876, 877, 878, 880, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897,
#           898, 899, 900,
#           901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915]
#
# Singapore = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#              31, 32,
#              33, 34, 3, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 35, 36, 38, 39,
#              41, 42, 43, 44, 45, 46,
#              47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 120, 121, 122, 123, 124, 125, 126, 127, 128,
#              129, 130, 131, 132, 133, 134,
#              135, 138, 139, 149, 150, 151, 152, 154, 155, 157, 158, 159, 160, 190, 191, 192, 193, 194, 195, 196, 177,
#              178, 179, 180, 181, 182, 183,
#              184, 185, 187, 188, 315, 316, 317, 318, 221, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 360,
#              361, 362, 363, 364, 365, 366,
#              367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 399,
#              400, 401, 402, 403, 405, 406,
#              407, 408, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
#              429, 430, 431, 432, 433, 434,
#              435, 436, 437, 438, 439, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 858, 860, 861, 862, 863, 864,
#              865, 866, 916, 917, 919, 920,
#              921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 786, 787, 789, 790, 791, 792, 794, 795, 796, 797,
#              798, 799, 800, 802, 945, 947,
#              949, 952, 953, 955, 956, 957, 958, 959, 960, 961, 962, 963, 966, 967, 968, 969, 971, 972, 975, 976, 977,
#              978, 979, 980, 981, 982, 983,
#              984, 988, 989, 990, 991, 992, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
#              1008, 1009, 1010, 1011,
#              1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1044, 1045, 1046, 1047,
#              1048, 1049, 1050, 1051,
#              1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069,
#              1070, 1071, 1072, 1073,
#              1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091,
#              1092, 1093, 1094, 1095,
#              1096, 1097, 1098, 1099, 1100, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109, 1110]
#
# Singapore_val = [17, 351, 36, 191, 188, 221, 278, 370, 401, 851, 789, 961, 981, 1010, 1098]
#
nuscenes_ignore = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313,
                   314, 419]  # 419 is not in canbus blacklist but it's speed is not okay

Boston = [287, 1091, 512, 892, 705, 305, 158, 802, 415, 1065, 813, 912, 907, 560, 751, 231, 546, 332, 441, 1052,
               256, 180,
               632, 364, 904, 586, 368, 1077, 792, 509, 761, 1051, 1078, 371, 139, 379, 1090, 634, 765, 1064, 1014, 641,
               733, 356, 28,
               716, 931, 177, 286, 769, 414, 227, 257, 629, 1017, 1066, 1016, 803, 678, 72, 915, 562, 917, 1108, 13,
               187, 56, 975, 646,
               757, 465, 5, 232, 905, 382, 128, 428, 43, 955, 519, 422, 923, 1057, 901, 717, 949, 361, 1019, 529, 246,
               474, 208, 159,
               433, 394, 577, 125, 1062, 260, 377, 1003, 658, 253, 400, 444, 947, 871, 650, 746, 1076, 764, 44, 65, 462,
               1011, 638, 957,
               853, 391, 295, 740, 330, 962, 29, 447, 531, 427, 925, 924, 1082, 1097, 479, 138, 900, 1099, 711, 75, 237,
               972, 676, 181,
               202, 997, 984, 961, 561, 230, 385, 880, 124, 179, 668, 735, 1020, 62, 1109, 663, 194, 437, 644, 222,
               1110, 1071, 242, 1056,
               359, 463, 967, 656, 104, 191, 98, 383, 351, 876, 778, 24, 718, 106, 643, 53, 645, 500, 452, 920, 777, 46,
               304, 599, 981,
               123, 149, 270, 994, 821, 804, 908, 726, 220, 945, 787, 258, 247, 451, 927, 597, 762, 213, 510, 283, 636,
               976, 501, 33, 928,
               666, 796, 662, 1079, 783, 303, 664, 370, 869, 1089, 571, 69, 155, 808, 1018, 593, 815, 1058, 374, 895,
               373, 667, 35, 36, 775,
               914, 392, 469, 996, 245, 417, 294, 583, 390, 297, 534, 254, 882, 74, 11, 26, 218, 865, 600, 539, 132,
               558, 655, 302, 67, 448,
               193, 669, 505, 477, 1046, 866, 517, 108, 640, 868, 318, 190, 982, 805, 1087, 21, 49, 93, 226, 708, 809,
               131, 1086, 1107, 19,
               468, 475, 426, 51, 275, 737, 405, 899, 1059, 653, 1047, 584, 157, 1025, 456, 323, 719, 7, 73, 893, 710,
               536, 582, 130, 681,
               1045, 476, 1084, 34, 263, 464, 436, 133, 450, 806, 57, 873, 416, 851, 872, 1070, 979, 252, 76, 822, 862,
               403, 991, 1050, 713,
               60, 278, 630, 1049, 352, 649, 983, 884, 684, 596, 633, 626, 151, 515, 758, 543, 324, 730, 855, 556, 695,
               12, 63, 306, 522,
               389, 375, 1012, 17, 1100, 697, 1022, 922, 789, 467, 195, 103, 1008, 276, 184, 563, 262, 150, 188, 1060,
               381, 134, 763, 443,
               243, 728, 478, 812, 55, 360, 411, 749, 236, 182, 966, 902, 395, 588, 661, 25, 54, 110, 502, 449, 209,
               207, 738, 59, 1021, 911,
               771, 398, 741, 408, 200, 122, 784, 525, 752, 212, 160, 1094, 192, 71, 272, 856, 675, 1080, 228, 576, 598,
               759, 299, 523, 206,
               894, 674, 41, 909, 240, 511, 204, 251, 438, 358, 380, 1068, 689, 528, 353, 677, 9, 864, 97, 101, 679,
               432, 1044, 513, 552, 225,
               203, 850, 570, 916, 672, 434, 506, 249, 878, 393, 454, 861, 1088, 589, 978, 397, 269, 910, 526, 2, 544,
               889, 446, 627, 648, 760,
               50, 424, 407, 121, 820, 557, 290, 953, 183, 715, 129, 4, 988, 52, 264, 860, 652, 15, 367, 350, 566, 568,
               791, 1095, 688, 642,
               1006, 647, 1096, 440, 401, 268, 8, 480, 95, 421, 107, 459, 977, 559, 3, 659, 39, 16, 301, 439, 706, 535,
               406, 292, 261, 399,
               1005, 1072, 241, 651, 386, 1098, 959, 1069, 1007, 508, 592, 248, 956, 274, 328, 100, 1009, 701, 66, 565,
               657, 127, 575, 906,
               794, 402, 384, 27, 354, 344, 709, 105, 696, 277, 594, 1105, 877, 896, 554, 239, 499, 887, 1048, 1083,
               541, 698, 58, 995, 714,
               635, 590, 238, 457, 574, 285]

Singapore_val = [30, 38, 126, 259, 291, 298, 316, 362, 365, 376, 378, 410, 429, 471, 504, 507, 524, 527, 542, 770, 781, 797,
             810, 819, 852,
             863, 883, 885, 921, 960, 998, 1024, 1055, 1067]

Singapore = [847, 816, 321, 135, 1010, 712, 458, 288, 418, 1004, 425, 20, 514, 244, 372, 671, 886, 154, 455, 229, 357,
              870, 795, 1,
              22, 747, 929, 396, 273, 70, 315, 1106, 369, 1001, 1063, 1000, 739, 990, 811, 18, 707, 210, 199, 572, 665,
              858, 1073, 234, 798,
              214, 800, 152, 45, 363, 744, 578, 293, 219, 96, 686, 790, 300, 731, 1053, 1093, 329, 413, 1054, 120, 848,
              537, 530, 453, 958,
              521, 178, 14, 431, 897, 683, 786, 538, 68, 255, 585, 587, 1023, 109, 553, 999, 700, 854, 42, 780, 591,
              625, 532, 250, 564, 48,
              430, 461, 670, 366, 61, 963, 573, 703, 64, 969, 930, 1102, 211, 442, 271, 518, 639, 926, 1013, 545, 595,
              768, 734, 980, 47,
              891, 99, 1075, 345, 445, 388, 952, 533, 660, 704, 555, 185, 1061, 968, 1015, 849, 971, 348, 888, 1081,
              102, 233, 898, 435, 992,
              420, 1104, 913, 196, 355, 23, 349, 346, 654, 1002, 919, 875, 317, 1085, 817, 224, 750, 673, 296, 412, 221,
              1074, 92, 6, 1101,
              284, 289, 235, 32, 736, 782, 989, 347, 94, 520, 423, 472, 331, 727, 767, 903, 580, 10, 31, 1092, 637, 687,
              890, 685, 799]


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



        self.file_map = {}
        self.idx_map = {}

        dataset = {'boston_train': self.boston_trainset, 'singapore_val': self.singapore_valset,
                   'singapore_test': self.singapore_testset}

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




class ImageDataset_withobj(Dataset):
    def __init__(self,
                dataset_path='/data/jimuyang/nuscenes/',
                dataset_version='v1.0-trainval',
                _set='boston_train',  # or val
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



        self.file_map = {}
        self.idx_map = {}

        dataset = {'boston_train': self.boston_trainset, 'singapore_val': self.singapore_valset,
                   'singapore_test': self.singapore_testset}

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

        #obj_in_5
        ego_pose_token = NUSC.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token']
        for ep in NUSC.ego_pose:
            if ep['token'] == ego_pose_token:
                _ego_pose = ep
                break
        cur_pose_mat = transform_matrix(_ego_pose['translation'], Quaternion(_ego_pose['rotation']), inverse=False)

        object_all_info=[]
        for i in range(5):
            next_sample = NUSC.get('sample',sample['next'])
            ego_pose_token = NUSC.get('sample_data', next_sample['data']['CAM_FRONT'])['ego_pose_token']
            for ep in NUSC.ego_pose:
                if ep['token'] == ego_pose_token:
                    _ego_pose = ep
                    break
            cur_pose_mat = transform_matrix(_ego_pose['translation'], Quaternion(_ego_pose['rotation']), inverse=False)
            coords = []
            sizes = []
            for anno_token in next_sample['anns']:
                obj = NUSC.get('sample_annotation',anno_token)
                obj_pose_mat = transform_matrix(obj['translation'], Quaternion(obj['rotation']),inverse=False)
                cords = np.zeros((1, 4))
                cords[:, -1] = 1.
                world_cords = np.dot(obj_pose_mat, np.transpose(cords))
                veh_cords = np.dot(np.linalg.inv(cur_pose_mat), world_cords)
                veh_cords = veh_cords[0:2,0]
                obj_size = obj['size'][0:2]
                obj_size = obj_size
                coords.append(veh_cords)

                sizes.append(obj_size)
            object_all_info.append([coords,sizes])
            sample = next_sample


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

        return rgb_images, rgb_name, np.array(locations_vehicle), cmd, speed,  object_all_info


class YoutubeDataset(Dataset):
    def __init__(self,
                 dataset_path='/data2/shared/images_jpeg_resize/',
                 batch_aug=1,
                 augment_strategy=None,
                 ):
        self.rgb_transform = transforms.ToTensor()
        #  self.rgb_transform = transforms.Compose([
        #      transforms.Resize((235, 400)),
        #      transforms.ToTensor()])
        self.batch_aug = batch_aug

        # self.nusc_can = NuScenesCanBus(dataroot=dataset_path)
        # self.nusc = NuScenes(version=dataset_version, dataroot=dataset_path, verbose=True)

        self.dataset_path = dataset_path
        self.image_path_list = []
        videos_folder = os.listdir(self.dataset_path)
        for videos_folder in videos_folder:
            video_path = dataset_path + videos_folder
            # image_names = os.listdir(video_path)
            data_dir = Path(video_path).expanduser()
            if not data_dir.is_dir():
                continue
            for file in data_dir.glob('*'):
                if has_file_allowed_extension(file.name, IMG_EXTENSIONS):
                    self.image_path_list.append(video_path + '/' + file.name)

        print('finish loading youtube video data in total', len(self.image_path_list), ' frame')

        print("augment with ", augment_strategy)
        # if augment_strategy is not None and augment_strategy != 'None':
        #    # self.augmenter = getattr(augmenter, augment_strategy)
        #     self.augmenter = None
        # else:
        #     self.augmenter = None

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        rgb_image = cv2.imread(self.image_path_list[idx])

        # resize the image to half size
        # rgb_image = cv2.resize(rgb_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        dim = (400, 225)
        rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)

        # print(rgb_image.shape)

        # if self.augmenter:
        #     rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in
        #                   range(self.batch_aug)]
        # else:
        rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]

        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])

        return rgb_images


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

        # yt_data = YoutubeDataset(batch_aug=_batch_aug)
        # yt_data = DataLoader(
        #     yt_data, batch_size=batch_size, num_workers=num_workers,
        #     shuffle=True, drop_last=False, pin_memory=True)

        return data #, yt_data

    def make_testset(dir_name):
        _num_workers = 0
        _batch_aug = 1
        _augment = None

        # data = ImageDataset(
        # dataset_path=dataset_dir, _set=dir_name, gap=gap, augment_strategy=_augment, batch_aug=_batch_aug)
        data = ImageDataset_withobj(
            dataset_path=dataset_dir, _set=dir_name, gap=gap, augment_strategy=_augment, batch_aug=_batch_aug)
        data = DataLoader(
            data, batch_size=1, num_workers=_num_workers,
            shuffle=False, drop_last=False, pin_memory=True)

        return data

    # boston_train,yt_data = make_dataset('boston_train', True)
    boston_train = make_dataset('boston_train', True)
    singapore_train = make_dataset('singapore_train', True)
    # singapore_val = make_testset('singapore_test')
    # singapore_test = make_testset('singapore_test')

    #return boston_train,singapore_val,singapore_test #,yt_data
    return singapore_val
    # return boston_train

    # return boston_train, singapore_val, singapore_test, yt_data


if __name__ == '__main__':
    batch_size = 1
    import tqdm

    #dataset = ImageDataset('/raid0/dian/carla_0.9.6_data/train')
    dataset = ImageDataset_withobj()
    loader = _dataloader(dataset, batch_size=batch_size, num_workers=16)
    mean = []
    #rgb_images, rgb_name, np.array(locations_vehicle), cmd, speed, object_all_info
    for rgb_img, rgb_name, locations, cmd, speed,object_all_info in tqdm.tqdm(loader):
        mean.append(rgb_img.mean(dim=(0, 2, 3)).numpy())

    print("Mean: ", np.mean(mean, axis=0))
    print("Std: ", np.std(mean, axis=0) * np.sqrt(batch_size))
