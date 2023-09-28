from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from pathlib import Path
from lyft_dataset_sdk.lyftdataset import LyftDataset
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import CAMERA_LIST, RING_CAMERA_LIST, STEREO_CAMERA_LIST

import math
import random

# import augmenter

# PIXEL_OFFSET = 10
# PIXELS_PER_METER = 5

# def world_to_pixel(x,y,ox,oy,ori_ox, ori_oy, offset=(-80,160), size=320, angle_jitter=15):
#     pixel_dx, pixel_dy = (x-ox)*PIXELS_PER_METER, (y-oy)*PIXELS_PER_METER

#     pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
#     pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

#     pixel_x = 320-pixel_x

#     return np.array([pixel_x, pixel_y]) + offset


# def project_to_image(pixel_x, pixel_y, tran=[0.,0.,0.], rot=[0.,0.,0.], fov=90, w=384, h=160, camera_world_z=1.4, crop_size=192):
#     # Apply fixed offset tp pixel_y
#     pixel_y -= 2*PIXELS_PER_METER

#     pixel_y = crop_size - pixel_y
#     pixel_x = pixel_x - crop_size/2

#     world_x = pixel_x / PIXELS_PER_METER
#     world_y = pixel_y / PIXELS_PER_METER

#     xyz = np.zeros((1,3))
#     xyz[0,0] = world_x
#     xyz[0,1] = camera_world_z
#     xyz[0,2] = world_y

#     f = w /(2 * np.tan(fov * np.pi / 360))
#     A = np.array([
#         [f, 0., w/2],
#         [0, f, h/2],
#         [0., 0., 1.]
#     ])
#     image_xy, _ = cv2.projectPoints(xyz, np.array(tran), np.array(rot), A, None)
#     image_xy[...,0] = np.clip(image_xy[...,0], 0, w)
#     image_xy[...,1] = np.clip(image_xy[...,1], 0, h)

#     return image_xy[0,0]


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
from pathlib import Path
import os
import numpy as np

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
import math

nuscenes_ignore = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313,
                   314, 419]  # 419 is not in canbus blacklist but it's speed is not okay

train_split = [287, 1091, 512, 892, 705, 305, 158, 802, 415, 1065, 813, 912, 907, 560, 751, 231, 546, 332, 441, 1052,
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

val_split = [30, 38, 126, 259, 291, 298, 316, 362, 365, 376, 378, 410, 429, 471, 504, 507, 524, 527, 542, 770, 781, 797,
             810, 819, 852,
             863, 883, 885, 921, 960, 998, 1024, 1055, 1067]

test_split = [847, 816, 321, 135, 1010, 712, 458, 288, 418, 1004, 425, 20, 514, 244, 372, 671, 886, 154, 455, 229, 357,
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

# NUSC_CAN = NuScenesCanBus(dataroot='/data/jimuyang/lbwyt/nuscenes/')
NUSC_CAN = NuScenesCanBus(dataroot='/data/jimuyang/nuscenes/')
# NUSC = NuScenes(version='v1.0-trainval', dataroot='/data/jimuyang/lbwyt/nuscenes/', verbose=True)
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

        # self.nusc_can = NuScenesCanBus(dataroot=dataset_path)
        # self.nusc = NuScenes(version=dataset_version, dataroot=dataset_path, verbose=True)

        self.dataset_path = dataset_path

        self.trainset = []
        self.valset = []
        self.testset = []

        for _scene in NUSC.scene:
            _ind = int(_scene['name'].split('-')[-1].lstrip('0'))
            if _ind in train_split:
                self.trainset.append(_scene)
            elif _ind in val_split:
                self.valset.append(_scene)
            elif _ind in test_split:
                self.testset.append(_scene)

        print("Trainset: %d, Valset: %d, Testset: %d" % (len(self.trainset), len(self.valset), len(self.testset)))

        print("augment with ", augment_strategy)
        if augment_strategy is not None and augment_strategy != 'None':
            # self.augmenter = getattr(augmenter, augment_strategy)
            self.augmenter = None
        else:
            self.augmenter = None

        self.file_map = {}
        self.idx_map = {}

        dataset = {'train': self.trainset, 'val': self.valset, 'test': self.testset}

        count = 0
        for my_scene in dataset[_set]:

            scene_name = my_scene['name']
            # if int(scene_name.split('-')[-1].lstrip('0')) in self.nusc_can.can_blacklist:
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

        rgb_image = cv2.imread(self.dataset_path + cam_front_data['filename'])

        # resize the image to half size
        # rgb_image = cv2.resize(rgb_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # dim = (235,400)
        rgb_image = cv2.resize(rgb_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        # print(rgb_image.shape)

        if self.augmenter:
            rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in
                          range(self.batch_aug)]
        else:
            rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]


        # rad = anno['cmd_rad']
        speed = anno['speed_ms']
        veh_locations = anno['veh_locations']
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

        return rgb_images,  np.array(locations_vehicle), cmd, speed

    #return rgb_images, np.array(locations), np.array(locations_vehicle), cmd, speed


class Lyft(Dataset):
    def __init__(self,
                 dataset_path='/data/ruizhao/lyft/test/',
                 json_path='/data/ruizhao/lyft/test/test_data',
                 gap=5,
                 batch_aug=1,
                 augment_strategy=None,
                 batch_read_number=819200,
                 ):
        self.rgb_transform = transforms.ToTensor()
        self.batch_aug = batch_aug

        self.dataset_path = dataset_path

        self.trainset = []
        self.valset = []
        self.testset = []

        self.level5data = LyftDataset(data_path=dataset_path, json_path=json_path, verbose=True)

        print("Trainset: %d, Valset: %d, Testset: %d" % (len(self.trainset), len(self.valset), len(self.testset)))



        self.file_map = {}
        self.idx_map = {}

        self.image_path_list = []
        self.trainslation_list = []
        self.timestamp = []
        self.flag=[] #flag that if it is the first or last frame of the scene
        self.first_frame_token = []
        self.way_points_num = 5

        #path = '/data/ruizhao/lyft/test/test_images/'

        for my_scene in self.level5data.scene:
            if len(self.flag)>0:
                self.flag[-1]=-1
            for i in range(my_scene['nbr_samples']-self.way_points_num):
                if i==0:
                    my_sample_token = my_scene["first_sample_token"]
                    sample = self.level5data.get("sample", my_sample_token)
                    self.flag.append(1)
                else:
                    my_sample_token = sample["next"]
                    sample = self.level5data.get("sample", my_sample_token)
                    self.flag.append(0)
                self.first_frame_token.append(my_sample_token)
                cam_front_data = self.level5data.get('sample_data', sample['data']['CAM_FRONT'])
                image_path = cam_front_data['filename']
                self.image_path_list.append(dataset_path+image_path)
                ego_pose = self.level5data.get('ego_pose', cam_front_data['ego_pose_token'])
                self.timestamp.append(ego_pose['timestamp'])
                self.trainslation_list.append(ego_pose['translation'])
                # this trainslation is a 3 element list
        self.flag[-1] = -1
        print("Finished loading %s. Length: %d" % (dataset_path, len(self.timestamp)))
        self.batch_read_number = batch_read_number

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        rgb_image = cv2.imread(self.image_path_list[idx])

        # resize the image to half size
        dim = (400, 225)
        rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)


        # print(rgb_image.shape)

        rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]




        location = np.array(self.trainslation_list[idx])[0:2]
        time=self.timestamp[idx]
        if idx==0:
            location_next =  np.array(self.trainslation_list[idx+1])[0:2]
            time_next = self.timestamp[idx+1]
            v = np.linalg.norm((location_next-location)/((time_next-time) * 10e-6))
        elif self.flag[idx]==1:
            location_next = np.array(self.trainslation_list[idx + 1])[0:2]
            time_next = self.timestamp[idx + 1]
            v = np.linalg.norm((location_next - location) / ((time_next - time) * 10e-6))
        elif self.flag[idx]==-1:
            location_pre = np.array(self.trainslation_list[idx - 1])[0:2]
            time_pre = self.timestamp[idx - 1]
            v = np.linalg.norm((location_pre - location) / ((time_pre - time) * 10e-6))
        else:
            location_next = np.array(self.trainslation_list[idx + 1])[0:2]
            time_next = self.timestamp[idx + 1]
            location_pre = np.array(self.trainslation_list[idx - 1])[0:2]
            time_pre = self.timestamp[idx - 1]
            v = np.linalg.norm((location_next - location_pre) / ((time_next - time_pre) * 10e-6))



        current_sample_token = self.first_frame_token[idx]

        sample = self.level5data.get("sample", current_sample_token)

        way_points = []
        for i in range(self.way_points_num):
            next_sample_token = sample["next"]
            sample = self.level5data.get("sample", next_sample_token)
            cam_front_data = self.level5data.get('sample_data', sample['data']['CAM_FRONT'])
            ego_pose = self.level5data.get('ego_pose', cam_front_data['ego_pose_token'])
            location_next = np.array(ego_pose['translation'])[0:2] - location
            way_points.append(location_next)
        way_points = np.vstack(way_points)

        future_x, future_y = way_points[-1,0],way_points[-1,1]

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

        return rgb_images, way_points, cmd, v


class Lyft_gap(Dataset):
    def __init__(self,
                 dataset_path='/data/ruizhao/lyft/test/',
                 json_path='/data/ruizhao/lyft/test/test_data',
                 gap=5,
                 batch_aug=1,
                 augment_strategy=None,
                 batch_read_number=819200,
                 ):
        self.rgb_transform = transforms.ToTensor()
        self.batch_aug = batch_aug

        self.dataset_path = dataset_path

        self.trainset = []
        self.valset = []
        self.testset = []

        self.level5data = LyftDataset(data_path=dataset_path, json_path=json_path, verbose=True)

        print("Trainset: %d, Valset: %d, Testset: %d" % (len(self.trainset), len(self.valset), len(self.testset)))



        self.file_map = {}
        self.idx_map = {}

        self.image_path_list = []
        self.trainslation_list = []
        self.timestamp = []
        self.flag=[] #flag that if it is the first or last frame of the scene
        self.first_frame_token = []
        self.way_points_num = 5

        #path = '/data/ruizhao/lyft/test/test_images/'

        for my_scene in self.level5data.scene:
            if len(self.flag)>0:
                self.flag[-1]=-1
            for i in range(my_scene['nbr_samples']-12):
                if i==0:
                    my_sample_token = my_scene["first_sample_token"]
                    sample = self.level5data.get("sample", my_sample_token)
                    self.flag.append(1)
                else:
                    my_sample_token = sample["next"]
                    sample = self.level5data.get("sample", my_sample_token)
                    self.flag.append(0)
                self.first_frame_token.append(my_sample_token)
                cam_front_data = self.level5data.get('sample_data', sample['data']['CAM_FRONT'])
                image_path = cam_front_data['filename']
                self.image_path_list.append(dataset_path+image_path)
                ego_pose = self.level5data.get('ego_pose', cam_front_data['ego_pose_token'])
                self.timestamp.append(ego_pose['timestamp'])
                self.trainslation_list.append(ego_pose['translation'])
                # this trainslation is a 3 element list
        self.flag[-1] = -1
        print("Finished loading %s. Length: %d" % (dataset_path, len(self.timestamp)))
        self.batch_read_number = batch_read_number

        # for my_scene in self.level5data.scene:
        #     if len(self.flag)>0:
        #         self.flag[-1]=-1
        #     i=0
        #     count=0
        #     while True:
        #     # for i in range(my_scene['nbr_samples']-self.way_points_num):
        #         if i==0:
        #             my_sample_token = my_scene["first_sample_token"]
        #             sample = self.level5data.get("sample", my_sample_token)
        #             self.flag.append(1)
        #         else:
        #             if i%2==1: #0.4s
        #                 my_sample_token = sample["next"]
        #                 sample = self.level5data.get("sample", my_sample_token)
        #                 my_sample_token = sample["next"]
        #                 sample = self.level5data.get("sample", my_sample_token)
        #                 self.flag.append(0)
        #                 count = count+2
        #             if i%2==0: #0.6s
        #                 my_sample_token = sample["next"]
        #                 sample = self.level5data.get("sample", my_sample_token)
        #                 my_sample_token = sample["next"]
        #                 sample = self.level5data.get("sample", my_sample_token)
        #                 my_sample_token = sample["next"]
        #                 sample = self.level5data.get("sample", my_sample_token)
        #                 self.flag.append(0)
        #                 count = count+3
        #             if count <
        #         i=i+1
        #         self.first_frame_token.append(my_sample_token)
        #         cam_front_data = self.level5data.get('sample_data', sample['data']['CAM_FRONT'])
        #         image_path = cam_front_data['filename']
        #         self.image_path_list.append(dataset_path+image_path)
        #         ego_pose = self.level5data.get('ego_pose', cam_front_data['ego_pose_token'])
        #         self.timestamp.append(ego_pose['timestamp'])
        #         self.trainslation_list.append(ego_pose['translation'])
        #         # this trainslation is a 3 element list
        #         if count>my_scene['nbr_samples']-3:
        #             break

        print("Finished loading %s. Length: %d" % (dataset_path, len(self.timestamp)))
        self.batch_read_number = batch_read_number

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        rgb_image = cv2.imread(self.image_path_list[idx])

        # resize the image to half size
        dim = (400, 225)
        rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)


        # print(rgb_image.shape)

        rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]




        location = np.array(self.trainslation_list[idx])[0:2]
        time=self.timestamp[idx]
        if idx==0:
            location_next =  np.array(self.trainslation_list[idx+1])[0:2]
            time_next = self.timestamp[idx+1]
            v = np.linalg.norm((location_next-location)/((time_next-time) * 10e-6))
        elif self.flag[idx]==1:
            location_next = np.array(self.trainslation_list[idx + 1])[0:2]
            time_next = self.timestamp[idx + 1]
            v = np.linalg.norm((location_next - location) / ((time_next - time) * 10e-6))
        elif self.flag[idx]==-1:
            location_pre = np.array(self.trainslation_list[idx - 1])[0:2]
            time_pre = self.timestamp[idx - 1]
            v = np.linalg.norm((location_pre - location) / ((time_pre - time) * 10e-6))
        else:
            location_next = np.array(self.trainslation_list[idx + 1])[0:2]
            time_next = self.timestamp[idx + 1]
            location_pre = np.array(self.trainslation_list[idx - 1])[0:2]
            time_pre = self.timestamp[idx - 1]
            v = np.linalg.norm((location_next - location_pre) / ((time_next - time_pre) * 10e-6))






        # way_points = []
        # for i in range(self.way_points_num):
        #     location_next = np.array(self.trainslation_list[idx+i+1])[0:2] - location
        #     way_points.append(location_next)
        # way_points = np.vstack(way_points)

        way_points = []
        #R.from_quat

        current_sample_token = self.first_frame_token[idx]
        sample = self.level5data.get("sample", current_sample_token)
        cam_front_data = self.level5data.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = self.level5data.get('ego_pose', cam_front_data['ego_pose_token'])
        # first_location = np.array(ego_pose['translation'])
        # first_rotation = np.array(ego_pose['rotation'])
        # r = R.from_quat(first_rotation)
        # Rmatrix = r.as_matrix()
        sensor = self.level5data.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        # camera_rotation = sensor['rotation']
        # rc = R.from_quat(camera_rotation)
        # Rcmatrix = rc.as_matrix()
        # Rmatrix = Rmatrix*Rcmatrix
        cur_pose_mat = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        cur_cam_pose_mat = transform_matrix(sensor['translation'], Quaternion(sensor['rotation']),
                                            inverse=False)
        cur_cam_intrinsic = sensor['camera_intrinsic']
        for i in range(self.way_points_num):
            if i%2==0:
                next_sample_token = sample["next"]
                sample = self.level5data.get("sample", next_sample_token)
                next_sample_token = sample["next"]
                sample = self.level5data.get("sample", next_sample_token)
            else:
                next_sample_token = sample["next"]
                sample = self.level5data.get("sample", next_sample_token)
                next_sample_token = sample["next"]
                sample = self.level5data.get("sample", next_sample_token)
                next_sample_token = sample["next"]
                sample = self.level5data.get("sample", next_sample_token)
            cam_front_data = self.level5data.get('sample_data', sample['data']['CAM_FRONT'])

            ego_pose = self.level5data.get('ego_pose', cam_front_data['ego_pose_token'])
            sensor =  self.level5data.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

            next_pose_mat = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
            next_cam_pose_mat = transform_matrix(sensor['translation'], Quaternion(sensor['rotation']),
                                                inverse=False)
            cords = np.zeros((1, 4))
            cords[:, -1] = 1.
            world_cords = np.dot(next_pose_mat, np.transpose(cords))
            veh_cords = np.dot(np.linalg.inv(cur_pose_mat), world_cords)
            cam_cords = np.dot(np.linalg.inv(cur_cam_pose_mat), veh_cords)
            cords_x_y_z = cam_cords[:3, :]
            #location_next = np.array(ego_pose['translation'])- first_location
            #location_next = np.dot(np.linalg.inv(Rmatrix)[0:2,0:2] , location_next[0:2].reshape((2,1)))   #np.linalg.inv(Rmatrix)

            #location_next = np.dot(np.linalg.inv(Rmatrix), location_next.reshape((3, 1)))
            #location_next = location_next[0:2,0]
            way_points.append(cords_x_y_z[0:2,0])
        way_points = np.vstack(way_points)

        future_x, future_y = way_points[-1,0],way_points[-1,1]

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

        return rgb_images, way_points, cmd, v

class Argoverse(Dataset):
    def __init__(self,
                 dataset_path='/data/ruizhao/argoverse/argoverse-tracking/test',
                 gap=5,
                 batch_aug=1,
                 ):

        self.rgb_transform = transforms.ToTensor()
        self.batch_aug = batch_aug
        self.dataset_path = dataset_path
        self.argoverse_loader = ArgoverseTrackingLoader(self.dataset_path)


        print("Total Argoverse scenes: %d" % (len(self.argoverse_loader)))

        self.image_path_list = []
        self.timestamp = []
        self.data_log_id = []
        self.idx_list = []

        #path = '/data/ruizhao/lyft/test/test_images/'

        for i ,argoverse_data in enumerate(self.argoverse_loader):
            # select the center front camera
            camera = argoverse_data.CAMERA_LIST[0]
            self.image_path_list.extend(argoverse_data.image_list_sync[camera][0:-25])
            lidar_time = argoverse_data.lidar_timestamp_list
            self.timestamp.extend(lidar_time[0:-25])
            for idx in range(len(lidar_time)-25):
                self.data_log_id.append(i)
                self.idx_list.append(idx)
        print("Finished loading %s. Length: %d" % (dataset_path, len(self.timestamp)))

        print("Finished loading %s. Length: %d" % (dataset_path, len(self.timestamp)))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        rgb_image = cv2.imread(self.image_path_list[idx])
        dim = (400, 225)
        rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)
        # print(rgb_image.shape)
        rgb_images = [rgb_image for i in range(self.batch_aug)]
        if self.batch_aug == 1:
            rgb_images = rgb_images[0]


        data_log_id = self.data_log_id[idx]
        data_log = self.argoverse_loader[data_log_id]
        idx_in_log = self.idx_list[idx] # index of this log
        current_pose = data_log.get_pose(idx=idx_in_log)
        current_translation = current_pose.translation # 3,
        next_pose = data_log.get_pose(idx=idx_in_log+1)
        next_translation = next_pose.translation  # 3,

        cureent_log_timestamp = data_log.lidar_timestamp_list
        if idx_in_log==0:
            v = (next_translation - current_translation)/((cureent_log_timestamp[idx_in_log+1]-cureent_log_timestamp[idx_in_log])*10e-9)
            v = np.linalg.norm(v)
        else:
            pre_pose = data_log.get_pose(idx=idx_in_log - 1)
            pre_translation = pre_pose.translation  # 3,
            v = (next_translation - pre_translation) / ((cureent_log_timestamp[idx_in_log + 1] - cureent_log_timestamp[idx_in_log-1]) * 10e-9)
            v = np.linalg.norm(v)



        way_points = []

        #R.from_quat
        current_transform_matrix = current_pose.transform_matrix  #4x4

        # cur_pose_mat = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        # cur_cam_pose_mat = transform_matrix(sensor['translation'], Quaternion(sensor['rotation']),
        #                                     inverse=False)
        for i in range(5):
            gap = (i+1)*5
            next_key_pose = data_log.get_pose(idx=idx_in_log+gap)
            next_transform_matrix = next_key_pose.transform_matrix
            cords = np.zeros((1, 4))
            cords[:, -1] = 1.
            world_cords = np.dot(next_transform_matrix, np.transpose(cords))
            veh_cords = np.dot(np.linalg.inv(current_transform_matrix), world_cords)
            cords_x_y_z = veh_cords[:3, :]
            way_points.append(cords_x_y_z[[1,0], 0])
        way_points = np.vstack(way_points)
        future_x, future_y = way_points[-1, 0], way_points[-1, 1]

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

        return rgb_images, way_points, cmd, v


#
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
        if augment_strategy is not None and augment_strategy != 'None':
           # self.augmenter = getattr(augmenter, augment_strategy)
            self.augmenter = None
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        rgb_image = cv2.imread(self.image_path_list[idx])

        # resize the image to half size
        # rgb_image = cv2.resize(rgb_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        dim = (400, 225)
        rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)

        # print(rgb_image.shape)

        if self.augmenter:
            rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in
                          range(self.batch_aug)]
        else:
            rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]

        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])

        return rgb_images




# class YoutubeDataset(Dataset):
#     def __init__(self,
#                  dataset_path='/data2/shared/images_jpeg_resize/Boston4K-CommonwealthStreetcars-DrivingDowntownUSA/',
#                  batch_aug=1,
#                  augment_strategy =None,
#                  ):
#         self.rgb_transform = transforms.ToTensor()
#         self.batch_aug = batch_aug
#
#         self.dataset_path = dataset_path
#         self.image_path_list = []
#        # videos_folder = os.listdir(self.dataset_path)
#         data_dir = Path(dataset_path).expanduser()
#
#         # if not data_dir.is_dir():
#         #     continue
#         for file in data_dir.glob('*'):
#             if has_file_allowed_extension(file.name, IMG_EXTENSIONS):
#                 self.image_path_list.append(self.dataset_path+'/'+file.name)
#
#
#             # image_names = os.listdir(video_path)
#             # for image_name in image_names:
#             #     self.image_path_list.append(video_path+'/'+image_name)
#         print('finish loading youtube video data in total', len(self.image_path_list), ' frame')
#
#         print("augment with ", augment_strategy)
#         if augment_strategy is not None and augment_strategy != 'None':
#             #self.augmenter = getattr(augmenter, augment_strategy)
#             self.augmenter = None
#         else:
#             self.augmenter = None
#
#     def __len__(self):
#         return len(self.image_path_list)
#
#     def __getitem__(self, idx):
#
#
#         rgb_image = cv2.imread(self.image_path_list[idx])
#
#         # resize the image to half size
#         # rgb_image = cv2.resize(rgb_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#         dim = (400,225)
#         rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)
#
#         # print(rgb_image.shape)
#
#         if self.augmenter:
#             rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in
#                           range(self.batch_aug)]
#         else:
#             rgb_images = [rgb_image for i in range(self.batch_aug)]
#
#         if self.batch_aug == 1:
#             rgb_images = rgb_images[0]
#
#
#         if self.batch_aug == 1:
#             rgb_images = self.rgb_transform(rgb_images)
#         else:
#             rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])
#
#         return rgb_images
#

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

        if not is_train:
            return data

        yt_data = YoutubeDataset(batch_aug=_batch_aug)
        yt_data = DataLoader(
            yt_data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=False, pin_memory=True)

        return data, yt_data

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

    def make_lyft_testset(dir_name):
        _num_workers = 0
        _batch_aug = 1
        _augment = None

        #data = Lyft()
        data = Lyft_gap()
        data = DataLoader(
            data, batch_size=batch_size, num_workers=_num_workers,
            shuffle=False, drop_last=False, pin_memory=True)

        return data

    def make_Argoverse_testset(dir_name):
        _num_workers = 0
        _batch_aug = 1
        _augment = None

        #data = Lyft()
        data = Argoverse()
        data = DataLoader(
            data, batch_size=batch_size, num_workers=_num_workers,
            shuffle=False, drop_last=False, pin_memory=True)

        return data
    train = make_dataset('train', True)
    val = make_dataset('val', False)
    #test = make_testset('test')
    #test = make_lyft_testset('test')
    test = make_Argoverse_testset('test')

    return train, val, test #, yt_data


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
