import cv2
import os
import pathlib
import numpy
import numpy as np
from math import cos, sin
import math
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from torchvision import transforms


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R


class ApolloSample:
    def __init__(self) -> None:
        self.image_path = None
        self.timestamp_nanosec = None
        self.previous_timestamp_nanosec = None
        self.next_timestamp_nanosec = None
        self.previous_translation = None
        self.next_translation = None
        self.velocity = None
        self.radian = None
        self.trajectory = None  # 这是错误的
        self.command = None  # 这是错误的
        self.rpyxyz = None
        self.radian_2 = None  # 这是正确的
        self.command_2 = None  # 这是正确的
        self.cityid = 11

    def setSE3(self, rpyxyz):
        roll = float(rpyxyz[0])
        pitch = float(rpyxyz[1])
        yaw = float(rpyxyz[2])
        x = float(rpyxyz[3])
        y = float(rpyxyz[4])
        z = float(rpyxyz[5])
        self.rpyxyz = [roll, pitch, yaw, x, y, z]
        self.translation = numpy.array([x, y, z])

        # yaw_matrix=numpy.array([[cos(yaw), -sin(yaw),0],
        #                         [sin(yaw), cos(yaw),0],
        #                         [0,0,1]])
        # pitch_matrix=numpy.array([[cos(pitch),0,sin(pitch)],
        #                           [0,1,0],
        #                           [-sin(pitch),0,cos(pitch)]])
        # roll_matrix=numpy.array([[1,0,0],
        #                          [0,cos(roll),-sin(roll)],
        #                          [0,sin(roll),cos(roll)]])

        # self.rotation=numpy.matmul(numpy.matmul(yaw_matrix,pitch_matrix),roll_matrix)

        self.rotation = euler_angles_to_rotation_matrix([roll, pitch, yaw])

        self.se3 = numpy.zeros((4, 4))
        self.se3[:3, :3] = self.rotation
        self.se3[:3, 3] = self.translation
        self.se3[3, 3] = 1

    def get_tensor_bgr(self):
        image_bgr = cv2.imread(str(self.image_path))  # numpy; color channels are BGR

        # height,width,channels=image_bgr.shape
        # resize_ratio=0.179254
        # dimension = (int(width*resize_ratio), int(height*resize_ratio)) # WIDTH, HEIGHT!!!!!!!

        # resize
        dimension = (400, 225)
        image_bgr = cv2.resize(image_bgr, dimension, interpolation=cv2.INTER_AREA)

        transform_function = transforms.ToTensor()
        image_bgr_tensor = transform_function(image_bgr)

        return image_bgr_tensor


class ApolloDataset:
    def __init__(self, dataset_dir="/home/data/apollo_train", default_cityid =11) -> None:
        '''
        @param dataset_dir
        只支持 /home/data/apollo_train
        %/home/data/apollo_test

        '''
        self.samples = []
        self.default_cityid = default_cityid
        dataset_path = pathlib.Path(dataset_dir)
        for road_folder in os.listdir(dataset_path):
            road_path = dataset_path / road_folder
            if not road_path.is_dir():
                continue

            records_dir_path = road_path / "image/GZ20180310B"

            # record folder
            for record_folder in os.listdir(records_dir_path):

                record_all_samples = []

                record_path = records_dir_path / record_folder
                if not record_path.is_dir():
                    continue

                print(road_folder, record_folder)

                camera_path = records_dir_path / record_folder / "Camera 5"
                pose_file_path = road_path / "pose/GZ20180310B" / record_folder / "Camera 5.txt"

                # 提取 pose 信息
                pose_file = open(pose_file_path, 'r')
                lines = pose_file.readlines()
                count = 0
                rpyxyz_map = {}
                for line in lines:
                    count += 1
                    image_filename = line.split()[0]
                    rpyxyz = line.split()[1][:-1].split(',')  # roll, pitch, yaw, x, y, z
                    rpyxyz_map[image_filename] = rpyxyz

                image_filenames = []
                for image_filename in os.listdir(camera_path):

                    if image_filename[0:2] == "._":
                        continue

                    image_filenames.append(image_filename)
                image_filenames = sorted(image_filenames)

                for image_filename in image_filenames:
                    image_path = camera_path / image_filename

                    # print(image_filename)
                    sample = ApolloSample()
                    sample.image_path = image_path

                    # 找 timestamp
                    timestamp = int(image_filename[7:7 + 9])
                    timestamp_nanosec = timestamp * 1e6
                    sample.timestamp_nanosec = timestamp_nanosec

                    sample.setSE3(rpyxyz_map[image_filename])

                    # 添加到 record samples
                    record_all_samples.append(sample)

                # 计算幀率
                # spf_list=[] # seconds per frame
                # for index in range(len(record_all_samples)-1):
                #     spf=(record_all_samples[index+1].timestamp_nanosec-record_all_samples[index].timestamp_nanosec)*1e-9/1
                #     spf_list.append(round(spf,2))
                # import statistics
                # mean_spf=statistics.mean(spf_list)

                # print(mean_spf)
                # print(spf_list)

                # num_future_points = 5
                # stride = round(0.5/mean_spf)  # 目標頻率 2fps 即 0.5 spf
                # #注意：av2 的幀率是 0.05spf，waymo 的幀率是 0.1spf, apollo 的幀率是不固定的.
                # num_unused_samples = num_future_points * stride

                # fill the next_timestamp and next_translation
                for index in range(len(record_all_samples) - 1):
                    record_all_samples[index].next_timestamp_nanosec = record_all_samples[index + 1].timestamp_nanosec
                    record_all_samples[index].next_translation = record_all_samples[index + 1].translation

                # fill the previous_timestamp and previous_translation
                for index in range(1, len(record_all_samples)):  #
                    record_all_samples[index].previous_timestamp_nanosec = record_all_samples[
                        index - 1].timestamp_nanosec
                    record_all_samples[index].previous_translation = record_all_samples[index - 1].translation

                # compute the velocity
                for index, sample in enumerate(record_all_samples):
                    if sample.previous_timestamp_nanosec is None:
                        first_timestamp_nanosec = sample.timestamp_nanosec
                        first_translation = sample.translation
                        second_timestamp_nanosec = sample.next_timestamp_nanosec
                        second_translation = sample.next_translation

                    elif sample.next_timestamp_nanosec is None:
                        first_timestamp_nanosec = sample.previous_timestamp_nanosec
                        first_translation = sample.previous_translation
                        second_timestamp_nanosec = sample.timestamp_nanosec
                        second_translation = sample.translation

                    else:
                        first_timestamp_nanosec = sample.previous_timestamp_nanosec
                        first_translation = sample.previous_translation
                        second_timestamp_nanosec = sample.next_timestamp_nanosec
                        second_translation = sample.next_translation

                    sample.velocity = (second_translation - first_translation) / (
                            (int(second_timestamp_nanosec) - int(first_timestamp_nanosec)) * 1e-9)
                    sample.velocity = numpy.linalg.norm(sample.velocity)

                # # compute the trajectory and command
                # for index, sample in enumerate(record_all_samples[0:-num_unused_samples]):

                #                 transform_matrix = sample.location.transform_matrix
                # transform_matrix = sample.se3
                # trajectory = []

                # for future_point_index in range(num_future_points):
                #     next_point_transform_matrix = record_all_samples[
                #         index + (future_point_index + 1) * stride].se3
                #     tracks = np.zeros((1, 4))
                #     tracks[:, -1] = 1.
                #     world_tracks = np.dot(next_point_transform_matrix, np.transpose(tracks))
                #     vehicle_tracks = np.dot(np.linalg.inv(transform_matrix), world_tracks)
                #     tracks_x_y_z = vehicle_tracks[:3, :]
                #     trajectory.append(tracks_x_y_z[[1, 0], 0])

                # trajectory = np.vstack(trajectory)
                # trajectory[:, 0] = -trajectory[:, 0]
                # future_point_x, future_point_y = trajectory[-1, 0], trajectory[-1, 1]

                # if future_point_y < 0:
                #     future_point_y = 0.0

                # radian = math.atan2(future_point_y, future_point_x)

                # command = None
                # # cmd start from 1
                # if radian >= 0 and radian < (math.pi * 85 / 180):
                #     command = 3  # turn right
                # elif radian >= (math.pi * 85 / 180) and radian < (math.pi * 95 / 180):
                #     command = 2  # go forward
                # elif radian >= (math.pi * 95 / 180) and radian <= (math.pi * 180 / 180):
                #     command = 1  # turn left

                # sample.radian = radian
                # sample.trajectory = trajectory
                # sample.command = command

                self.fill_command_and_trajectory(record_all_samples)
                self.fill_command_2(record_all_samples)

                no_use_index = -1
                for index in range(len(record_all_samples)):
                    if record_all_samples[index].command_2 is None or \
                            record_all_samples[index].radian_2 is None or \
                            record_all_samples[index].trajectory is None:
                        no_use_index = index
                        break

                # if(record_folder=="Record026"):
                #     print((record_all_samples[-1].timestamp_nanosec-record_all_samples[0].timestamp_nanosec)*1e-9/len(record_all_samples))
                #     print(len(record_all_samples))

                print(len(record_all_samples[:no_use_index]), "samples added")
                self.samples.extend(record_all_samples[:no_use_index])  # todo

    def fill_command_2(self, record_all_samples):
        list_timestamp_nanosec = []
        for sample in record_all_samples:
            list_timestamp_nanosec.append(sample.timestamp_nanosec)

        list_yaw = []
        for sample in record_all_samples:
            list_yaw.append(sample.rpyxyz[2])
        list_yaw = np.array(list_yaw)

        # xnew = np.arange(list_timestamp_nanosec[0],list_timestamp_nanosec[-1],50000000)
        # ynew = f(xnew)   # use interpolation function returned by `interp1d`

        # plt.plot(list_timestamp_nanosec, list_se3[:,col,row], 'o', xnew, ynew, '-')
        # plt.show()

        for sample in record_all_samples:
            if sample.timestamp_nanosec + 5 * 0.5 * 1e9 > list_timestamp_nanosec[-1]:
                break

            first_yaw = sample.rpyxyz[2]
            second_yaw = self.interpolate_yaw(list_yaw, list_timestamp_nanosec,
                                              sample.timestamp_nanosec + 5 * 0.5 * 1e9)

            if abs(second_yaw - first_yaw) > math.pi:
                if (second_yaw > first_yaw):
                    first_yaw += 2 * math.pi
                else:
                    second_yaw += 2 * math.pi

            radian = (second_yaw - first_yaw + math.pi / 2) % (math.pi * 2)  # 一般情况

            command = None
            # cmd start from 1
            if radian >= 0 and radian < (math.pi * 85 / 180):
                command = 3  # turn right
                # command = 1
            elif radian >= (math.pi * 85 / 180) and radian < (math.pi * 95 / 180):
                command = 2  # go forward
            elif radian >= (math.pi * 95 / 180) and radian <= (math.pi * 180 / 180):
                command = 1  # turn left
                # command = 3
            else:
                # print("ERROR RADIAN!!!!!",radian)
                # print("first",first_yaw)
                # print("second",second_yaw)
                command = 2

            sample.radian_2 = radian
            sample.command_2 = command

        no_command_index = None
        for index in range(len(record_all_samples)):
            if record_all_samples[index].command_2 is None:
                no_command_index = index
                break

        print(no_command_index)
        return record_all_samples[:no_command_index]

    def interpolate_yaw(self, list_yaw, list_timestamp_nanosec, timestamp_nanosec):

        f = interp1d(list_timestamp_nanosec, list_yaw)
        return f(timestamp_nanosec)

        # xnew = np.arange(list_timestamp_nanosec[0],list_timestamp_nanosec[-1],50000000)
        # ynew = f(xnew)   # use interpolation function returned by `interp1d`

        # plt.plot(list_timestamp_nanosec, list_yaw, 'o', xnew, ynew, '-')
        # plt.show()

    def fill_command_and_trajectory(self, record_all_samples):

        list_timestamp_nanosec = []
        for sample in record_all_samples:
            list_timestamp_nanosec.append(sample.timestamp_nanosec)

        list_se3 = []
        for sample in record_all_samples:
            list_se3.append(sample.se3)
        list_se3 = np.array(list_se3)

        # self.interpolate(list_se3,list_timestamp_nanosec,list_timestamp_nanosec[54])

        num_future_points = 5

        for index, sample in enumerate(record_all_samples):

            if sample.timestamp_nanosec + 5 * 0.5 * 1e9 > list_timestamp_nanosec[-1]:
                break

            future_virtual_se3_list = []
            for i in range(5):
                future_virtual_se3_list.append(self.interpolate_se3(list_se3, list_timestamp_nanosec,
                                                                    sample.timestamp_nanosec + (i + 1) * 0.5 * 1e9))

            # transform_matrix = sample.location.transform_matrix
            transform_matrix = sample.se3
            trajectory = []

            for future_point_index in range(num_future_points):
                next_point_transform_matrix = future_virtual_se3_list[future_point_index]
                tracks = np.zeros((1, 4))
                tracks[:, -1] = 1.
                world_tracks = np.dot(next_point_transform_matrix, np.transpose(tracks))
                vehicle_tracks = np.dot(np.linalg.inv(transform_matrix), world_tracks)
                tracks_x_y_z = vehicle_tracks[:3, :]
                trajectory.append(tracks_x_y_z[[1, 0], 0])

            trajectory = np.vstack(trajectory)
            trajectory[:, 0] = -trajectory[:, 0]

            # a = trajectory[:, 0]
            # b = trajectory[:, 1]
            # trajectory[:, 0] = -b
            # trajectory[:, 1] = -a

            # trajectory[:, 1] = -trajectory[:, 1]
            future_point_x, future_point_y = trajectory[-1, 0], trajectory[-1, 1]

            if future_point_y < 0:
                future_point_y = 0.0

            radian = math.atan2(future_point_y, future_point_x)

            radian -= math.pi / 2

            command = None
            # cmd start from 1
            if radian >= 0 and radian < (math.pi * 85 / 180):
                command = 3  # turn right
            elif radian >= (math.pi * 85 / 180) and radian < (math.pi * 95 / 180):
                command = 2  # go forward
            elif radian >= (math.pi * 95 / 180) and radian <= (math.pi * 180 / 180):
                command = 1  # turn left

            sample.radian = radian
            sample.command = command
            sample.trajectory = trajectory

        # no_command_count=0
        # for sample in record_all_samples:
        #     if sample.command is None:
        #         no_command_count+=1

        no_command_index = None
        for index in range(len(record_all_samples)):
            if record_all_samples[index].command is None:
                no_command_index = index
                break

        print(no_command_index)
        return record_all_samples[:no_command_index]

    def interpolate_se3(self, list_se3, list_timestamp_nanosec, timestamp_nanosec):

        result = np.zeros(shape=(4, 4))

        for col in range(3):
            for row in range(4):
                f = interp1d(list_timestamp_nanosec, list_se3[:, col, row])
                result[col, row] = f(timestamp_nanosec)

                # xnew = np.arange(list_timestamp_nanosec[0],list_timestamp_nanosec[-1],50000000)
                # ynew = f(xnew)   # use interpolation function returned by `interp1d`

                # plt.plot(list_timestamp_nanosec, list_se3[:,col,row], 'o', xnew, ynew, '-')
                # plt.show()

        result[3, 3] = 1

        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample.get_tensor_bgr(), str(sample.image_path), sample.trajectory, sample.command_2, sample.velocity,  self.default_cityid
        #sample.cityid