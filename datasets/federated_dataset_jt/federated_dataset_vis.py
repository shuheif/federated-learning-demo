FEDERATED_DATASET_JT_VERSION = "v1.0 beta"
FEDERATED_DATASET_PICKLES_DIR = "/data/jt_data_space/feddrive_project/pickles_fds/"
import pathlib
import pickle


class FederatedDataset():
    def __init__(self):
        self.av2_dataset = None
        self.nuscenes_dataset = None
        self.waymo_dataset = None
        self.apollo_dataset = None

    def set_av2(self, cities, crop_and_resize=None, dataset_dir="/data/shared/av2_all/train", proportion=1):
        from .av2_dataset_jt import AV2Dataset
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

        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def set_nuscenes(self, cities, random_seed=99, train_or_test="TRAIN", train_proportion=1.0):
        from .nuscenes_dataset_jt import NuScenesDataset
        """
        @param cities:
            ['BOS','SGP'] 
            default: empty

        """
        self.nuscenes_dataset = NuScenesDataset(cities=cities, random_seed=random_seed,
                                                train_or_test=train_or_test,
                                                train_proportion=train_proportion)
        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def set_waymo(self, cities, tfrecords_dir="/data/shared/waymo_val", proportion=1,
                  images_dir="/home/data/waymo_val"):
        # from .waymo_dataset_jt import WaymoDataset
        from .waymo import WaymoDataset
        '''

        @param cities: ["location_phx","location_sf","location_other"]
        @param tfrecords_dir:
            supports "/data/shared/waymo_val" 
            "/data/shared/waymo_test" 
        @param images_dir:
            supports "/home/data/waymo_val"
            "/home/data/waymo_test"
        @param proportion: (0,1]
            default 1
        @return:
        '''
        self.waymo_dataset = WaymoDataset(cities=cities, proportion=proportion, tfrecords_dir=tfrecords_dir,
                                          images_dir=images_dir)
        print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    # def set_apollo(self):
    #     from .apollo_dataset import ApolloDataset
    #     self.apollo_dataset = ApolloDataset()
    #     print("FederatedDataset: There are", self.__len__(), "samples overall. ")

    def __len__(self):

        length = 0
        if self.av2_dataset is not None:
            length += len(self.av2_dataset)

        if self.nuscenes_dataset is not None:
            length += len(self.nuscenes_dataset)

        if self.waymo_dataset is not None:
            length += len(self.waymo_dataset)

        if self.apollo_dataset is not None:
            length += len(self.apollo_dataset)

        return length

    def __getitem__(self, index):

        if index < 0:
            index = index + self.__len__()

        if index < 0 or index >= self.__len__():
            raise Exception("Index", index, "out of range")

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

        if self.apollo_dataset is not None:
            if index < len(self.apollo_dataset):
                return self.apollo_dataset[index]
            else:
                index = index - len(self.apollo_dataset)

    def save_to_pickle(self, picklename):
        pickle_file_path = pathlib.Path(FEDERATED_DATASET_PICKLES_DIR + picklename + '.pickle')
        try:
            pickle_file = open(pickle_file_path, 'wb')
            pickle.dump(self.__dict__, pickle_file)
            pickle_file.close()
            print("Pickle saved successfully")
        except BaseException as e:
            print(e.args)

    def load_from_pickle(self, picklename):
        pickle_file_path = pathlib.Path(
            FEDERATED_DATASET_PICKLES_DIR + picklename + '.pickle')
        try:
            assert pickle_file_path.is_file()
            pickle_file = open(str(pickle_file_path), 'rb')
            temp_dict = pickle.load(pickle_file)
            pickle_file.close()
            self.__dict__.update(temp_dict)
            print("Dataset loaded successfully")
        except BaseException as e:
            print(e.args)

