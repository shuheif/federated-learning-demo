Federated Dataset Package by JT 
v1.1.1 beta



v1.1.1  
1. 改进了 crop_and_resize。

v1.1  
1. av2 的图片可以支持两种 crop_and_resize 方式。
2. nuscenes 支持 train 和 test 分离。




使用指导：

请直接将此 package 放入 project 的主目录中。请不要更改各文件名。


```python

from federated_dataset_jt.federated_dataset_jt import *
from federated_dataset_jt.crop_and_resize import *

train_dataset=FederatedDataset()
train_dataset.set_av2(cities=["ATX"],dataset_dir="/data/shared/av2_all/train",crop_and_resize=crop_car_and_resize)
# crop_central_and_resize 是只保留中间扁平的部分。
# crop_car_and_resize 是只裁剪汽车。
train_dataset.set_nuscenes(cities=["SGP","BOS"],train_proportion=0.8,train_or_test="TRAIN",random_seed=78)


test_dataset=FederatedDataset()
test_dataset.set_av2(cities=["ATX"],dataset_dir="/data/shared/av2_all/test",crop_and_resize=crop_car_and_resize)
test_dataset.set_nuscenes(cities=["SGP","BOS"],train_proportion=0.8,train_or_test="TEST",random_seed=78)
# 请确保 random_seed 和 train_proportion 和 train_dataset 的相同。




```



v1.0：
1. 无论是 av2 和 nuscene，都是直接 resize 为 width 400, height 225。  
2. av2 无内存缓存。  
3. nuscene 的城市标签是根据时区信息提取的。  
