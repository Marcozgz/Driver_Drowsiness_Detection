# Drive_Drowsiness_Detection
Detect drivers' drowsiness condition using CNNs models.
The code only perform testing not training the models.(Training the models is currently private and indent to add in the future)


#### Requirements

~~~shell
- cuda
- pytorch==0.4.1+
- torchvision
- opencv-python
~~~


#### Datasets
Datasets are stored in server 109
##### Drowsiness_dataset1
/home/disk01/zgz/head_pose_estimation/data/drowsiness_dataset/
##### Drowsiness_dataset2
/home/disk01/zgz/head_pose_estimation/data/DriverDrowsinessDataset/
##### Drowsiness_dataset3
/home/disk01/zgz/head_pose_estimation/data/dataset_new/


#### Testing

~~~shell
$ python test.py
~~~
