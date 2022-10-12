# Drivers' Behaviour Detection
Detect drivers' behaviours using CNNs models.
Drivers' drowsiness condition and whether wearing helmets or smoking can be detected.


#### Requirements

~~~shell
- cuda
- pytorch==0.4.1+
- torchvision
- tensorflow==1.14.0
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
##### Helmet_dataset
/home/disk01/zgz/smoking_detection/data/helmet
##### Smoking_dataset
/home/disk01/zgz/smoking_detection/data



#### Run

##### Drowsiness detection
~~~shell
$ python drowsiness_detection.py
~~~

##### Helmet and smoking detection
~~~shell
$ python helmet_smoking_detection.py
~~~

##### Drivers' behaviour detection
~~~shell
$ python main.py
~~~
