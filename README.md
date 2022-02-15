# Visualize KITTI Objects in Camera, Point Cloud and BEV in Videos
This repository can be used to visualize objects of KITTI in camera image, point cloud and bird's eye view. It can be adapted to visualize objects in other point cloud datasets.

## Installation
* Python >= 3.7 (Anaconda is recommended!)
* Download the repository
```
(base)$ git clone https://github.com/HengLan/Visualize-KITTI-Objects-in-Videos.git
(base)$ cd Visualize-KITTI-Objects-in-Videos
```
* Create a new environment
```
(base)$ conda create -n pointcloud python=3.7 # note that, the vtk is incompatible with Python 3.8
(base)$ conda activate pointcloud
```

* Install required packages 
```
(pointcloud)$ pip install opencv-python
(pointcloud)$ pip install pillow
(pointcloud)$ pip install scipy
(pointcloud)$ conda install mayavi -c conda-forge
```

I use Anaconda with Python 3.8 on Ubuntu 20.04 for running the code!

## Data Preparation (KITTI)
* Download <a href="http://www.cvlibs.net/datasets/kitti/eval_tracking.php">KITTI tracking data</a>, including `left color images`, `velodyne`, `camera calibration` and `training labels`.
* Unzip all the downloaded files.
* Remove `test` subfolder in each folder, and re-organize each folder as follows
```
KITTI
  --- [label_2]
        --- {0000-0020}.txt
  --- [calib]
        --- {0000-0020}.txt
  --- [image_2]
        --- [0000-0020] folders with .png images
  --- [velodyne]
        --- [0000-0020] folders with .bin files
```
If you don't want to download the dataset, a smaller version in `root_path_to_this_repo/data/KITTI/` is provided in this repository with a simplified seuqnece (sequence `0001`). You can also refer this to prepare the dataset.

## Usage

```
usage: visualization_demo.py [-h] [--dataset_path DATASET_PATH]
                             [--sequence_id SEQUENCE_ID]
                             [--vis_data_type {camera,pointcloud,bev}] [--fov]
                             [--vis_box] [--box_type {2d,3d}] [--save_img]
                             [--save_path SAVE_PATH]

Visualize KITTI Objects in Videos

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        the path to KITTI, a default dataset is provided
  --sequence_id SEQUENCE_ID
                        the id of sequence to visualize
  --vis_data_type {camera,pointcloud,bev}
                        show object in camera, pointcloud or birds eye view
  --fov                 only show front view of pointcloud
  --vis_box             show object box or not
  --box_type {2d,3d}    designed for visualization in camera, show 2d or 3d
                        object box
  --save_img            save visualization result or not
  --save_path SAVE_PATH
                        path to save visualization result

```
### Examples
* Visualize objects using 2D box in camera in a video
```
(pointcloud)$ python visualization_demo.py --dataset_path=path_to_KITTI --sequence_id=0 --vis_data_type='camera' --vis_box --box_type='2d'
```
* Visualize objects using 3D box in camera in a video, and save the visualization to images
```
(pointcloud)$ python visualization_demo.py --dataset_path=path_to_KITTI --sequence_id=0 --vis_data_type='camera' --vis_box --box_type='3d' --save_img
```
* Visualize objects in point cloud in a video
```
(pointcloud)$ python visualization_demo.py --dataset_path=path_to_KITTI --sequence_id=0 --vis_data_type='pointcloud' --vis_box

```
* Visualize objects in point cloud in a video in front camera view, and save visualization to images
```
(pointcloud)$ python visualization_demo.py --dataset_path=path_to_KITTI --sequence_id=0 --vis_data_type='pointcloud' --fov --vis_box --save_img

```
* Visualize objects in bird's eye view in a video
```
(pointcloud)$ python visualization_demo.py --dataset_path=path_to_KITTI --sequence_id=0 --vis_data_type='bev' --vis_box

```

## Visualization
### Visualization of objects in camera image in a video  
<center><img src="gifs/camera.gif" width = "90%" height = ""></center>


### Visualization of objects in point cloud in a video  
<img src="gifs/pointcloud.gif" width = "60%">

### Visualization of objects in point cloud in a video (front camera view)
<img src="gifs/pointcloud_fov.gif" width = "60%">

### Visualization of objects in BEV in a video
<img src="gifs/bev.gif" width = "60%">

## Questions and comments
Questions and comments are welcomed!
