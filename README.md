# Visualize KITTI Objects in Camera, Point Cloud and BEV in Videos
This repository can be used to visualize objects of KITTI in camera image, point cloud and bird's eye view. It can be adapted to visualize objects in other point cloud datasets.

## Requirements
* Python >= 3.7 (Anaconda is recommended!)
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

## Usage
TBD

## Visualization
### Visualization of objects in camera image in a video  
<center><img src="gifs/camera.gif" width = "90%" height = ""></center>


### Visualization of objects in point cloud in a video  
<img src="gifs/pointcloud.gif" width = "60%">

### Visualization of objects in point cloud in a video (front camera view)
<img src="gifs/pointcloud_fov.gif" width = "60%">

### Visualization of objects in BEV in a video
<img src="gifs/bev.gif" width = "60%">
