"""
define some utility functions
"""
import numpy as np


def get_all_index_in_list(L, item):
    """
    get all the indexies of the same items in the list
    :param L: list
    :param item: item to be found
    :return: the indexies of all same items in the list
    """

    return [index for (index, value) in enumerate(L) if value == item]


def custom_colors():
    """
    define some colors in BGR, add more if needed
    :return: return a list of colors
    """

    colors = []

    colors.append([0, 255, 255])  # yellow
    colors.append([245, 135, 56])    # light blue
    colors.append([0, 255, 0])       # green
    colors.append([255, 0, 255])     # magenta
    colors.append([240, 32, 160])    # purple
    colors.append([255, 255, 0])     # cyan
    colors.append([0, 0, 255])       # red
    colors.append([0, 215, 255])     # gold
    colors.append([144, 238, 144])   # light green
    colors.append([128, 0, 0])       # navy
    colors.append([0, 0, 128])       # maroon
    colors.append([255, 0, 0])  # blue
    colors.append([128, 128, 0])     # teal
    colors.append([0, 128, 128])     # olive
    colors.append([128, 0, 0])       # navy

    return colors


def transform_3dbox_to_pointcloud(dimension, location, rotation):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght/2, lenght/2, -lenght/2, -lenght/2,  lenght/2,  lenght/2, -lenght/2, -lenght/2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # from camera coordinate to velodyne coordinate
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

    return corners_3d


def velodyne_to_camera_2(pcloud, calib):

    pcloud_temp = np.hstack((pcloud[:, :3], np.ones((pcloud.shape[0], 1), dtype=np.float32)))  # [N, 4]
    pcloud_C0 = np.dot(pcloud_temp, np.dot(calib['Tr_velo_cam'].T, calib['Rect'].T))  # [N, 3]

    pcloud_C0_temp = np.hstack((pcloud_C0, np.ones((pcloud.shape[0], 1), dtype=np.float32)))
    pcloud_C2 = np.dot(pcloud_C0_temp, calib['P2'].T)  # [N, 3]
    pcloud_C2_depth = pcloud_C2[:, 2]
    pcloud_C2 = (pcloud_C2[:, :2].T / pcloud_C2[:, 2]).T

    return pcloud_C2_depth, pcloud_C2


def remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, img_size):

    inds = pcloud_C2_depth > 0
    inds = np.logical_and(inds, pcloud_C2[:, 0] > 0)
    inds = np.logical_and(inds, pcloud_C2[:, 0] < img_size['width'])
    inds = np.logical_and(inds, pcloud_C2[:, 1] > 0)
    inds = np.logical_and(inds, pcloud_C2[:, 1] < img_size['height'])

    pcloud_in_img = pcloud[inds]

    return pcloud_in_img


def transform_3dbox_to_image(dimension, location, rotation, calib):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght / 2, lenght / 2, -lenght / 2, -lenght / 2, lenght / 2, lenght / 2, -lenght / 2, -lenght / 2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # only show 3D bounding box for objects in front of the camera
    if np.any(corners_3d[:, 2] < 0.1):
        corners_3d_img = None
    else:
        # from camera coordinate to image coordinate
        corners_3d_temp = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_3d_img = np.matmul(corners_3d_temp, calib['P2'].T)
        corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2][:, None]

    return corners_3d_img






