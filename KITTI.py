"""
Written by Heng Fan
The KITTI class
"""
import os
import glob
import cv2 as cv
import mayavi.mlab as mlab
import numpy as np
# import seaborn as sns
from utility import *


class KITTI(object):
    """
    Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,
    Andreas Geiger, Philip Lenz, and Raquel Urtasun,
    CVPR, 2012.
    """
    def __init__(self, dataset_path):
        '''
        :param dataset_path: path to the KITTI dataset
        '''
        super(KITTI, self).__init__()
        self.dataset_path = dataset_path
        self._get_sequence_list()
        self.categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        # self.colors = sns.color_palette(palette='muted', n_colors=len(self.categories))
        self.colors = custom_colors()

    def _get_sequence_list(self):
        """
        :return: the sequence list
        """

        # used to store the sequence info
        self.sequence_list = []

        # get all video names
        vid_names = os.listdir('{}/velodyne'.format(self.dataset_path))
        vid_names.sort()
        self.sequence_num = len(vid_names)

        for vid in vid_names:
            # store information of a sequence
            sequence = dict()

            sequence['name'] = vid
            sequence['img_list'] = glob.glob('{}/image_2/{}/*.png'.format(self.dataset_path, vid))
            sequence['img_list'].sort()
            sequence['img_size'] = self.get_sequence_img_size(sequence['img_list'][0])
            sequence['pcloud_list'] = glob.glob('{}/velodyne/{}/*.bin'.format(self.dataset_path, vid))
            sequence['pcloud_list'].sort()
            sequence['label_list'] = self.get_sequence_labels(vid)   # get lables in this sequence
            sequence['calib'] = self.get_sequence_calib(vid)

            self.sequence_list.append(sequence)

    def get_sequence_img_size(self, initial_img_path):
        """
        get the size of image in the sequence
        :return: image size
        """

        img = cv.imread(initial_img_path)  # read image

        img_size = dict()

        img_size['height'] = img.shape[0]
        img_size['width'] = img.shape[1]

        return img_size

    def get_sequence_calib(self, sequence_name):
        """
        get the calib parameters
        :param sequence_name: sequence name
        :return: calib
        """

        # load data
        sequence_calib_path = '{}/calib/{}.txt'.format(self.dataset_path, sequence_name)
        with open(sequence_calib_path, 'r') as f:
            calib_lines = f.readlines()

        calib = dict()
        calib['P0'] = np.array(calib_lines[0].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P1'] = np.array(calib_lines[1].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P2'] = np.array(calib_lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P3'] = np.array(calib_lines[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['Rect'] = np.array(calib_lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
        calib['Tr_velo_cam'] = np.array(calib_lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['Tr_imu_velo'] = np.array(calib_lines[6].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

        return calib

    def get_sequence_labels(self, sequence_name):
        """
        get labels for all frames in the sequence
        :param sequence_name: sequence name
        :return: the labels of a sequence
        """

        sequence_label_path = '{}/label_2/{}.txt'.format(self.dataset_path, sequence_name)
        with open(sequence_label_path, 'r') as f:
            labels = f.readlines()

        # parse each line
        # 1 frame number, 2 track id, 3 object type, 4 truncated, 5 occluded (0: full visible, 1: partly occluded, 2: largely occluded),
        # 6 alpha, 7-10 2d bbox in RGB image, 11-13 dimension (height, width, length in meters), 14-16 center location (x, y, z in meters),
        # 17 rotation around Y-axis
        frame_id_list = []
        object_list = []
        for line in labels:
            # process each line
            line = line.split()
            frame_id, object_id, object_type, truncat, occ, alpha, l, t, r, b, height, width, lenght, x, y, z, rotation = line

            # map string to int or float
            frame_id, object_id, truncat, occ = map(int, [frame_id, object_id, truncat, occ])
            alpha, l, t, r, b, height, width, lenght, x, y, z, rotation = map(float, [alpha, l, t, r, b, height, width, lenght, x, y, z, rotation])

            if object_type != 'DontCare':
                object = dict()    # store the information of this object
                object['id'] = object_id
                object['object_type'] = object_type
                object['truncat'] = truncat
                object['occ'] = occ
                object['alpha'] = alpha
                object['bbox'] = [l, t, r, b]
                object['dimension'] = [height, width, lenght]
                object['location'] = [x, y, z]
                object['rotation'] = rotation

                object_list.append(object)
                frame_id_list.append(frame_id)

        # number of frames in this sequence
        frame_num = frame_id + 1

        # collect labels for each single frame
        sequence_label = []     # the labels of all frames in the sequence
        for i in range(frame_num):
            # get all the labels in frame i
            frame_ids = get_all_index_in_list(frame_id_list, i)
            if len(frame_ids) > 0:
                frame_label = object_list[frame_ids[0]:frame_ids[-1]+1]
                sequence_label.append(frame_label)
            else:
                # for some frames, there are no objects
                sequence_label.append([])

        return sequence_label

    def show_sequence_rgb(self, vid_id, vis_2dbox=False, vis_3dbox=False, save_img=False, save_path=None, wait_time=30):
        """
        visualize the sequence in RGB
        :param vid_id: id of the sequence, starting from 0
        :return: none
        """

        assert vid_id>=0 and vid_id<len(self.sequence_list), \
            'The id of the sequence should be in the range [0, {}]'.format(str(self.sequence_num-1))

        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        img_list = sequence['img_list']     # get the image list of this sequence
        labels = sequence['label_list']
        calib = sequence['calib']

        assert len(img_list) == len(labels), 'The number of image and number of labels do NOT match!'
        assert not(vis_2dbox == True and vis_3dbox == True), 'It is NOT good to visualize both 2D and 3D boxes simultaneously!'

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_2dbox:
                    save_path = os.path.join('./seq_camera_vis', sequence_name+'_2D_box')
                elif vis_3dbox:
                    save_path = os.path.join('./seq_camera_vis', sequence_name+'_3D_box')
                else:
                    save_path = os.path.join('./seq_camera_vis', sequence_name+'_no_box')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # show the sequence
        for img_name, img_label in zip(img_list, labels):
            img = cv.imread(img_name)   # BGR image format
            thickness = 2

            # visualize 2d boxes in the image
            if vis_2dbox:
                # load and show object bboxes
                for object in img_label:
                    object_type = object['object_type']
                    bbox = object['bbox']
                    bbox = [int(tmp) for tmp in bbox]
                    bbox_color = self.colors[self.categories.index(object_type)]
                    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])
                    cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=thickness)

                    cv.putText(img, text=object_type + '-ID: ' + str(object['id']), org=(bbox[0], bbox[1] - 5),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

            # visualize 3d boxes in the image
            if vis_3dbox:
                # load and show object bboxes
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = self.colors[self.categories.index(object_type)]
                    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])

                    corners_3d_img = transform_3dbox_to_image(object['dimension'], object['location'], object['rotation'], calib)

                    if corners_3d_img is None:
                        # None means object is behind the camera, and ignore this object.
                        continue
                    else:
                        corners_3d_img = corners_3d_img.astype(int)

                        # draw lines in the image
                        # p10-p1, p1-p2, p2-p3, p3-p0
                        cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

                        # p4-p5, p5-p6, p6-p7, p7-p0
                        cv.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                        # p0-p4, p1-p5, p2-p6, p3-p7
                        cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)

                        # draw front lines
                        cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                                (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                                (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                        cv.putText(img, text=object_type + '-ID: ' + str(object['id']), org=(corners_3d_img[4, 0], corners_3d_img[4, 1]-5),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

            cv.imshow('Play {}'.format(sequence['name']), img)
            # save visualization image if you want
            if save_img:
                cv.imwrite(os.path.join(save_path, img_name.split('/')[-1].split('.')[0] + '.png'), img)
            cv.waitKey(wait_time)

        cv.destroyAllWindows()

    def show_sequence_pointcloud(self, vid_id, img_region=False, vis_box=False, save_img=False, save_path=None):
        """
        visualize the sequence in point cloud
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_box: show 3D boxes or not
        :return: none
        """

        assert 0 <= vid_id < len(self.sequence_list), 'The sequence id should be in [0, {}]'.format(str(self.sequence_num - 1))
        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        pcloud_list = sequence['pcloud_list']
        labels = sequence['label_list']
        img_size = sequence['img_size']
        calib = sequence['calib']

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_pointcloud_vis', sequence_name+'_3D_box')
                else:
                    save_path = os.path.join('./seq_pointcloud_vis', sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # # load point cloud
        # pcloud = np.fromfile(pcloud_list[0], dtype=np.float32).reshape(-1, 4)
        #
        # pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        # plt = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)
        # # another way is to use animate function in mlab to play cloud
        # # but somehow, it sometimes works, but sometimes fails
        #
        # @mlab.animate(delay=100)
        # def anim():
        #     for i in range(1, len(pcloud_list)):
        #         pcloud_name = pcloud_list[i]
        #         print(pcloud_name)
        #         # load point cloud
        #         pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)
        #         plt.mlab_source.reset(x=pcloud[:, 0], y=pcloud[:, 1], z=pcloud[:, 2])
        #         mlab.savefig(filename='temp_img2/' + str(i) + '.png')
        #         yield
        #
        # anim()
        # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=50.0)
        # mlab.show()

        # visualization
        pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        for pcloud_name in pcloud_list:
            # clear
            mlab.clf()

            # BE CAREFUL!
            # the reason why doing so is because there are bin files missing in some sequences (e.g., sequence 0001)
            # e.g., in label file, the seuqnece is: 000001, 000002, 000003, 000004, 000005
            # but in bin file, the sequence is:     000001, 000004, 000005
            img_label = labels[int(pcloud_name.split('/')[-1].split('.')[0])]

            # load point cloud
            # point[:, 0]: x; point[:, 1]: y; point[:, 2]: z; point[:, 3]: reflectance information
            pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)

            # remove point clouds not in RBG image
            if img_region:
                # velodyne coordinate to camera 0 coordinate
                pcloud_C2_depth, pcloud_C2 = velodyne_to_camera_2(pcloud, calib)

                # remove points out of image
                pcloud_in_img = remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, img_size)
                pcloud = pcloud_in_img

            # show point cloud
            plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), mode='point', figure=pcloud_fig)
            # plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)

            # load and show 3d boxes
            if vis_box:
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = self.colors[self.categories.index(object_type)]
                    bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                    corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                    # draw lines
                    # a utility function to draw a line
                    def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=3, figure=fig)

                    # draw the bootom lines
                    draw_line_3d(corners_3d[0], corners_3d[1], bbox_color)
                    draw_line_3d(corners_3d[1], corners_3d[2], bbox_color)
                    draw_line_3d(corners_3d[2], corners_3d[3], bbox_color)
                    draw_line_3d(corners_3d[3], corners_3d[0], bbox_color)

                    # draw the up lines
                    draw_line_3d(corners_3d[4], corners_3d[5], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[6], bbox_color)
                    draw_line_3d(corners_3d[6], corners_3d[7], bbox_color)
                    draw_line_3d(corners_3d[7], corners_3d[4], bbox_color)

                    # draw the vertical lines
                    draw_line_3d(corners_3d[4], corners_3d[0], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[1], bbox_color)
                    draw_line_3d(corners_3d[6], corners_3d[2], bbox_color)
                    draw_line_3d(corners_3d[7], corners_3d[3], bbox_color)

                    # draw front lines
                    draw_line_3d(corners_3d[4], corners_3d[1], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[0], bbox_color)

                    mlab.text3d(x=corners_3d[5, 0], y=corners_3d[5, 1], z=corners_3d[5, 2], \
                                text=object_type+'-ID: '+str(object['id']), color=bbox_color, scale=0.35)

            # fix the view of the camera
            mlab.view(azimuth=180, distance=30, elevation=60, focalpoint=np.mean(pcloud, axis=0)[:-1])
            if save_img:
                mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
            else:
                mlab.savefig(filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file

    def show_sequence_BEV(self, vid_id, img_region=False, vis_box=False, save_img=False, save_path=None):
        """
        visualize the sequence in bird's eye view
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_3dbox: show 3D boxes or not
        :return: none
        """

        assert 0 <= vid_id < len(self.sequence_list), 'The sequence id should be in [0, {}]'.format(str(self.sequence_num - 1))
        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        pcloud_list = sequence['pcloud_list']
        labels = sequence['label_list']
        img_size = sequence['img_size']
        calib = sequence['calib']

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_BEV_vis', sequence_name + '_BEV_box')
                else:
                    save_path = os.path.join('./seq_BEV_vis', sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # visualization
        pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        for pcloud_name in pcloud_list:
            # clear
            mlab.clf()

            # BE CAREFUL!
            # the reason why doing so is because there are bin files missing in some sequences (e.g., sequence 0001)
            # e.g., in label file, the seuqnece is: 000001, 000002, 000003, 000004, 000005
            # but in bin file, the sequence is:     000001, 000004, 000005
            img_label = labels[int(pcloud_name.split('/')[-1].split('.')[0])]

            # load point cloud
            # point[:, 0]: x; point[:, 1]: y; point[:, 2]: z; point[:, 3]: reflectance information
            pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)

            # remove point clouds not in RBG image
            if img_region:
                # velodyne coordinate to camera 0 coordinate
                pcloud_C2_depth, pcloud_C2 = velodyne_to_camera_2(pcloud, calib)

                # remove points out of image
                pcloud_in_img = remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, img_size)
                pcloud = pcloud_in_img

            # show point cloud
            plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), mode='point', figure=pcloud_fig)
            # plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)

            # load and show 3d boxes
            if vis_box:
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = self.colors[self.categories.index(object_type)]
                    bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                    corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                    # draw lines
                    # a utility function to draw a line
                    def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=4, figure=fig)

                    # draw the lines in X-Y space
                    draw_line_3d(corners_3d[4], corners_3d[5], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[6], bbox_color)
                    draw_line_3d(corners_3d[6], corners_3d[7], bbox_color)
                    draw_line_3d(corners_3d[7], corners_3d[4], bbox_color)

                    mlab.text3d(x=corners_3d[7, 0], y=corners_3d[7, 1]-0.5, z=corners_3d[7, 2], \
                                text=object_type + '-ID: ' + str(object['id']), color=bbox_color, scale=0.7)

            # fix the view of the camera
            mlab.view(azimuth=180, distance=100, elevation=0, focalpoint=np.mean(pcloud, axis=0)[:-1])
            if save_img:
                mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
            else:
                mlab.savefig(filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file


# # This is for debug
# if __name__ == '__main__':
#     kitti_path = '/mnt/Data/dataset/3D single object tracking/KITTI'
#     kitti = KITTI(kitti_path)
#     # kitti.show_sequence_rgb(1, vis_2dbox=False, vis_3dbox=True, save_img=True)
#     # kitti.show_sequence_pointcloud(1, img_region=False, vis_box=True, save_img=True)
#     # kitti.show_sequence_BEV(0, vis_box=True, save_img=True)
#     print('end!')
