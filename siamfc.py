#!/usr/bin/python
#VOT toolkit integration for Siamese FC


import vot
import sys
import time
from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox

class SiamFCTracker(object):
    def __init__(self, image, gt):
        # avoid printing TF debugging information
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # TODO: allow parameters from command line or leave everything in json files?
        hp, evaluation, run, env, design = parse_arguments()
        # Set size for use with tf.image.resize_images with align_corners=True.
        # For example,
        #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
        # instead of
        # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
        final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        # build TF graph once for all
        filename, image, templates_z, scores, z_crops, x_crops, anchor_coord = siam.build_tracking_graph(final_score_sz,  design, env)


    def track():

        gt, frame_name_list, frame_sz, _ = _init_video(env, evaluation, evaluation.video)
        pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
        bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
                                filename, image, templates_z, scores, evaluation.start_frame, evaluation.video,
                                frame_sz, z_crops, x_crops, anchor_coord)





    def _init_video(env, evaluation, video):
        video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
        frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
        frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
        frame_name_list.sort()
        with Image.open(frame_name_list[0]) as img:
            frame_sz = np.asarray(img.size)
            frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

        # read the initialization from ground truth
        gt_file = os.path.join(video_folder, 'groundtruth.txt')
        gt = np.genfromtxt(gt_file, delimiter=',')
        n_frames = len(frame_name_list)
        assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

        return gt, frame_name_list, frame_sz, n_frames


handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    handle.report(selection, confidence)
    time.sleep(0.01)