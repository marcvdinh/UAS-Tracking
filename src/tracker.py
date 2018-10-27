
import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time
import json
import utils
from scipy.misc import imresize

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores, show_score
from darkflow.net.build import TFNet

# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

# read default parameters and override with custom ones
def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame, video_name, frame_sz, z_crops, x_crops, anchor_coord):
    num_frames = np.size(frame_name_list) - start_frame
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames,4))
    reinitialize = False
    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz

    #detector settings
    options = {"model": "/home/mdinh/siamfc-tf/cfg/yolo-mio.cfg", "pbLoad": "/home/mdinh/siamfc-tf/built_graph/yolo-mio.pb", "metaLoad": "/home/mdinh/siamfc-tf/built_graph/yolo-mio.meta", "gpu": 0.4, "threshold": 0.4}

    tfnet = TFNet(options)
    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}

    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # save first frame position (from ground-truth)
        bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

        image_, templates_z_ = sess.run([image, templates_z], feed_dict={
                                                                        siam.pos_x_ph: pos_x,
                                                                        siam.pos_y_ph: pos_y,
                                                                        siam.z_sz_ph: z_sz,
                                                                        filename: frame_name_list[start_frame]})
        new_templates_z_ = templates_z_

        t_start = time.time()

        # Get an image from the queue
        for i in range(1, num_frames):
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors
            image_, scores_, x_crops_ , anchor_coord__= sess.run(
                [image, scores, x_crops, anchor_coord],
                feed_dict={
                    siam.pos_x_ph: pos_x,
                    siam.pos_y_ph: pos_y,
                    siam.x_sz0_ph: scaled_search_area[0],
                    siam.x_sz1_ph: scaled_search_area[1],
                    siam.x_sz2_ph: scaled_search_area[2],
                    templates_z: np.squeeze(templates_z_),
                    filename: frame_name_list[i + start_frame],
                }, **run_opts)
            scores_ = np.squeeze(scores_)
            # penalize change of scale
            scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
            scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
            # update scaled sizes
            x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]
            target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
            target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
            # select response with new_scale_id
            score_ = scores_[new_scale_id,:,:]

            max_score = score_.max()
            min_score = score_.min()
            score_augmentation = np.full((257, 257), float(0))
            #g = utils.gaussian_kernel(128,max_score, 2 * (target_w + target_h))
            #run detector for drift

            if abs(pos_x) > frame_sz[1] or abs(pos_y) > frame_sz[0] or pos_x < 0 or pos_y < 0:
                print('OOB')
                OOB = True
                reinitialize = True
                best_detection = []


            else:
                OOB = False
                bbox_crop = []
                best_detection = []


            if OOB == False:
                result_crop = tfnet.return_predict(x_crops_[new_scale_id, :, :])
                print (result_crop)


            if result_crop:
                maxdetection_crop = max(result_crop, key=lambda x: x['confidence'])
                reinitialize = True



                for detection in result_crop:
                    bbox_crop.append([detection['topleft']['x'], detection['topleft']['y'],
                                          detection['bottomright']['x'] - detection['topleft']['x'],
                                          detection['bottomright']['y'] - detection['topleft']['y']])
                    best_crop = ([maxdetection_crop['topleft']['x'], maxdetection_crop['topleft']['y'],
                                      maxdetection_crop['bottomright']['x'] - maxdetection_crop['topleft']['x'],
                                      maxdetection_crop['bottomright']['y'] - maxdetection_crop['topleft']['y']])

                    peak_x = (maxdetection_crop['bottomright']['x'] + maxdetection_crop['topleft']['x']) / 2
                    peak_y = (maxdetection_crop['bottomright']['y'] + maxdetection_crop['topleft']['y']) / 2

                    target_h_crop = best_crop[3]
                    target_w_crop = best_crop[2]
                    print([peak_x,peak_y])
                    if peak_x in range(200,300) and peak_y in range(200,300): #ignore edge results (gaussian curve)

                        print("using detection")

                        # generate an augmentation map
                        score_augmentation = np.full((design.search_sz, design.search_sz), float(min_score))
                        for x in range(int(peak_x - 5), int(peak_x + 5)):
                            for y in range(int(peak_y - 5), int(peak_y + 5)):
                                score_augmentation[y, x] = max_score

                    score_augmentation = imresize(score_augmentation, (final_score_sz,final_score_sz),'bicubic')
                    #np.multiply(score_augmentation,g, score_augmentation)

                    bbox_frame = [ element * scaled_search_area[new_scale_id] / design.search_sz for element in best_crop ]
                    #transform from crop coordinate to frame coordinate
                    bbox_frame[0] = best_crop[0] + anchor_coord__[new_scale_id, 0]
                    bbox_frame[1] = best_crop[1] + anchor_coord__[new_scale_id, 1]

                    #print(anchor_coord__[new_scale_id, 0])
                    #print(anchor_coord__[new_scale_id, 1])

                    #pos_x = 2 * peak_x * scaled_search_area[new_scale_id] / 512
                    #pos_y = 2 * peak_y * scaled_search_area[new_scale_id] / 512

                    #target_h = (bbox_frame[3] + target_h) / 2
                    #target_w = (bbox_frame[2] + target_w) / 2
                    print("small scale detection :")
                    print([bbox_frame[0], bbox_frame[1]])
                    #pos_x = bbox_frame[0]
                    #pos_y = bbox_frame[1]
                    target_h = ((hp.scale_lr) * bbox_frame[3] + (1 - hp.scale_lr) * target_h)
                    target_w = ((hp.scale_lr) * bbox_frame[2] + (1 - hp.scale_lr) * target_w)
                    #reinitialize = False





            #update score map

            score_updated = (1 - hp.scale_lr)*score_ + (hp.scale_lr) *score_augmentation
            score_updated = score_updated - np.min(score_updated)
            score_updated = score_updated / np.sum(score_updated)
            # apply displacement penalty
            score_updated = (1 - hp.window_influence) * score_updated + hp.window_influence * penalty
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_updated, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            print(pos_x,pos_y)

            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            bbox_detection = [0, 0, 0, 0]


            # run detector on whole frame


            result = tfnet.return_predict(image_)




             #print ([sorted([object['confidence'] for object in result])])
            if len(result) > 0:
                maxdetection = max(result, key=lambda x: x['confidence'])

                #mindetection = min(result, key=lambda x: x['confidence'])
                #for detection in result:
                #    bbox_detection.append([detection['topleft']['x'],detection['topleft']['y'], detection['bottomright']['x'] - detection['topleft']['x'], detection['bottomright']['y'] - detection['topleft']['y']])

                best_detection = ([maxdetection['topleft']['x'], maxdetection['topleft']['y'],
                                      maxdetection['bottomright']['x'] - maxdetection['topleft']['x'],
                                      maxdetection['bottomright']['y'] - maxdetection['topleft']['y']])

                print("large scale detection :")
                print ([best_detection[0], best_detection[1]])

                iou = utils.iou(best_detection, bbox_frame)
                print(iou)

                if iou > 0.5 and iou > 0:
                    reinitialize = True




                # update the target representation with a rolling average
            if hp.z_lr>0:
                new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                siam.pos_x_ph: pos_x,
                                                                siam.pos_y_ph: pos_y,
                                                                siam.z_sz_ph: z_sz,
                                                                image: image_
                                                                })

                if reinitialize == True:
                    print(reinitialize)

                    # assign new target height and width
                    if iou < 0.5 and iou > 0 or OOB == True:
                        OOB = False
                        iou = 0
                        bboxes[i, :] = best_detection
                        target_h = best_detection[3]
                        target_w = best_detection[2]
                        pos_x = best_detection[0] + best_detection[2]/2
                        pos_y = best_detection[1] + best_detection[3]/2


                        print("large scale reinit")


                    context = design.context * (target_w + target_h)
                    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
                    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

                    # thresholds to saturate patches shrinking/growing
                    min_z = hp.scale_min * z_sz
                    max_z = hp.scale_max * z_sz
                    min_x = hp.scale_min * x_sz
                    max_x = hp.scale_max * x_sz
                    #compute new template
                    new_templates_z_ = sess.run([templates_z], feed_dict={
                        siam.pos_x_ph: pos_x,
                        siam.pos_y_ph: pos_y,
                        siam.z_sz_ph: z_sz,
                        image: image_
                    })
                    templates_z_ = np.asarray(new_templates_z_)
                    print("using new template")

                    reinitialize = False
                else:
                    templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
                    reinitialize = False
                    #print(reinitialize)






            # update template patch size
            z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]

            if run.visualization:
                show_frame(image_, bboxes[i,:],bbox_detection, i, video_name,1)
                #show_crops(x_crops_, best_crop,i,video_name, 2)
                #show_scores(scores_,1)
                #show_score(score_,i, video_name,1)
                #show_score(score_augmentation, i, video_name,1)
                #show_score(score_updated,i, video_name, 2)


        #end of loop
        t_elapsed = time.time() - t_start
        speed = num_frames/t_elapsed

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline-search.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

    plt.close('all')

    return bboxes, speed


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


