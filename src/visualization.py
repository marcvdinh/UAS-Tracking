import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def show_frame(frame, bbox, bbox_detection, frame_number, video, fig_n):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    r = patches.Rectangle((bbox_detection[0], bbox_detection[1]), bbox_detection[2], bbox_detection[3], linewidth=2, edgecolor='b', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    #for box in bbox_detection:
    #    r = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='b', fill=False)
    #    ax.imshow(np.uint8(frame))
    #    ax.add_patch(r)
    #plt.ion()
    plt.savefig("./data/local_detector/" + str(video)+"/" + str(frame_number) + ".png")
    #plt.show()
    #plt.pause(0.001)
    plt.clf()

def show_detection(frame, bbox, fig_n):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)

    for box in bbox:
        r = patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=2, edgecolor='b', fill=False)
        ax.imshow(np.uint8(frame))
        ax.add_patch(r)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()

def show_crops(crops, bbox, frame_number,video, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(132)
    #ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    #ax2.imshow(np.uint8(crops[1,:,:,:]))
    #ax3.imshow(np.uint8(crops[2,:,:,:]))
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='b', fill=False)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax1.add_patch(r)
    #for box in bbox:
    #    r = patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=2, edgecolor='b', fill=False)
    #    ax1.imshow(np.uint8(crops[0,:,:,:]))
    #    ax1.add_patch(r)
    plt.ion()
    plt.savefig("./data/crops/" + str(video) + "/" + str(frame_number) + ".png")
    plt.show()
    plt.pause(0.001)
    plt.clf()


def show_scores(score, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)

def show_score(score,frame_number,video, fig_n):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    ax.imshow(score, interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.savefig("./data/score/" + str(video) + "/" + str(frame_number) + ".png")
    plt.pause(0.001)