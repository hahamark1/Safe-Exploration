import pygame
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

import cv2

VIDEO_FILE = 'VIDEO'

def make_video(screen, _image_num, video_path):
    # _image_num += 1
    str_num = "000" + str(_image_num)
    file_name = "{}/{}.jpg".format(video_path, str_num[-4:])
    pygame.image.save(screen, file_name)

def create_video_from_images(video_path, experiment_name, outimg=None, fps=60, size=None,
               is_color=True, format='XVID'):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*'XVID')
    vid = None

    vid_path = '{}/videos/'.format(os.path.dirname(os.path.dirname(video_path)))

    if not os.path.exists(vid_path):
        os.makedirs(vid_path)
    print(vid_path)

    outvid = '{}{}.avi'.format(vid_path, experiment_name)
    print(outvid)

    for file in sorted(os.listdir(video_path)):
        file_path = os.path.join(video_path, file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(image)

        img = imread(file_path)

        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    cv2.destroyAllWindows()
    # return vid
