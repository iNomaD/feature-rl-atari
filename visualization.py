import os
import sys

import shutil
import numpy as np
from scipy.misc import imsave

class ImagePostProcessor:

    def __init__(self, env_id, height_range, frames_to_images=False, debug=False, save_best=False):
        self.env_id = env_id
        self.extra_height = height_range[0]
        self.frames_to_images = frames_to_images
        self.debug = debug
        self.save_best = save_best
        self.frame_counter = 0
        self.best_score = -sys.maxsize -1
        self.history = []
        self.reward_sum = 0


    def post_pipeline(self, original_image, reward, terminal, instances, features):
        if self.debug:
            print(features.to_raw())

        if self.frames_to_images or self.save_best:
            palette = np.copy(original_image)
            for cls in instances:
                for (x, y, width, height) in cls:
                    drawrect(palette, (x - 1, y - 1 + self.extra_height), (x + width, y + height + self.extra_height), (0, 255, 0))

            if self.frames_to_images:
                if not os.path.exists('frames'):
                    os.makedirs('frames')
                imsave('frames/' + str(self.frame_counter) + '.png', palette)
                self.frame_counter += 1

            if self.save_best:
                self.history.append(palette)
                self.reward_sum += reward
                if terminal:
                    if self.reward_sum > self.best_score:
                        self.best_score = self.reward_sum
                        self.save_best_result()
                    self.history.clear()
                    self.reward_sum = 0


    def save_best_result(self):
        print('New best result!')
        env_path = 'best/' + self.env_id + '/'
        if not os.path.exists('best'):
            os.makedirs('best')
        if not os.path.exists(env_path):
            os.makedirs(env_path)
        else:
            shutil.rmtree(env_path)
            os.makedirs(env_path)
        frame_counter = 0
        for frame in self.history:
            imsave(env_path + str(frame_counter) + '.png', frame)
            frame_counter += 1


def drawrect(img,pt1,pt2,color):
    pts = []

    # top/bottom lines
    for x in range(pt1[0], pt2[0], 2):
        pts.append((x, pt1[1]))
        pts.append((x, pt2[1]))
    pts.append((pt2[0], pt1[1]))
    pts.append((pt2[0], pt2[1]))

    # left/right line
    for y in range(pt1[1], pt2[1], 2):
        pts.append((pt1[0], y))
        pts.append((pt2[0], y))
    pts.append((pt1[0], pt2[1]))
    pts.append((pt2[0], pt2[1]))

    for point in pts:
        if point[1] >= 0 and point[1] < img.shape[0] and point[0] >= 0 and point[0] < img.shape[1]:
            img[point[1], point[0]] = color