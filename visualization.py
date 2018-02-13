import os

import cv2
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
        self.best_score = 0


    def post_pipeline(self, original_image, instances, features):
        if self.debug:
            print(features.to_raw())

        if self.frames_to_images:
            palette = np.copy(original_image)
            for cls in instances:
                for (x, y, width, height) in cls:
                    cv2.rectangle(palette, (x - 1, y - 1 + self.extra_height), (x + width, y + height + self.extra_height), (0, 255, 0), 1)

            if not os.path.exists('frames'):
                os.makedirs('frames')
            imsave('frames/' + str(self.frame_counter) + '.png', palette)
            self.frame_counter += 1
