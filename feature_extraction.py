import itertools
import os
import numpy as np
from scipy.misc import toimage, imread, imsave
import cv2

class ImageProcessor():

    def __init__(self, env_id):
        self.env_id = env_id
        #self.bg = imread('res/background/'+env_id+"-bg.bmp")
        #self.bg_r = self.bg[..., 0]
        #self.bg_g = self.bg[..., 1]
        #self.bg_b = self.bg[..., 2]
        self.frame = 0
        self.classes = []
        class_path = 'res/classes/' + env_id
        class_subdirs = os.listdir(class_path)
        for subdir in class_subdirs:
            class_filenames = os.listdir(class_path + '/' + subdir)
            contents = []
            for file in class_filenames:
                contents.append(imread(class_path + '/' + subdir + '/' + file))
            self.classes.append(contents)

        if self.env_id == "MsPacman-v0":
            self.height_range = (0, 172)
        elif self.env_id == "Pong-v0":
            self.height_range = (35, 193)
        elif self.env_id == "Breakout-v0":
            self.height_range = (20, 198)
        elif self.env_id == "SpaceInvaders-v0":
            self.height_range = (20, 198)
        else:
            self.height_range = (0, 210)

    def pipeline(self, image):
        #image = self.remove_background(image)
        image = self.crop_image(image)
        image = self.detect_instances(image)
        #toimage(image).show()

    def crop_image(self, image):
        h_beg, h_end = self.height_range
        return image[h_beg:h_end, ...]

    def detect_instances(self, image):
        palette = np.copy(image)
        features = []
        for templates in self.classes:
            obj = self.find_objects(image, templates, palette)
            features.append(obj)
            #print("objects: " + str(len(obj)))

        #toimage(image).show()
        #imsave('output/frame'+str(self.frame)+'.bmp', palette)
        self.frame = self.frame + 1

    def find_objects(self, image, templates, palette):
        template = templates[0]
        h = template.shape[0]
        w = template.shape[1]
        for ii, jj in itertools.product(range(0, h), range(0, w)):
            if not np.array_equal(template[ii, jj], [0, 0, 0]):
                t_r = template[ii, jj, 0]
                t_g = template[ii, jj, 1]
                t_b = template[ii, jj, 2]
                break

        H, W, _ = image.shape
        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]
        cond = (R == t_r) & (G == t_g) & (B == t_b)
        pic = np.zeros((H, W), dtype=np.uint8)
        pic[cond] = 255

        pic, conts, hierarchy = cv2.findContours(pic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        for i in range(len(conts)):
            x, y, width, height = cv2.boundingRect(conts[i])
            for tpl in templates:
                if tpl.shape[0] == height and tpl.shape[1] == width:
                    cv2.rectangle(palette, (x-1, y-1), (x + width, y + height), (255, 0, 0), 1)
                    res.append((x, y, width, height))
                    break
        return res

    def findMatches(self, img, tpl, percent):
        H = img.shape[0]
        W = img.shape[1]
        h = tpl.shape[0]
        w = tpl.shape[1]

        mask = np.empty((h, w), dtype=bool)
        tpl_size = 0
        for ii in range(0, h):
            for jj in range(0, w):
                if np.array_equal(tpl[ii, jj], [0, 0, 0]):
                    mask[ii, jj] = False
                else:
                    mask[ii, jj] = True
                    tpl_size = tpl_size + 1

        res = []
        for i in range(0, H-h+1):
            for j in range(0, W-w+1):
                count = 0
                for ii, jj in itertools.product(range(0, h), range(0, w)):
                    if mask[ii, jj]:
                        if np.array_equal(img[i+ii, j+jj], tpl[ii, jj]):
                            count = count + 1
                            if count >= tpl_size * percent:
                                res.append((i, j))
                                break
                        else:
                            break
        return res

    def remove_background(self, image):
        assert image.shape == self.bg.shape
        H, W, _ = image.shape
        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]
        cond = (R == self.bg_r) & (G == self.bg_g) & (B == self.bg_b)
        image[cond] = [0, 0, 0]
        return image
