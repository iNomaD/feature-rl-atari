import itertools
import os
import numpy as np
from scipy.misc import toimage, imread, imsave
import cv2


class FeatureVector:

    def __init__(self, n_class_instances, feature_vector_size):
        self.n_class_instances = n_class_instances
        self.raw_size = feature_vector_size
        self.vector = []

    def fill_next_class(self, n_alive, object_list):
        self.vector.append((n_alive, object_list))

    def to_raw(self):
        raw_vector = []
        for (n_alive, object_list) in self.vector:
            raw_vector.append(n_alive)  # n
            for obj in object_list:
                if obj == []:
                    raw_vector.append(0.0)  # x
                    raw_vector.append(0.0)  # y
                    raw_vector.append(0.0)  # vx
                    raw_vector.append(0.0)  # vy
                else:
                    raw_vector.append(obj[0])  # x
                    raw_vector.append(obj[1])  # y
                    raw_vector.append(obj[2])  # vx
                    raw_vector.append(obj[3])  # vy

        assert len(raw_vector) == self.raw_size
        return raw_vector

    def get_class(self, i):
        assert len(self.vector) == len(self.n_class_instances)
        return self.vector[i]


class ImageProcessor:

    def __init__(self, env_id, frames_to_images):
        self.env_id = env_id
        self.frames_to_images = frames_to_images
        self.frame = 0

        # load classes of game objects
        self.classes = []
        self.n_class_instances = []
        class_path = 'res/classes/' + env_id
        class_subdirs = os.listdir(class_path)
        for subdir in class_subdirs:
            class_filenames = os.listdir(class_path + '/' + subdir)
            max_instances = 1
            contents = []
            for file in class_filenames:
                if file.endswith('.max'):
                    max_instances = int(os.path.splitext(file)[0])
                else:
                    contents.append(imread(class_path + '/' + subdir + '/' + file))
            self.classes.append(contents)
            self.n_class_instances.append(max_instances)

        # generate initial feature vector
        self.feature_vector_size = len(self.n_class_instances)+4*sum(self.n_class_instances)
        vector = FeatureVector(self.n_class_instances, self.feature_vector_size)
        for i in range(0, len(self.classes)):
            object_list = []
            for j in range(0, self.n_class_instances[i]):
                object_list.append([])
            vector.fill_next_class(0, object_list)
        self.previous_vector = vector

        # hardcoded sizes of actual game screen (speedup preprocessing)
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

        # try to load background (optional)
        try:
            bg_load = imread('res/background/'+env_id+"-bg.bmp")
            self.bg = self.crop_image(bg_load)
            self.bg_r = self.bg[..., 0]
            self.bg_g = self.bg[..., 1]
            self.bg_b = self.bg[..., 2]
        except:
            self.bg = None

    def get_feature_vector_size(self):
        return self.feature_vector_size

    def pipeline(self, image):
        image = self.crop_image(image)
        instances = self.detect_instances(image)
        features = self.generate_feature_vector(instances)
        print(features.to_raw())
        return features.to_raw()

    def crop_image(self, image):
        h_beg, h_end = self.height_range
        return image[h_beg:h_end, ...]

    def remove_background(self, image):
        assert image.shape == self.bg.shape
        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]
        cond = (R == self.bg_r) & (G == self.bg_g) & (B == self.bg_b)
        image[cond] = [0, 0, 0]
        return image

    def detect_instances(self, image):
        palette = np.copy(image)
        instances = []
        for cls in self.classes:
            obj = self.find_objects(image, cls, palette)
            instances.append(obj)

        #toimage(image).show()
        if self.frames_to_images:
            if not os.path.exists('frames'):
                os.makedirs('frames')
            imsave('frames/'+str(self.frame)+'.png', palette)
            self.frame = self.frame + 1

        return instances

    def find_objects(self, image, templates, palette):
        # assume all the templates of the same class are the same color
        template = templates[0]
        h = template.shape[0]
        w = template.shape[1]
        for ii, jj in itertools.product(range(0, h), range(0, w)):
            if not np.array_equal(template[ii, jj], [0, 0, 0]):
                # assume color of the object is constant
                t_r = template[ii, jj, 0]
                t_g = template[ii, jj, 1]
                t_b = template[ii, jj, 2]
                break

        # apply color filter and kill background
        H, W, _ = image.shape
        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]
        cond = (R == t_r) & (G == t_g) & (B == t_b)
        if self.bg is not None:
            cond = cond & ~((R == self.bg_r) & (G == self.bg_g) & (B == self.bg_b))
        pic = np.zeros((H, W), dtype=np.uint8)
        pic[cond] = 255

        # detect contours
        pic, conts, hierarchy = cv2.findContours(pic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        for i in range(len(conts)):
            x, y, width, height = cv2.boundingRect(conts[i])
            for tpl in templates:
                if tpl.shape[0] == height and tpl.shape[1] == width:
                    cv2.rectangle(palette, (x-1, y-1), (x + width, y + height), (0, 255, 0), 1)
                    res.append((x, y, width, height))
                    break
        return res

    def generate_feature_vector(self, instances):
        vector = FeatureVector(self.n_class_instances, self.feature_vector_size)
        p_vector = self.previous_vector

        for c in range(0, len(instances)):
            instance_class = instances[c]
            p_instance_class = p_vector.get_class(c)
            n_max = self.n_class_instances[c]
            n_alive = len(instance_class)

            # actual values
            object_list = []
            i = 0
            while i < len(instance_class) and i < n_max:
                position = instance_class[i]
                p_position = p_instance_class[1][i]
                x = position[0] + position[2] / 2
                y = position[1] + position[3] / 2
                if p_position == []:
                    object_list.append((x, y, 0, 0))
                else:
                    vx = x - p_position[0]
                    vy = y - p_position[1]
                    object_list.append((x, y, vx, vy))
                i = i + 1

            # missing values
            while i < n_max:
                object_list.append([])
                i = i + 1

            vector.fill_next_class(n_alive, object_list)

        self.previous_vector = vector
        return vector

