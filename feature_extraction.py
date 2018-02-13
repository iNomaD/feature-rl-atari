import itertools
import os
import sys

import cv2
import numpy as np
from scipy.misc import imread

import visualization


class FeatureVector:

    def __init__(self, n_class_instances, feature_vector_size):
        self.n_class_instances = n_class_instances
        self.raw_size = feature_vector_size
        self.vector = []

    def fill_next_class(self, n_alive, object_list, nearest_entity):
        self.vector.append((n_alive, object_list, nearest_entity))

    def to_raw(self):
        assert len(self.vector) == len(self.n_class_instances)
        raw_vector = []
        j = 0
        for (n_alive, object_list, nearest_entity) in self.vector:
            raw_vector.append(n_alive)  # n
            for obj in object_list:
                raw_vector.append(obj[0])  # x
                raw_vector.append(obj[1])  # y
                raw_vector.append(obj[2])  # vx
                raw_vector.append(obj[3])  # vy
            for i in range(self.n_class_instances[j] - len(object_list)):
                raw_vector.append(0.0)  # x
                raw_vector.append(0.0)  # y
                raw_vector.append(0.0)  # vx
                raw_vector.append(0.0)  # vy
            if j != 0:  # skip the first class (player)
                if nearest_entity != None:
                    raw_vector.append(nearest_entity[0])  # player_x - x
                    raw_vector.append(nearest_entity[1])  # player_y - y
                    raw_vector.append(nearest_entity[2])  # player_vx - vx
                    raw_vector.append(nearest_entity[3])  # player_vy - vy
                else:
                    raw_vector.append(0.0)  # player_x - x
                    raw_vector.append(0.0)  # player_y - y
                    raw_vector.append(0.0)  # player_vx - vx
                    raw_vector.append(0.0)  # player_vy - vy
            j = j + 1

        assert len(raw_vector) == self.raw_size
        return raw_vector

    def get_class(self, i):
        assert len(self.vector) == len(self.n_class_instances)
        return self.vector[i]


class ImageProcessor:

    def __init__(self, env_id, frames_to_images=False, debug=False, save_best=False):
        self.env_id = env_id

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
        self.feature_vector_size = 1+5*(len(self.n_class_instances)-1)+4*sum(self.n_class_instances)
        vector = FeatureVector(self.n_class_instances, self.feature_vector_size)
        for i in range(0, len(self.classes)):
            vector.fill_next_class(0, [], None)
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

        self.ipp = visualization.ImagePostProcessor(env_id, self.height_range, frames_to_images, debug, save_best)

        # width and height for normalization
        self.width = 160
        self.height = self.height_range[1] - self.height_range[0]

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


    def pipeline(self, original_image):
        cropped_image = self.crop_image(original_image)
        instances = self.detect_instances(cropped_image)
        features = self.generate_feature_vector(instances)
        self.ipp.post_pipeline(original_image, instances, features)
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
        instances = []
        for cls in self.classes:
            obj = self.find_objects(image, cls)
            instances.append(obj)
        return instances


    def find_objects(self, image, templates):
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
                    res.append((x, y, width, height))
                    break
        return res


    def generate_feature_vector(self, instances):
        vector = FeatureVector(self.n_class_instances, self.feature_vector_size)
        p_vector = self.previous_vector

        # assume the first instance of the first class is the player
        player_alive = len(instances[0]) != 0
        if player_alive:
            player_position = instances[0][0]
            x = player_position[0] + player_position[2] / 2
            y = player_position[1] + player_position[3] / 2
            player_x = round(2 * x / self.width - 1, 3)
            player_y = round(2 * y / self.height - 1, 3)
            prev_player_position = p_vector.get_class(0)[1]
            if prev_player_position == []:
                player_vx = 0
                player_vy = 0
            else:
                player_vx = round(player_x - prev_player_position[0][0], 3)
                player_vy = round(player_y - prev_player_position[0][1], 3)
            vector.fill_next_class(1, [(player_x, player_y, player_vx, player_vy)], None)
        else:
            player_vx = 0
            player_vy = 0
            vector.fill_next_class(0, [], None)

        # fill positions and speeds of all other objects
        for c in range(1, len(instances)):
            detected_instances = instances[c]
            prev_instances_positions = p_vector.get_class(c)[1]
            n_max = self.n_class_instances[c]
            n_alive = len(detected_instances)

            # track the game entity nearest to the player
            nearest_entity = None
            max_player_dist = sys.maxsize

            # actual values
            object_list = []
            i = 0
            for position in detected_instances:
                # position = instance_class[i]
                x = position[0] + position[2] / 2
                y = position[1] + position[3] / 2

                # normalize position to [-1, 1] with 3 digits after .
                x = round(2 * x / self.width - 1, 3)
                y = round(2 * y / self.height - 1, 3)

                # find the nearest instance from the previous frame
                vx = 0
                vy = 0
                max_dist = sys.maxsize
                for prev_position in prev_instances_positions:
                    dist_to_other = (prev_position[0] - x)*(prev_position[0] - x) + (prev_position[1] - y)*(prev_position[1] - y)
                    if dist_to_other < max_dist:
                        max_dist = dist_to_other
                        vx = round(x - prev_position[0], 3)
                        vy = round(y - prev_position[1], 3)

                # append to list only if it's not full
                if i < n_max:
                    object_list.append((x, y, vx, vy))
                    i = i + 1

                # find nearest game entity if player exists
                if player_alive:
                    dist_to_player = (player_x - x)*(player_x - x) + (player_y - y)*(player_y - y)
                    if dist_to_player < max_player_dist:
                        max_player_dist = dist_to_player
                        nearest_entity = (x, y, vx, vy)

            # push the nearest entity to the front
            if nearest_entity != None:
                # find if it exists
                index_to_pop = -1
                for i in range(len(object_list)):
                    obj = object_list[i]
                    if np.isclose(obj, nearest_entity).all():
                        index_to_pop = i
                if index_to_pop != -1:
                    if index_to_pop != 0:
                        object_list.pop(index_to_pop)
                        object_list.insert(0, nearest_entity)
                else:  # index_to_pop doesn't exist <=> i == n_max
                    a = object_list.pop()
                    object_list.insert(0, nearest_entity)

            # calculate relative position and velocity
            if nearest_entity != None:
                x_relative = round(player_x - nearest_entity[0], 3)
                y_relative = round(player_y - nearest_entity[1], 3)
                vx_relative = round(player_vx - nearest_entity[2], 3)
                vy_relative = round(player_vy - nearest_entity[3], 3)
                nearest_entity = (x_relative, y_relative, vx_relative, vy_relative)

            vector.fill_next_class(n_alive, object_list, nearest_entity)

        self.previous_vector = vector
        return vector

