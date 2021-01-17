import numpy as np
import cv2


class PCA_01:

    def __init__(self, image_matrix, image_labels, image_targets, no_of_elements, img_largura, img_altura, quality_percent):
        self.images_matrix = image_matrix
        self.image_labels = image_labels
        self.image_targets = image_targets
        self.no_of_elements = no_of_elements
        self.img_largura = img_largura
        self.img_altura = img_altura
        self.quality_percent = quality_percent

        self.mean_face = np.mean(self.images_matrix, 0)
        self.images_mean_substracted = self.images_matrix - self.mean_face

    def give_P(self, eig_vals):
        sum_original = np.sum(eig_vals)
        sum_threshold = sum_original * self.quality_percent/100
        sum_temp = 0
        P = 0
        while sum_temp < sum_threshold:
            sum_temp += eig_vals[P]
            P += 1
        return P

    def reduce_dim(self):
        no_of_images = self.images_matrix.shape[0]
        g_t = np.zeros((self.img_altura, self.img_largura))

        for i in range(no_of_images):
            temp_gt = np.dot(self.images_mean_substracted[i].T, self.images_mean_substracted[i])
            g_t += temp_gt

        g_t /= no_of_images
        e_vals, e_vec = np.linalg.eig(g_t)
        P = self.give_P(eig_vals=e_vals)
        self.new_bases = e_vec[:, 0:P]
        self.new_coordinates = np.dot(self.images_matrix, self.new_bases)
        return self.new_coordinates

    def new_cords(self, single_image):
        return np.dot(single_image, self.new_bases)

    def show_image(self, label_to_show, old_cords):
        old_cords_matrix = np.reshape(old_cords, [self.img_largura, self.img_altura])
        old_cords_integers = np.array(old_cords_matrix, dtype=np.uint8)
        resized_image = cv2.resize(old_cords_integers, (500, 500))
        cv2.imshow(
            label_to_show, resized_image)
        cv2.waitKey()

    def img_from_path(self, path):
        gray = cv2.imread(path, 0)
        return cv2.resize(gray, (self.img_largura, self.img_altura))

    def recognize_face(self, new_cords_of_image):

        no_of_images = len(self.image_labels)
        distance = []
        for i in range(no_of_images):
            temp_img = self.new_coordinates[i]
            temp_dist = np.linalg.norm(new_cords_of_image - temp_img)
            distance += [temp_dist]

        min = np.argmin(distance)
        label = self.image_labels[min]
        return self.image_targets[label]
