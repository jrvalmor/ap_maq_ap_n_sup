import os
import cv2
import numpy as np

class imgParaMatrizClass:

    def __init__(self, images_paths, largura_img, altura_img):
        self.images_paths = images_paths
        self.largura_img = largura_img
        self.altura_img = altura_img

    def get_matrix(self):
        img_mat = np.zeros(
            (len(self.images_paths), self.altura_img, self.largura_img), dtype=np.uint8
        )

        i = 0
        for path in self.images_paths:
            gray = cv2.imread(path, 0)
            gray_scaled = cv2.resize(gray, (self.largura_img, self.altura_img))
            mat = np.asmatrix(gray_scaled)
            img_mat[i, :, :] = mat
            i += 1
        return img_mat


class dadosClass:

    def __init__(self, required_no):
        dir = "ORL" #+ dataset_name

        self.images_path_for_training = []
        self.labels_for_training = []
        self.no_of_images_for_training = []

        self.images_path_for_testing = []
        self.labels_for_testing = []
        self.no_of_images_for_testing = []

        self.images_target = []

        per_no = 0
        for name in os.listdir(dir):
            dir_path = os.path.join(dir, name)
            if os.path.isdir(dir_path):
                if len(os.listdir(dir_path)) >= required_no:
                    i = 0
                    for img_name in os.listdir(dir_path):
                        img_path = os.path.join(dir_path, img_name)

                        if i < required_no:
                            self.images_path_for_training += [img_path]
                            self.labels_for_training += [per_no]

                            if len(self.no_of_images_for_training) > per_no:
                                self.no_of_images_for_training[per_no] += 1
                            else:
                                self.no_of_images_for_training = [1]

                            if i is 0:
                                self.images_target += [name]

                        else:
                            self.images_path_for_testing += [img_path]
                            self.labels_for_testing += [per_no]

                            if len(self.no_of_images_for_testing) > per_no:
                                self.no_of_images_for_testing[per_no] += 1
                            else:
                                self.no_of_images_for_testing += [1]
                        i += 1

                    per_no += 1
