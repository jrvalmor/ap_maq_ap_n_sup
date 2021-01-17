import cv2
from Principal import PCA_01
from dados import imgParaMatrizClass
from dados import dadosClass

#reco_type = "image"

no_of_images_of_one_person = 8
dataset_obj = dadosClass(no_of_images_of_one_person)

images_paths_for_training = dataset_obj.images_path_for_training
labels_for_training = dataset_obj.labels_for_training
no_of_elements_for_training = dataset_obj.no_of_images_for_training

images_paths_for_testing = dataset_obj.images_path_for_testing
labels_for_testing = dataset_obj.labels_for_testing
no_of_elements_for_testing = dataset_obj.no_of_images_for_testing

images_targests = dataset_obj.images_target

im_largura, im_altura = 50, 50
imageToMatrixClassObj = imgParaMatrizClass(images_paths_for_training, im_largura, im_altura)
img_matrix = imageToMatrixClassObj.get_matrix()


objeto_PCA01_class = PCA_01(img_matrix, labels_for_training, images_targests, no_of_elements_for_training, im_largura, im_altura, quality_percent=90)
new_coordinates = objeto_PCA01_class.reduce_dim()

sim = 0
nao = 0
i = 0

for img_path in images_paths_for_testing:
       img = objeto_PCA01_class.img_from_path(img_path)
       new_cords_for_image = objeto_PCA01_class.new_cords(img)

       finded_name = objeto_PCA01_class.recognize_face(new_cords_for_image)
       target_index = labels_for_testing[i]
       original_name = images_targests[target_index]

       if finded_name is original_name:
           sim += 1
           print("% Resultado %", "acuracia:", finded_name)
       else:
           nao += 1
           print("% Result %", "acuracia:", finded_name)
       i += 1
