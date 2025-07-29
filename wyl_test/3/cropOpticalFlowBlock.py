import cv2

from getFaceCoordinatesDict import whole_face_block_coordinates
import os

def crop_optical_flow_block():
    # 获取存放各人脸关键点坐标的字典
    face_block_coordinates_dict = whole_face_block_coordinates()

    ''' 拿到各样本对应的光流图，并在其上按照刚才的坐标点进行分块 '''
    whole_optical_flow_path = '../../datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}  # 存放每个图片的子图的字典

    for imgName in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[imgName] = []  # 字典的value，初始为空
        img = cv2.imread(whole_optical_flow_path + '/' + imgName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_coordinates = face_block_coordinates_dict[imgName]

        l_eye = img[face_coordinates[0][0] - 7: face_coordinates[0][0] + 7,
                face_coordinates[0][1] - 7: face_coordinates[0][1] + 7]
        r_eye = img[face_coordinates[1][0] - 7: face_coordinates[1][0] + 7,
                face_coordinates[1][1] - 7: face_coordinates[1][1] + 7]
        l_lips = img[face_coordinates[3][0] - 7: face_coordinates[3][0] + 7,
                 face_coordinates[3][1] - 7: face_coordinates[3][1] + 7]
        r_lips = img[face_coordinates[4][0] - 7: face_coordinates[4][0] + 7,
                 face_coordinates[4][1] - 7: face_coordinates[4][1] + 7]

        four_parts_optical_flow_imgs[imgName].append(l_eye)
        four_parts_optical_flow_imgs[imgName].append(r_eye)
        four_parts_optical_flow_imgs[imgName].append(l_lips)
        four_parts_optical_flow_imgs[imgName].append(r_lips)

    print(len(four_parts_optical_flow_imgs))  # 442
    return four_parts_optical_flow_imgs
