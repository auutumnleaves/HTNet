import os
import time
from os import path

import cv2
import numpy as np
import pandas
import torch
from facenet_pytorch.models.mtcnn import MTCNN
import torch.nn as nn


def whole_face_block_coordinates():
    df = pandas.read_csv('../../combined_3_class2_for_optical_flow.csv')
    m, n = df.shape
    base_data_src = '../../datasets/combined_datasets_whole'

    image_size_u_v = 28

    face_block_coordinates = {}

    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(df['filename_o'][i]) + ' .png'
        img_path_apex = base_data_src + '/' + df['imagename'][i]
        train_face_image_apex = cv2.imread(img_path_apex)
        face_apex = cv2.resize(train_face_image_apex, (28, 28), interpolation=cv2.INTER_AREA)
        face_apex_rgb = cv2.cvtColor(face_apex, cv2.COLOR_BGR2RGB)

        # 获取面部关键点
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex_rgb, landmarks=True)

        # 如果未检测到面部关键点
        if batch_landmarks is None:
            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])
        row_n, col_n = np.shape(batch_landmarks[0])
        for j in range(0, row_n):
            for k in range(0, col_n):
                if batch_landmarks[0][j][k] < 7:
                    batch_landmarks[0][j][k] = 7
                if batch_landmarks[0][j][k] > 21:
                    batch_landmarks[0][j][k] = 21
        batch_landmarks = batch_landmarks.astype(int)
        face_block_coordinates[image_name] = batch_landmarks[0]

    return face_block_coordinates


# 把28*28的变为四个14*14的
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()

    whole_optical_flow_path = '../../datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)

    # print(whole_optical_flow_imgs)
    four_parts_optical_flow_imgs = {}

    for n_img in whole_optical_flow_imgs:
        # print(n_img, type(n_img))
        four_parts_optical_flow_imgs[n_img] = []
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        # print(four_part_coordinates)
        l_eye = flow_image[four_part_coordinates[0][0] - 7: four_part_coordinates[0][0] + 7,
                four_part_coordinates[0][1] - 7: four_part_coordinates[0][1] + 7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7: four_part_coordinates[1][0] + 7,
                 four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7: four_part_coordinates[2][0] + 7,
               four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7: four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7: four_part_coordinates[4][0] + 7,
                 four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 3)
        self.bn1 = nn.BatchNorm1d(3)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(6, 2)
        self.relu = nn.ReLU()

    def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
        fuse_five_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
        fuse_out = self.fc1(fuse_five_features)
        # fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        fuse_whole_five_parts = torch.cat((whole_feature, fuse_out))
        # fuse_whole_five_parts = self.bn1(fuse_whole_five_parts)
        fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
        fuse_whole_five_parts = self.d1(fuse_whole_five_parts)
        out = self.fc_2(fuse_whole_five_parts)
        return out


# def main(config):
def main():
    learning_rate = 0.00005
    batch_size = 256
    epochs = 800
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss()
    # if (config.train):
    #     if not path.exists('ourmodel_threedatasets_weights'):
    #         os.mkdir('ourmodel_threedatasets_weights')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()

    main_path = '../../datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    print(subName)

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []

        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))

        print(y_train)
        print(len(y_train))

if __name__ == '__main__':
    main()
