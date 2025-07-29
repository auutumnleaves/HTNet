import argparse
import os
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from os import path
import cv2
from torch import dtype
from torch.utils.data import TensorDataset, DataLoader

from cropOpticalFlowBlock import crop_optical_flow_block
from recognitionEvaluation import recognition_evaluation
from Model import HTNet


def main(config):
    learning_rate = 0.00005  # 定义学习率、batch_size、epochs
    batch_size = 256
    epochs = 800

    all_accuracy_dict = {}  # 对于每名受试者，在测试集上的最佳预测结果和测试集本身实际标签，存放在该字典里

    is_cuda = torch.cuda.is_available()  # 使用GPU
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

    if config.train:  # 若以训练模式执行，则需要新建一个存放训练参数权重文件的目录
        if not path.exists('model_weights'):
            os.mkdir('model_weights')

    print("lr=%f, epochs=%d, device=%s\n" % (learning_rate, epochs, device))

    total_gt = []  # 真实标签
    total_pred = []  # 最后一轮epoch的预测结果
    best_total_pred = []  # 所有轮epochs中最佳的那次预测结果

    t = time.time()  # 记一个训练开始前的时间点

    main_path = '../../datasets/three_norm_u_v_os'
    subNames = os.listdir(main_path)
    four_parts_optical_flow = crop_optical_flow_block()
    # print(subNames)

    ''' 开始对每个受试者目录进行处理 '''
    for subName in subNames:
        print('Subject:', subName)
        # 训练图片和标签
        img_train = []
        label_train = []
        # 测试图片和标签
        img_test = []
        label_test = []

        # 装填训练数据及标签
        train_dir = main_path + '/' + subName + '/u_train'  # 训练集路径
        labels = os.listdir(train_dir)  # 类别文件夹列表
        for label in labels:
            imgs = os.listdir(train_dir + '/' + label)  # 当前类别下的所有图片列表
            for img in imgs:
                label_train.append(int(label))
                l_eye_lips = cv2.hconcat([four_parts_optical_flow[img][0], four_parts_optical_flow[img][2]])
                r_eye_lips = cv2.hconcat([four_parts_optical_flow[img][1], four_parts_optical_flow[img][3]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                img_train.append(lr_eye_lips)

        # 装填测试数据及标签
        test_dir = main_path + '/' + subName + '/u_test'  # 测试集路径
        labels = os.listdir(test_dir)  # 类别文件夹列表
        for label in labels:
            imgs = os.listdir(test_dir + '/' + label)  # 当前类别下的所有图片列表
            for img in imgs:
                label_test.append(int(label))
                l_eye_lips = cv2.hconcat([four_parts_optical_flow[img][0], four_parts_optical_flow[img][2]])
                r_eye_lips = cv2.hconcat([four_parts_optical_flow[img][1], four_parts_optical_flow[img][3]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                img_test.append(lr_eye_lips)

        weight_path = 'model_weights' + '/' + subName + '.pth'

        # 下面开始训练
        model = HTNet(
            image_size=28,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=3
        )

        model = model.to(device)

        # 训练模式还是测试模式
        if config.train:
            print('train')
        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器

        # 训练集封装
        label_train = torch.Tensor(label_train).to(dtype=torch.long)  # 标签转Tensor，要求long类型
        img_train = torch.Tensor(np.array(img_train)).permute(0, 3, 1, 2)  # 图片转Tensor，并重排维度
        dataset_train = TensorDataset(img_train, label_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

        # 测试集封装
        label_test = torch.Tensor(label_test).to(dtype=torch.long)
        img_test = torch.Tensor(np.array(img_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(img_test, label_test)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

        # 存放最佳结果
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            # 训练
            if config.train:
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in dataloader_train:
                    optimizer.zero_grad()
                    x = batch[0].to(device)  # x的形状是[batch, 3, 28, 28]
                    y = batch[1].to(device)  # y的形状是[batch]
                    y_pred = model(x)  # y_pred的形状是[batch, 3]
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * x.size(0)  # x.size(0)=batch
                    num_train_correct += (torch.max(y_pred, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(dataloader_train.dataset)

            # 测试，不需要判断是否为train
            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in dataloader_test:
                x = batch[0].to(device)
                y = batch[1].to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss += loss.data.item() * x.size(0)
                num_val_correct += (torch.max(y_pred, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(dataloader_test.dataset)

            # 检查是否需要更新最佳预测结果，若为最佳结果，则还需更新模型参数文件
            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(y_pred, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred
                # 更新权重文件
                if config.train:
                    torch.save(model.state_dict(), weight_path)

        # 为计算UF1和UAR做准备工作
        print("Best Predicted	:", best_each_subject_pred)  # 打印一下在这名受试者的测试集上的最佳预测结果
        accuracy_dict = {}
        accuracy_dict['pred'] = best_each_subject_pred  # 在测试集上的最佳预测结果
        accuracy_dict['truth'] = y.tolist()  # 测试集真实标签列表
        all_accuracy_dict[subName] = accuracy_dict

        print("Ground Truth	:", y.tolist())  # 打印一下这名受试者的测试集本身的真实标签
        print("Evaluation until this subject:")
        total_pred.extend(torch.max(y_pred, 1)[1].tolist())  # 最后一轮epoch的预测结果
        total_gt.extend(y.tolist())  # 真实标签
        best_total_pred.extend(best_each_subject_pred)  # 最佳的那次epoch的预测结果

        # 开始计算UF1和UAR
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)  # 真实标签与最后一轮之间的
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)  # 真实标签与最佳那轮之间的
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    # 计算最终UF1和UAR
    print("Final Evaluation: ")
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred)
    print('total best UF1:', round(best_UF1, 4), '| total best UAR:', round(best_UAR, 4))
    print(np.shape(total_gt))
    print("Total Time Taken: ", time.time() - t)
    print(all_accuracy_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=strtobool, default=False)
    config = parser.parse_args()
    config.train = True
    main(config)
