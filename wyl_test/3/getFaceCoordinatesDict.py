""" 按照combined_3_class2_for_optical_flow.csv中的信息
    对combined_datasets_whole目录下的所有样本图片进行面部关键点坐标的识别
    并将识别结果（该图片的5个面部关键坐标）以图片名为key，存储至一个字典当中 """
import cv2
import pandas
import numpy as np
from facenet_pytorch import MTCNN


def whole_face_block_coordinates():
    df = pandas.read_csv('../../combined_3_class2_for_optical_flow.csv')  # csv中的每一行对应一个样本
    m, n = df.shape  # 获取行数、列数，以便遍历每个样本，及样本的每个属性
    base_data_src = '../../datasets/combined_datasets_whole'  # 样本图片存放的目录，便于访问
    image_size_u_v = 28  # 要resize的尺寸
    face_block_coordinates = {}  # 对每个图片进行处理，获得5个坐标点，按图片名存放在此字典中

    for i in range(0, m):  # 遍历每个样本
        img_path = base_data_src + '/' + df['imagename'][i]  # 获得当前样本文件路径
        img = cv2.imread(img_path)  # 按路径读取图片
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_resize = cv2.resize(img_rgb, (image_size_u_v, image_size_u_v), interpolation=cv2.INTER_AREA)
        ''' 在这个地方可以试一下RGB格式和BGR格式的识别结果有什么区别，可以发现有很明显的区别，BGR有很多识别为None，而RGB明显较好 '''
        mtcnn = MTCNN()
        boxes, _, landmarks = mtcnn.detect(img_rgb_resize, landmarks=True)

        ''' 有些未识别出来的，不能让它空着，需要手动赋值 '''
        if landmarks is None:
            landmarks = np.array([[[9.528073, 11.062551]
                                      , [21.396168, 10.919773]
                                      , [15.380184, 17.380562]
                                      , [10.255435, 22.121233]
                                      , [20.583706, 22.25584]]])

        ''' 有些识别位置太歪的，需要纠正到规定位置，否则后续的7×7子区域会划到图像外边 '''
        for row in range(5):  # 遍历五个坐标
            for col in range(2):  # 遍历横、纵坐标
                if landmarks[0][row][col] < 7:
                    landmarks[0][row][col] = 7
                if landmarks[0][row][col] > 21:
                    landmarks[0][row][col] = 21

        ''' 给None赋值，并将歪的摆正，现在需要将坐标取整（因为图片处理时无法处理小于1的像素） '''
        landmarks = landmarks.astype(int)  # 向下取整

        ''' 5个关键点坐标已处理好，现在再把图片名称拿来，组合成字典元素，存入字典即可 '''
        imageName = df['sub'][i] + '_' + df['filename_o'][i] + ' .png'  # 这里.png前面有一个空格，是因为光流图文件名的.png前面都有一个空格，可能是懒得改了
        face_block_coordinates[imageName] = landmarks[0]

    return face_block_coordinates
