import pandas as pd
import cv2 as cv
from facenet_pytorch import MTCNN
import numpy as np

df = pd.read_csv('../../combined_3_class2_for_optical_flow.csv')
m, n = df.shape

base_data_src = '../../datasets/combined_datasets_whole'
image_size_u_v = 28

MARKER_COLOR = (0, 0, 255)
MARKER_SIZE = 2
WINDOW_NAME = "Face KeyPoints"
FONT = cv.FONT_HERSHEY_SIMPLEX

cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

for i in range(0, m):
    image_name = str(df['sub'][i]) + '_' + str(df['filename_o'][i]) + '.png'
    # print(image_name)
    img_path_apex = base_data_src + '/' + df['imagename'][i]
    train_face_image_apex = cv.imread(img_path_apex)
    train_face_image_apex_rgb = cv.cvtColor(train_face_image_apex, cv.COLOR_BGR2RGB)
    face_apex = cv.resize(train_face_image_apex, (28, 28), interpolation=cv.INTER_AREA)
    face_apex_rgb = cv.cvtColor(face_apex, cv.COLOR_BGR2RGB)

    # 获取面部及面部框体
    mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
    batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex_rgb, landmarks=True)

    if batch_landmarks is None:
        print(f"{df['imagename'][i]}未检测到人脸")
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

    print(batch_landmarks[0])



    # 绘制关键点
    for idx, (x, y) in enumerate(batch_landmarks[0]):
        # 绘制实心圆
        cv.circle(face_apex,
                  (round(x), round(y)),
                  radius=MARKER_SIZE,
                  color=MARKER_COLOR,
                  thickness=-1)

    # 放大显示（28*28太小，放大8倍）
    # zoom_factor = 1
    # large_vis = cv.resize(face_apex, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv.INTER_NEAREST)

    # 显示图像
    cv.imshow(WINDOW_NAME, face_apex)

    # 交互控制
    while True:
        key = cv.waitKey(0)
        if key == 13:  # Enter键继续
            break
        elif key == 27:  # ESC键退出
            cv.destroyAllWindows()
            print("用户终止操作")
            exit()

# 结束清理
cv.destroyAllWindows()
