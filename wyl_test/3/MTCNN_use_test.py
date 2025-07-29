import cv2
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image

img = cv2.imread("../../datasets/combined_datasets_whole/s20_s20_ne_10.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
img_rgb = img  # 不转RGB，直接使用BGR

img_rgb_resize = cv2.resize(img_rgb, (28, 28), interpolation=cv2.INTER_AREA)

detector = MTCNN(select_largest=True, post_process=False, device='cuda:0')
boxes, probs, landmarks = detector.detect(img_rgb_resize, landmarks=True)

print(boxes)
print("-----")
print(probs)
print("-----")
print(landmarks)

draw_img = img_rgb_resize.copy()  # 创建图像的副本用于绘制

if boxes is not None:
    # 由于boxes是人脸框列表，有可能不止一个人脸，因此需要遍历处理
    for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
        # 1.绘制人脸边界框
        box = box.astype(int)  # 向下取整
        cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)  # 绿色框，线宽2

        # 2.绘制置信度文本
        # text = f"Conf: {prob:.4f}"
        # cv2.putText(draw_img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        # 3.绘制五个关键点
        # 关键点名称和颜色
        landmark_types = ["Left Eye", "Right Eye", "Nose", "Mouth Left", "Mouth right"]
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        # 绘制每个关键点
        for j, point in enumerate(landmark):
            pos = tuple(point.astype(int))  # 由于mtcnn.detect()返回的landmark坐标是列表，转为元组，并且向下取整
            cv2.circle(draw_img, pos, 5, colors[j], -1)  # 彩色实心圆
            cv2.putText(draw_img, f"{j + 1}", (pos[0] + 7, pos[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 1)

# 显示结果
cv2.imshow("Face Detection Result", draw_img)
print(draw_img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
