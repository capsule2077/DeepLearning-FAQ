#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


from project import LocalDataset

print("this is main-hotfix")
print("this is main-hotfix222")

class ColorManager(object):

    def __init__(self):
        self.caches = {}
        print("ssssssssssssssssssssssssssssssssssssssssss")

    def __call__(self, c_id):
        try:
            return self.caches[c_id]
        except KeyError:
            color = np.random.randint(low=50, high=250, size=3).tolist()
            self.caches[c_id] = color
            return color


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray, nonzero=1e-10):
    area1 = np.maximum((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]), nonzero)  # [M]
    area2 = np.maximum((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]), nonzero)  # [M]

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.maximum(rb - lt, nonzero)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter

    iou = inter / union  # [N,M]
    return iou


class CvatDuplicateChecker(LocalDataset):

    def __init__(self, iou_threshold=0.5, image_dir_name='images', label_dir_name='labels', txt_suffix='.txt'):
        self.iou_threshold = iou_threshold
        self.txt_suffix = txt_suffix
        self.label_dir_path = self.dataset_path.joinpath(label_dir_name)
        self.image_dir_path = self.dataset_path.joinpath(image_dir_name)
        self.duplicate_image_path = self.output_root.joinpath(f'{self.dataset_path.name}-duplicate-iou{self.iou_threshold}')
        self.colors = ColorManager()

    def run(self):
        self.duplicate_image_path.mkdir(parents=True, exist_ok=True)
        # 遍历所有的图片
        for image_path in self.image_dir_path.rglob('*.*'):
            # 图片对应的label文件
            label_path = self.label_dir_path.joinpath(image_path.relative_to(self.image_dir_path).with_suffix(self.txt_suffix))
            # 加载标签
            with label_path.open() as f:
                boxes = [x.split() for x in f.read().strip().splitlines() if len(x)]
            # 转为numpy数据
            boxes = np.asarray(boxes, dtype=np.float32)
            # 类别
            classes = boxes[:, 0]
            # 框
            boxes = boxes[:, 1:]
            # 转换数据格式: [xc, yc, w, h] -> [x1, y1, x2, y2]
            boxes[:, 0:2] -= boxes[:, 2:4]/2
            boxes[:, 2:4] += boxes[:, 0:2]
            # 如果 x1 > x2，则交换 x1, x2
            ix = boxes[:, 0] > boxes[:, 2]
            boxes[ix, 0::2] = boxes[ix, 2::-2]
            # 如果 y1 > y2，则交换 y1, y2
            iy = boxes[:, 1] > boxes[:, 3]
            boxes[iy, 1::2] = boxes[iy, 3::-2]
            # 计算 iou
            iou = box_iou(boxes, boxes)
            # 使用上三角矩阵,去除重复的对比
            iou = np.triu(iou, 1)
            # 根据阈值判断冲突
            collisions = iou > self.iou_threshold
            # 如果有冲突, 则输出
            if np.any(collisions):
                print(image_path)
                # 加载图片
                image = cv2.imread(str(image_path))
                # 图片尺寸
                height, width, c = image.shape
                # 还原框的真实大小
                boxes *= [width, height, width, height]
                # 转换为 int 类型
                boxes = boxes.astype(int)
                # 得到冲突的具体坐标
                for index in np.argwhere(collisions):
                    class_1, class_2 = classes[index]
                    box_1, box_2 = boxes[index]
                    # 渲染框
                    cv2.rectangle(image, box_1[0:2], box_1[2:4], self.colors(int(class_1)), thickness=2)
                    cv2.rectangle(image, box_2[0:2], box_2[2:4], self.colors(int(class_2)), thickness=2)
                    # 渲染偏移
                    cv2.line(image, (box_1[0], box_1[1]), (box_2[0], box_2[1]), (0, 0, 0), thickness=2)
                    cv2.line(image, (box_1[0], box_1[3]), (box_2[0], box_2[3]), (0, 0, 0), thickness=2)
                    cv2.line(image, (box_1[2], box_1[3]), (box_2[2], box_2[3]), (0, 0, 0), thickness=2)
                    cv2.line(image, (box_1[2], box_1[1]), (box_2[2], box_2[1]), (0, 0, 0), thickness=2)
                # 写入文件
                cv2.imwrite(str(self.duplicate_image_path.joinpath(image_path.name)), image)


if __name__ == '__main__':
    CvatDuplicateChecker(iou_threshold=0.7).run()
