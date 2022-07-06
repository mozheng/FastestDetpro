import os
import cv2
import numpy as np

import torch
import random

def random_integer(lens):
    return random.randint(1, lens)

def merge_bboxes(bboxes, cutx, cuty):
    # 保存修改后的检测框
    merge_box = []

    # 遍历每张图像，共4个
    for i, boxes in enumerate(bboxes):

        # 每张图片中需要删掉的检测框
        index_list = []

        # 遍历每张图的所有检测框,index代表第几个框
        for index, box in enumerate(boxes[0]):

            # axis=1纵向删除index索引指定的列，axis=0横向删除index指定的行
            # box[0] = np.delete(box[0], index, axis=0)

            # 获取每个检测框的宽高
            x1, y1, x2, y2 = box[:4]

            # 如果是左上图，修正右侧和下侧框线
            if i == 0:
                # 如果检测框左上坐标点不在第一部分中，就忽略它
                if x1 > cutx or y1 > cuty:
                    index_list.append(index)

                # 如果检测框右下坐标点不在第一部分中，右下坐标变成边缘点
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        index_list.append(index)

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    # 如果修正后的左上坐标和右下坐标之间的距离过小，就忽略这个框
                    if x2-x1 < 5:
                        index_list.append(index)

            # 如果是右上图，修正左侧和下册框线
            if i == 1:
                if x2 < cutx or y1 > cuty:
                    index_list.append(index)

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        index_list.append(index)

                if x1 <= cutx and x2 >= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        index_list.append(index)

            # 如果是左下图
            if i == 2:
                if x1 > cutx or y2 < cuty:
                    index_list.append(index)

                if y1 <= cuty and y2 >= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        index_list.append(index)

                if x1 <= cutx and x2 >= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        index_list.append(index)

            # 如果是右下图
            if i == 3:
                if x2 < cutx or y2 < cuty:
                    index_list.append(index)

                if x1 <= cutx and x2 >= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        index_list.append(index)

                if y1 <= cuty and y2 >= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        index_list.append(index)

            # 更新坐标信息
            bboxes[i][0][index][:4] = np.array(
                [x1, y1, x2, y2])  # 更新第i张图的第index个检测框的坐标

        # 删除不满足要求的框，并保存
        merge_box.append(np.delete(bboxes[i][0], index_list, axis=0))

    # 返回坐标信息
    return merge_box


# 对传入的四张图片数据增强
def get_random_data(image_list, input_shape, min_offset=(0.3 ,0.3)):
    w, h = input_shape  # 获取图像的宽高
    '''设置拼接的分隔线位置'''
    min_offset_x = min_offset[0]
    min_offset_y = min_offset[1]
    scale_low = 1 - min(min_offset_x, min_offset_y)  # 0.6
    scale_high = scale_low + 0.2  # 0.8

    image_datas = []  # 存放图像信息
    box_datas = []  # 存放检测框信息

    #（1）图像分割
    for index, frame_list in enumerate(image_list):

        frame = frame_list[0]  # 取出的某一张图像
        box = np.array(frame_list[1:])  # 该图像对应的检测框坐标

        ih, iw = frame.shape[0:2]  # 图片的宽高

        cx = (box[0, :, 0] + box[0, :, 2]) // 2  # 检测框中心点的x坐标
        cy = (box[0, :, 1] + box[0, :, 3]) // 2  # 检测框中心点的y坐标

        # 对输入图像缩放
        new_ar = w/h  # 图像的宽高比
        scale = np.random.uniform(scale_low, scale_high)   # 缩放0.6--0.8倍
        # 调整后的宽高
        nh = int(scale * h)  # 缩放比例乘以要求的宽高
        nw = int(nh * new_ar)  # 保持原始宽高比例

        # 缩放图像
        frame = cv2.resize(frame, (nw, nh))

        # 调整中心点坐标
        cx = cx * nw/iw
        cy = cy * nh/ih

        # 调整检测框的宽高
        bw = (box[0, :, 2] - box[0, :, 0]) * nw/iw  # 修改后的检测框的宽高
        bh = (box[0, :, 3] - box[0, :, 1]) * nh/ih

        # 创建一块[416,416]的底版
        new_frame = np.zeros((h, w, 3), np.uint8)

        # 确定每张图的位置
        if index == 0:
            new_frame[0:nh, 0:nw] = frame   # 第一张位于左上方
        elif index == 1:
            new_frame[0:nh, w-nw:w] = frame  # 第二张位于右上方
        elif index == 2:
            new_frame[h-nh:h, 0:nw] = frame  # 第三张位于左下方
        elif index == 3:
            new_frame[h-nh:h, w-nw:w] = frame  # 第四张位于右下方

        # 修正每个检测框的位置
        if index == 0:  # 左上图像
            box[0, :, 0] = cx - bw // 2  # x1
            box[0, :, 1] = cy - bh // 2  # y1
            box[0, :, 2] = cx + bw // 2  # x2
            box[0, :, 3] = cy + bh // 2  # y2

        if index == 1:  # 右上图像
            box[0, :, 0] = cx - bw // 2 + w - nw  # x1
            box[0, :, 1] = cy - bh // 2  # y1
            box[0, :, 2] = cx + bw // 2 + w - nw  # x2
            box[0, :, 3] = cy + bh // 2  # y2

        if index == 2:  # 左下图像
            box[0, :, 0] = cx - bw // 2  # x1
            box[0, :, 1] = cy - bh // 2 + h - nh  # y1
            box[0, :, 2] = cx + bw // 2  # x2
            box[0, :, 3] = cy + bh // 2 + h - nh  # y2

        if index == 3:  # 右下图像
            box[0, :, 0] = cx - bw // 2 + w - nw  # x1
            box[0, :, 1] = cy - bh // 2 + h - nh  # y1
            box[0, :, 2] = cx + bw // 2 + w - nw  # x2
            box[0, :, 3] = cy + bh // 2 + h - nh  # y2

        # 保存处理后的图像及对应的检测框坐标
        image_datas.append(new_frame)
        box_datas.append(box)

    #（2）将四张图像拼接在一起
    # 在指定范围中选择横纵向分割线
    cutx = np.random.randint(int(w*min_offset_x), int(w*(1-min_offset_x)))
    cuty = np.random.randint(int(h*min_offset_y), int(h*(1-min_offset_y)))

    # 创建一块[416,416]的底版用来组合四张图
    new_image = np.zeros((h, w, 3), np.uint8)
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[:cuty, cutx:, :] = image_datas[1][:cuty, cutx:, :]
    new_image[cuty:, :cutx, :] = image_datas[2][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[3][cuty:, cutx:, :]

    # 处理超出图像边缘的检测框
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    # # 复制一份合并后的图像
    # modify_image_copy = new_image.copy()

    # # 绘制修正后的检测框
    # for boxes in new_boxes:
    #     # 遍历每张图像中的所有检测框
    #     for box in boxes:
    #         # 获取某一个框的坐标
    #         x1, y1, x2, y2 = box[:4]
    #         cv2.rectangle(modify_image_copy, (int(x1), int(y1)),
    #                       (int(x2), int(y2)), (0, 255, 0), 2)
    #         cv2.putText(modify_image_copy, str(box[4]), (int(x1), int(
    #             y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    # cv2.imshow('new_img_bbox', modify_image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return new_image, new_boxes


def random_scale(image, boxes):
    height, width, _ = image.shape
    # random crop imgage
    cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape
    
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx - cx)/roi_w, (by - cy)/roi_h
        bw, bh = bw/roi_w, bh/roi_h

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)

    return roi, output

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class TensorDataset():
    def __init__(self, path, img_width, img_height, aug=False):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.aug = aug
        self.random_mosac = True
        self.path = path
        self.data_list = []
        self.img_width = img_width
        self.img_height = img_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']

        # 数据检查
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)
    def getimageinfo(self, index):
        img_path = self.data_list[index]
        label_path = img_path.split(".")[0] + ".txt"

        # 加载图片
        img = cv2.imread(img_path)
        # 加载label文件
        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([0, l[0], l[1], l[2], l[3], l[4]])
            label = np.array(label, dtype=np.float32)

            if label.shape[0]:
                assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                #assert (label >= 0).all(), 'negative labels: %s'%label_path
                #assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path) 
        return img, label

    def getoneimage(self, index):
        img, label = self.getimageinfo(index)
        # 是否进行数据增强
        if self.aug:
            img, label = random_scale(img, label)
        return img, label

    def getmosacimage(self, index):
        img, label = self.getimageinfo(index)
        image_list = []
        for i in range(4):
            img, labels = self.getoneimage(random_integer(self.len(self.data_list)))
            h, w = img.shape[:2]
            xyxylabels = []
            for label in labels:
                _, l, b1, b2, b3, b4 = label
                xyxylabels.append(
                    [(b1 - b3/2) * w, (b2 - b4/2) * h, (b1 + b3/2) * w, (b2 + b4/2) * h, l])
            image_list.append([img, xyxylabels])
        image, bboxes = get_random_data(image_list, input_shape=(self.img_size_width, self.img_size_height))
        labels = []
        for bbox in (bboxes):
            for box in bbox:
                x = label[0] / self.img_size_width
                y = label[1] / self.img_size_height
                w = (label[2] - label[0]) / self.img_size_width
                h = (label[3] - label[1]) / self.img_size_height
                labels.append([0, label[4], x+w/2, y+h/2, w, h])
        return image, labels

    def __getitem__(self, index):
        if random_integer(2) % 2 == 0 and self.random_mosac:  # 随机采取mosac方法。
            img, label = self.getmosacimage(index)
        else:
            img, label = self.getoneimage(index)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation = cv2.INTER_LINEAR) # 尺寸变换
        img = img.transpose(2,0,1)
        
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    data = TensorDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)