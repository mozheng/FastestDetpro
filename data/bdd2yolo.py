import os
import json
import cv2
from tqdm import tqdm
from collections import OrderedDict


DATASET_PATH = '/home/jovyan/fast-data/'
TRAIN_PATH = os.path.join(DATASET_PATH, 'bdd100k/images/100k')
LABELS_PATH = os.path.join(DATASET_PATH, 'bdd100k/labels/100k')
TRAIN_IMAGES_PATH = os.path.join(TRAIN_PATH, 'train')
TRAIN_LABELS_PATH = os.path.join(LABELS_PATH, 'train')
VAL_IMAGES_PATH = os.path.join(TRAIN_PATH, 'val')
VAL_LABELS_PATH = os.path.join(LABELS_PATH, 'val')

SAVE_TRAIN_LABELS_FILE = os.path.join(DATASET_PATH, 'train.txt')
SAVE_VAL_LABELS_FILE = os.path.join(DATASET_PATH, 'val.txt')


# classname = OrderedDict((('traffic light', 0), ('traffic sign', 1), ('bus', 2),
#                    ('truck', 3), ('person', 4), ('motor', 5),
#                    ('bike', 6), ('rider', 7), ('train', 8), ('car', 9)))
classname = OrderedDict((('person', 0), ('bike', 1), ('motor', 2), ('car', 3),
                         ('bus', 4), ('truck', 5), ('train', 6)))


def convert_bdd2yolo(read_images_path, read_labels_path, save_file, save_dir):
    total_name = {}
    save_file = open(save_file, 'w')
    for filename in tqdm(os.listdir(read_labels_path)):
        # read json file
        basename, ext = os.path.splitext(filename)
        if ext != '.json':
            continue
        data = json.load(open(read_labels_path + '/' + filename, 'r'))
        imagepath = os.path.join(read_images_path, basename+'.jpg')
        img = cv2.imread(imagepath)
        height, width = img.shape[:2]
        txtpath = os.path.join(save_dir, basename+'.txt')
        with open(txtpath, 'w') as labelf:
          for frame in data['frames']:
            for object in frame['objects']:
                if 'box2d' in object:
                    obj_name = object['category']
                    if obj_name == "rider":
                        obj_name = "person"
                    if obj_name not in classname.keys():
                        continue
                    x1 = float(object['box2d']['x1'])
                    y1 = float(object['box2d']['y1'])
                    x2 = float(object['box2d']['x2'])
                    y2 = float(object['box2d']['y2'])

                    x_c = (x1 + (x2 - x1)/2) / width
                    y_c = (y1 + (y2 - y1)/2) / height
                    bbox_w = (x2 - x1) / width
                    bbox_h = (y2 - y1) / height
                    class_id = classname[obj_name]
                    labelf.write('{} {} {} {} {}\n'.format(
                        class_id, x_c, y_c, bbox_w, bbox_h))

                    if obj_name in total_name.keys():
                        total_name[obj_name] += 1
                    else:
                        total_name[obj_name] = 0
        save_file.write(imagepath + '\n')
    save_file.close()
    return total_name


if __name__ == "__main__":
    SAVE_TRAIN_LABELS_DIR = TRAIN_IMAGES_PATH
    SAVE_VAL_LABELS_DIR = VAL_IMAGES_PATH
    train_total_name = convert_bdd2yolo(
        TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, SAVE_TRAIN_LABELS_FILE, SAVE_TRAIN_LABELS_DIR)
    val_total_name = convert_bdd2yolo(
        VAL_IMAGES_PATH, VAL_LABELS_PATH, SAVE_VAL_LABELS_FILE, SAVE_VAL_LABELS_DIR)

    print('========================================================')
    print("Train set total classes")
    for key in classname.keys():
        print("{:13}\t{}".format(key, train_total_name[key]))

    print('========================================================')
    print("validation set total classes")
    for key in classname.keys():
        print("{:13}\t{}".format(key, val_total_name[key]))
