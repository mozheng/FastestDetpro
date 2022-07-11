import cv2

classname = ['person', 'bike',  'car', 'bus', 'truck']

if __name__ == "__main__":
    imgpath = "fast-data/bdd100k/images/100k/train/00abf44e-421f6ed7.jpg"
    labelpath = "fast-data/bdd100k/images/100k/train/00abf44e-421f6ed7.txt"
    img = cv2.imread(imgpath)
    h, w = img.shape[:2]
    with open(labelpath, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            class_id = int(line[0])
            x_c = float(line[1]) * w
            y_c = float(line[2]) * h
            bbox_w = float(line[3]) * w
            bbox_h = float(line[4]) * h
            x1 = int(x_c - bbox_w/2)
            y1 = int(y_c - bbox_h/2)
            x2 = int(x_c + bbox_w/2)
            y2 = int(y_c + bbox_h/2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, classname[class_id], (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img = cv2.resize(img, (640, 384))
    cv2.imwrite("a.jpg", img)
