#coding:utf8
import sys
if '/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/' not in sys.path:
    sys.path.append('/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/')
if '/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/yolov2/' not in sys.path:
    sys.path.append('/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/yolov2/')
# print (sys.path)

from os.path import join
from yolov2 import get_bboxes,get_darknet
from tqdm import tqdm

TEST_IMG_PATH = '/home/snk/WindowsDisk/Download/KITTI/data_object_image_2/testing/image_2/'
SAVE_PATH = '/home/snk/WindowsDisk/Download/KITTI/test_labels/'

TEST_NUM = 7518

get_darknet('/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/yolov2/cfg/yolo-kitti.cfg', '/home/snk/Desktop/CNN-based-Image-Compression-Guided-by-YOLOv2/yolov2/kitti_backup/000084.weights')
list_file = open('/home/snk/WindowsDisk/Download/KITTI/test.txt', 'w')

for idx in tqdm(range(0,TEST_NUM)):
    out_file = open(join(SAVE_PATH, '%06d.txt' % idx), 'w')
    img_path = join(TEST_IMG_PATH, '%06d.png' % idx)
    # img_path = "/home/snk/WindowsDisk/Download/KITTI/data_object_image_2/training/image_2/000000.png"
    list_file.write(img_path + '\n')
    # print ('%06d.txt' % idx)
    # img = Image.open(join(IMAGE_PATH, '%06d.png' % idx))
    # img.show()
    # w, h = img.size
    # pdb.set_trace()
    # label = open(join(LABEL_PATH, '%06d.txt' % idx))
    label = get_bboxes(img_path)
    for line in label:
        cls_id = line[6]
        # if cls == 'DontCare':
        #     cls = 'Misc'
        # if cls not in CLASS:
        #     print ('Invalid ClassName %s!' % cls)
        # cls_id = CLASS.index(cls)
        # bb = convert((w,h), tuple(map(float, line[4:8])))
        bb = line[:4]
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()
list_file.close()


# print (get_bboxes('yolov2/cfg/yolo-kitti.cfg', 'yolov2/kitti_backup/000084.weights', '/home/snk/WindowsDisk/Download/KITTI/data_object_image_2/testing/image_2/000000.png'))