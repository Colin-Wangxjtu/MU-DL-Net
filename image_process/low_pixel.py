import cv2
import os

curDir = os.curdir  # 获取当前执行python文件的文件夹
sourceDir = os.path.join(curDir, 'image/new_img/milk1')
resultDir = os.path.join(curDir, 'image/new_img/milk1_low')

def resolution_lower_handler(sourceDir, resultDir):
    img_list = os.listdir(sourceDir)

    for i in img_list:
        pic = cv2.imread(os.path.join(sourceDir, i), cv2.IMREAD_COLOR)
        pic_n = cv2.resize(pic, (1224, 1024))
        pic_name = i
        cv2.imwrite(os.path.join(resultDir, i), pic_n)

if __name__ == '__main__':
    resolution_lower_handler(sourceDir, resultDir)