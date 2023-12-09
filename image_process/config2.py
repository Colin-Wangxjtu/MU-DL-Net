import cv2
import numpy as np
import os
from numpy.linalg import norm

def get_color_points(color_img):
    # 定义颜色阈值范围（以BGR格式）
        lower_red = np.array([0, 0, 100])
        upper_red = np.array([70, 70, 255])

        lower_green = np.array([0, 100, 0])
        upper_green = np.array([90, 255, 70])

        lower_blue = np.array([100, 70, 0])
        upper_blue = np.array([255, 155, 40])
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([60, 255, 255])
        
        lower_purple = np.array([120, 30, 50])
        upper_purple = np.array([150, 60, 120])

        # 使用颜色阈值分割来提取红、绿和蓝的像素
        
        red_mask = cv2.inRange(color_img, lower_red, upper_red)
        green_mask = cv2.inRange(color_img, lower_green, upper_green)
        blue_mask = cv2.inRange(color_img, lower_blue, upper_blue)
        yellow_mask = cv2.inRange(color_img, lower_yellow, upper_yellow)
        purple_mask = cv2.inRange(color_img, lower_purple, upper_purple)

        # color_img[red_mask] = np.array([0, 0, 255])
        # color_img[green_mask] = np.array([0, 255, 0])
        # color_img[blue_mask] = np.array([255, 0, 0])
        # color_img[red_mask] = np.array([0, 255, 255])
        # color_img[red_mask] = np.array([128, 0, 128])
        # cv2.imwrite(os.path.join(input_folder, '05.jpg'), color_img)
        
        # 找到标签点的位置
        red_points = cv2.findNonZero(red_mask)
        green_points = cv2.findNonZero(green_mask)
        blue_points = cv2.findNonZero(blue_mask)
        yellow_points = cv2.findNonZero(yellow_mask)
        purple_points = cv2.findNonZero(purple_mask)
        
        red_points = red_points.reshape(len(red_points), 2)
        green_points = green_points.reshape(len(green_points), 2)
        blue_points = blue_points.reshape(len(blue_points), 2)
        yellow_points = yellow_points.reshape(len(yellow_points), 2)
        purple_points = purple_points.reshape(len(purple_points), 2)

        red_points = np.mean(red_points, axis=0)
        green_points = np.mean(green_points, axis=0)
        blue_points = np.mean(blue_points, axis=0)
        yellow_points = np.mean(yellow_points, axis=0)
        purple_points = np.mean(purple_points, axis=0)
        
        return red_points, green_points, blue_points, yellow_points, purple_points

if __name__ == '__main__':
    
    before_cfg_img = cv2.imread()
    cfg_img = cv2.imread()
    to_cfg_img = cv2.imread()

    std_red, std_green, std_blue, std_yellow, std_purple = get_color_points(cfg_img)
    red_points, green_points, blue_points, _, _ = get_color_points(before_cfg_img)
    
    src_loc = np.array([red_points, green_points, blue_points]).astype(np.float32)
    dst_loc = np.array([std_red, std_green, std_blue], dtype=np.float32)
    transform = cv2.getAffineTransform(src_loc, dst_loc)
    output_image = cv2.warpAffine(to_cfg_img, transform, cfg_img.shape[:1])
    _, _, _, yellow_points, purple_points = get_color_points(output_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('', output_image)
    dst = np.mean((norm(std_yellow-yellow_points, 2), norm(std_purple-purple_points), 2))