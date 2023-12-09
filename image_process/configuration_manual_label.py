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
    
    input_folder = 'image/new_img/no_RGB'
    output_folder = 'image/new_img/no_cfg'
    
    letter_list = ['2', '5', '8', 'w', 'q']
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    dst_size = 1216
    std_img_dict = {}
    for letter in letter_list:
        std_image = cv2.imread(os.path.join(input_folder, (letter+'_std.jpg')))
        std_shape = np.array((std_image.shape[0], std_image.shape[1]))
        std_red, std_green, std_blue, std_yellow, std_purple = get_color_points(std_image)
        std_red = std_red / std_shape * dst_size
        std_blue = std_blue / std_shape * dst_size
        std_green = std_green / std_shape * dst_size
        std_yellow = std_yellow / std_shape * dst_size
        std_purple = std_purple / std_shape * dst_size
        std_img_dict[letter] = (std_red, std_green, std_blue, std_yellow, std_purple)
    
    for img in image_files:
        for letter in letter_list:
            if img == letter + '_std.jpg' or img[0] != letter:
                continue
            # 读取彩色图像
            std_red, std_green, std_blue, std_yellow, std_purple = std_img_dict[letter]
            color_img = cv2.imread(os.path.join(input_folder, img))
            not_gray = color_img[:, :, 0] != color_img[:, :, 1]

            red_points, green_points, blue_points, _, _ = get_color_points(color_img)
            
            src_loc = np.array([red_points, green_points, blue_points]).astype(np.float32)
            dst_loc = np.array([std_red, std_green, std_blue], dtype=np.float32)
            transform = cv2.getAffineTransform(src_loc, dst_loc)
            output_image = cv2.warpAffine(color_img, transform, (dst_size, dst_size))
            _, _, _, yellow_points, purple_points = get_color_points(output_image)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output_folder, img), output_image)
            dst = np.mean((norm(std_yellow-yellow_points, 2), norm(std_purple-purple_points), 2))
            print(f'image {img} has {dst:.4f} pixel with standard image')