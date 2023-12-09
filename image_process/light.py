import cv2
import numpy as np
import os

light_path = 'image/milk0/1/0.jpg'
low_path = 'image/milk0/1/90.jpg'

out_fold = 'image/milk0/1_light'
img_name = '90.jpg'

if not os.path.exists(out_fold):
    os.makedirs(out_fold)

light_img = cv2.imread(light_path, 0)
low_img = cv2.imread(low_path, 0)

light_mean = light_img.mean()
low_mean = low_img.mean()

s = light_mean / low_mean

low_img = np.clip(low_img + 100, 0, 255)

cv2.imwrite(os.path.join(out_fold, img_name), low_img)