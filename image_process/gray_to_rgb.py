import os
import cv2

# 输入和输出文件夹路径
input_folder = "image/new_img/no"  # 替换为你的输入文件夹路径
output_folder = "image/new_img/no_RGB"  # 替换为你的输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有文件
input_files = os.listdir(input_folder)

for input_file in input_files:
    input_path = os.path.join(input_folder, input_file)
    output_path = os.path.join(output_folder, input_file)

    # 读取灰度图像
    grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 转换为RGB图像
    rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # 保存RGB图像到输出文件夹
    cv2.imwrite(output_path, rgb_image)

print("转换完成。RGB图像保存在", output_folder)
