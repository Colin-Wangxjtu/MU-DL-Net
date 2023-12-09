import os
import cv2

# 定义输入文件夹和输出文件夹的路径
input_folder = 'image/milk0/1_cfg'
output_folder = 'image/milk0/1_processed'

# 创建输出文件夹（如果它不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 遍历每张图片并进行处理
for image_file in image_files:
    # 读取原始图像
    input_path = os.path.join(input_folder, image_file)
    original_image = cv2.imread(input_path)
    
    # 获取图像的高度和宽度
    height, width, _ = original_image.shape
    
    # 将图像分为四等分
    top_left = original_image[0:height//2, 0:width//2]
    top_right = original_image[0:height//2, width//2:]
    bottom_left = original_image[height//2:, 0:width//2]
    bottom_right = original_image[height//2:, width//2:]
    
    # 左右翻转每个四等分图像
    top_left_flip = cv2.flip(top_left, 1)  # 左右翻转
    top_right_flip = cv2.flip(top_right, 1)
    bottom_left_flip = cv2.flip(bottom_left, 1)
    bottom_right_flip = cv2.flip(bottom_right, 1)
    
    # 旋转图像
    top_left_rotate = cv2.rotate(top_left, cv2.ROTATE_180)  # 旋转180度
    top_right_rotate = cv2.rotate(top_right, cv2.ROTATE_180)
    bottom_left_rotate = cv2.rotate(bottom_left, cv2.ROTATE_180)
    bottom_right_rotate = cv2.rotate(bottom_right, cv2.ROTATE_180)
    top_left_flip_rotate = cv2.rotate(top_left_flip, cv2.ROTATE_180)  # 旋转180度
    top_right_flip_rotate = cv2.rotate(top_right_flip, cv2.ROTATE_180)
    bottom_left_flip_rotate = cv2.rotate(bottom_left_flip, cv2.ROTATE_180)
    bottom_right_flip_rotate = cv2.rotate(bottom_right_flip, cv2.ROTATE_180)
    
    # 保存处理后的图像到输出文件夹
    output_file_prefix = os.path.splitext(image_file)[0]
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_left.jpg'), top_left)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_right.jpg'), top_right)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_left.jpg'), bottom_left)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_right.jpg'), bottom_right)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_left_flip.jpg'), top_left_flip)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_right_flip.jpg'), top_right_flip)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_left_flip.jpg'), bottom_left_flip)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_right_flip.jpg'), bottom_right_flip)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_left_rotate.jpg'), top_left_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_right_rotate.jpg'), top_right_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_left_rotate.jpg'), bottom_left_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_right_rotate.jpg'), bottom_right_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_left_flip_rotate.jpg'), top_left_flip_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_top_right_flip_rotate.jpg'), top_right_flip_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_left_flip_rotate.jpg'), bottom_left_flip_rotate)
    cv2.imwrite(os.path.join(output_folder, f'{output_file_prefix}_bottom_right_flip_rotate.jpg'), bottom_right_flip_rotate)

print("处理完成。")