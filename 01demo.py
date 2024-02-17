# 导入必要的库
import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 导入segment-anything包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# 输入必要的参数demo：
model_path = "model/sam_vit_h_4b8939.pth"  # 默认权重

image_path = "notebooks/images/109.png"  # 需要分割的图片
output_floder = "result_test_png/109/"  # 输出mask图片保存的文件夹

# 红色通道图片测试
# image_path = "notebooks/images/090_red.png"
# output_floder = "result_test_red/090/"


# mask单张图片测试
# image_path = "imageAndMask/090/green/green_15.png"  # 需要分割的图片
# output_floder = "singlePredictor/090/15"  # 输出mask图片保存的文件夹

# img = cv2.imread(image_path)
# plt.imshow(img, cmap='gray')
# plt.show()

# spilt方法抠图结果测试
# image_path = "imageAndMask/090/res_15_split.jpg"  # 需要分割的图片
# output_floder = "singleMaskPredictor/090/"  # 输出图片保存的文件夹


print("image path= ", image_path)
# 查看图像通道数:
image_shape = cv2.imread(image_path)
print("image shape= ", image_shape.shape)

# 确保输出文件夹存在
os.makedirs(output_floder, exist_ok=True)

# 官方demo加载模型的方式
sam_model = sam_model_registry["vit_h"](checkpoint=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = sam_model.to(device)

# 输出模型加载完成的当前时间：
current_time_start = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())
print("Model loaded start... ", current_time_start)

# 输出模型结束时间
current_time_end = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())
print("Model loaded end... ", current_time_end)

# 这里是加载图片
# 将图像转换为SAM模型期望的格式
from segment_anything.utils.transforms import ResizeLongestSide

image_demo = cv2.imread(image_path)
# image = cv2.cvtColor(image_demo, cv2.COLOR_BGR2RGB)
# transform = ResizeLongestSide(sam_model.image_encoder.img_size)
# input_image = transform.apply_image(image)
# input_image_torch = torch.as_tensor(input_image, device=device)
# transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
# input_image = sam_model.preprocess(transformed_image)

# 进行全图分割：
mask_generator = SamAutomaticMaskGenerator(sam_model)
masks = mask_generator.generate(image_demo)   #from segment_anything/utils/automatic_mask_generator.py

# 输出模型预测结束时间
predict_time_end = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())
print("Model predict end... ", predict_time_end)

# 输出第一个腺体mask[0]的面积
print(masks[0])  # 得到的mask为一张张灰度图，需要后面进行合并

print(masks[1])

# gray = cv2.cvtColor(image_demo, cv2.COLOR_BGR2GRAY)
# x1, y1, w, h = masks[0]['bbox']
# draw_1 = cv2.rectangle(gray, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 0, 255), 1)
# # 保存掩码
# cv2.imwrite(output_floder + "box.png", draw_1)


import xlwt
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)

arealist = list()


# 循环统计mask面积
# 遍历 masks 列表并保存每个掩码：
for i, mask in enumerate(masks):
    mask_array = mask['segmentation']
    mask_unit8 = (mask_array * 255).astype(np.uint8)

    # 为每个掩码生成一个唯一的文件名：
    output_filename = os.path.join(output_floder, f"mask_{i + 1}.png")
    arealist.append(mask['area'])
    # 保存掩码
    # cv2.imwrite(output_filename, mask_unit8)

# print("datalist len=", len = len(datalist))

# 输出完整的mask
# 获取输入图像的尺寸
height, width, _ = image_demo.shape

# 创建一个全零数组，用于合并掩码：
merged_mask = np.zeros((height, width), dtype=np.uint8)


#统计细胞浆腺体数量
num_nucleus = 0
# 遍历masks列表并合并每个掩码
for i, mask in enumerate(masks):
    mask_array = mask['segmentation']
    mask_unit8 = (mask_array * 255).astype(np.uint8)

    # 为每个掩码生成一个唯一的文件名：
    output_filename = os.path.join(output_floder, f"mask_{i + 1}.png")
    num_nucleus = num_nucleus + 1

    print(mask['bbox'])
    # 输出每一个细胞核面积, area (int): The area in pixels of the mask.
    print("number=", num_nucleus, "  area=", mask['area'])

    cv2.imwrite(output_filename, mask_unit8)

    # 将当前掩码添加到合并掩码上
    merged_mask = np.maximum(merged_mask, mask_unit8)


print("细胞浆腺体总共数量=", (num_nucleus), " 个")

arealist_len = len(arealist)

print("arealist_len=", arealist_len)

# 将数据保存到excel中
for i in range(0,arealist_len):
    sheet.write(i,1,arealist[i])

savepath = 'excel_nuclues_109.xls'
# book.save(savepath)

# 保存合并后的掩码
save_imagename = "mask_all_109.png"
merged_output_file = os.path.join(output_floder, save_imagename)

# 保存合并后的掩码
cv2.imwrite(merged_output_file, merged_mask)

# 展示预测结果image 和 mask
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(image_demo)
# plt.subplot(1,2,2)
# plt.imshow(masks[0]) # plt.show()


# 释放cv2
# cv2.destroyAllWindows()
