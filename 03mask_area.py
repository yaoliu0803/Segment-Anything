# -*- coding=GBK -*-
import cv2 as cv
import numpy as np

# 单张图片测试
# src = cv.imread("green/res_15_black.jpg")
# # cv.imshow("before", src)
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # 转灰度图
# #像素值大于10的取值为255
# retval, syn_binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
# # retval, syn_binary，第一个返回值是**阈值**的float型，第二个返回值是图片像素处理后的结果。
# print("设置阈值大小为= ", retval)
# print(syn_binary.shape)
# cv.imwrite("save_new_res_mask.png", syn_binary)
#
# thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
#
# # mask 面积计算：
# img = np.array(syn_binary)
# x, y = syn_binary.shape  #x:高；y:宽
# pixels = 0
# background_area = 0
# for row in range(x):
#     for col in range(y):
#         if (img[row][col]) != 0:
#             pixels = pixels+1
#         else:
#             background_area = background_area+1
#
# mask_area = pixels
# mask_area_rate = mask_area/(x*y)
# # image_area = (src.shape[0]) * (src.shape[1])
# print("mask_area=",mask_area)
# print("mask_area_rate=",mask_area_rate)


#多张图片

def mask_area_cul(image_path):
    src = cv.imread(image_path)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # 转灰度图
    # #像素值大于10的取值为255
    retval, syn_binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    # retval, syn_binary，第一个返回值是**阈值**的float型，第二个返回值是图片像素处理后的结果。
    print("设置阈值大小为= ", retval)
    print(syn_binary.shape)
    # cv.imwrite("save_new_res_mask.png", syn_binary)
    thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    # mask 面积计算：
    img = np.array(syn_binary)
    x, y = syn_binary.shape  #x:高；y:宽
    pixels = 0
    background_area = 0
    for row in range(x):
        for col in range(y):
            if (img[row][col]) != 0:
                pixels = pixels+1
            else:
                background_area = background_area+1
    mask_area = pixels
    print("mask_area=",mask_area)
    return mask_area

import os
import xlwt

if __name__ == "__main__":
    mask_path = "imageAndMask/090/png/"
    # mask_path = "result_test_png/090/"
    files_list = os.listdir(mask_path)  # 得到文件夹下的所有文件名称，存在字符串列表中
    print("files_list=", files_list, "len=", len(files_list))


    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
    nucleus_area_list = list()

    for i in range(0, len(files_list)-1):
        image_path = os.path.join(mask_path, f"res_mask_{i + 1}.jpg")
        print("image_path=", image_path)
        area = mask_area_cul(image_path)  # 保存格式为jpg
        nucleus_area_list.append(area)
        print("ending")

    print(nucleus_area_list)
    len = len(nucleus_area_list)
    for i in range(0, len):
        sheet.write(i, 0, nucleus_area_list[i])
    savepath = 'excel_nuclues_3.xls'
    book.save(savepath)



