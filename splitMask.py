import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# image_path = "notebooks/images/test_row.jpg"


# # for name in os.listdir(imagefile_path):
#     img=cv2.imread(os.path.join(imagefile_path,name))
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     plt.imshow(img,cmap='gray')
#     plt.show()
#     msk=cv2.imread('C:\\Users\\DELL\\Desktop\\internship\\auto_focus\\20220302_173219_CCMI2020_2_copy\\mask\\mask.jpg',cv2.IMREAD_GRAYSCALE)
#     msk_cov = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=msk) #将image的相素值和mask像素值相加得到结果
#     plt.imshow(msk_cov,cmap='gray')
#     plt.show()

def mask_split(imagefile_path_name, mask_path, image_and_mask_result):
    img = cv2.imread(imagefile_path_name)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    plt.show()
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    msk_cov = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=msk)  # 将image的相素值和mask像素值相加得到结果
    print("msk_cov shape= ", msk_cov.shape)
    # plt.imshow(msk_cov, cmap='gray')
    plt.imshow(msk_cov)
    print("split end...")
    plt.show()
    cv2.imwrite(image_and_mask_result, msk_cov)
    print("save image success...")


def mask_split_2(imagefile_path_name, mask_path, image_and_mask_result):
    img1 = cv2.imread(imagefile_path_name)
    img2 = cv2.imread(mask_path)
    alpha = 0.5
    meta = 1 - alpha
    gamma = 0
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # image = cv2.addWeighted(img1,alpha,img2,meta,gamma)
    image = cv2.add(img1, img2)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(image_and_mask_result, image)
    print("split end...")


if __name__ == "__main__":


    image_path = "notebooks/images/test_row.jpg"
    mask_path = "result_test/090/"

    # 保存图片：
    image_and_mask_floder = "imageAndMask/090/"
    files_list = os.listdir(mask_path)  # 得到文件夹下的所有文件名称，存在字符串列表中
    for mask_image in files_list:
        mask_image = mask_image
        image_and_mask_result = os.path.join(image_and_mask_floder, "res_15_split_2.jpg")
        # 确保输出文件夹存在
        os.makedirs(image_and_mask_floder, exist_ok=True)
        mask_split(image_path, mask_path, image_and_mask_result)


    # 方法一：
    #

    # 方法二：
    # mask_split_2(image_path, mask_path, image_and_mask_result)
