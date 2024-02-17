import os
import cv2
import numpy as np
from PIL import Image


# 分割是透明背景32位的
def gen_foreground(img_path, mask_path, res_path):
    print("img_path= ", img_path)
    print("mask_path= ", mask_path)
    print("res_path= ", res_path)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    print("image shape = ", img.shape)
    height, width, channel = img.shape
    b, g, r = cv2.split(img)
    res = np.zeros((4, height, width), dtype=img.dtype)
    res[0][0:height, 0:width] = b
    res[1][0:height, 0:width] = g
    res[2][0:height, 0:width] = r
    res[3][0:height, 0:width] = mask
    print("res shape=", res.shape)
    cv2.imwrite(res_path, cv2.merge(res))


# 分割是黑色背景的
def gen_blackground(img_path, mask_path, res_path):
    print("img_path= ", img_path)
    print("mask_path= ", mask_path)
    print("res_path= ", res_path)
    test_image = cv2.imread(img_path)
    # back = cv2.imread("背景图")
    # 这里将mask图转化为灰度图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 将背景图resize到和原图一样的尺寸
    # back = cv2.resize(back, (person.shape[1], person.shape[0]))
    # 这一步是将背景图中的关注部分抠出来，也就是关注部分的像素值为0
    # scenic_mask = ~mask
    # scenic_mask = scenic_mask / 255.0
    # back[:, :, 0] = back[:, :, 0] * scenic_mask
    # back[:, :, 1] = back[:, :, 1] * scenic_mask
    # back[:, :, 2] = back[:, :, 2] * scenic_mask
    # 这部分是将我们的关注部分抠出来，也就是背景部分的像素值为0
    mask = mask / 255.0
    test_image[:, :, 0] = test_image[:, :, 0] * mask
    test_image[:, :, 1] = test_image[:, :, 1] * mask
    test_image[:, :, 2] = test_image[:, :, 2] * mask
    # 这里做个相加就可以实现合并
    # result = cv2.add(back, person)
    print("test_image shape=",test_image.shape)
    cv2.imwrite(res_path, test_image)
    print()



import os

if __name__ == "__main__":
    # image_path = "notebooks/images/090.png"
    # mask_path = "result_test_png/090/mask_15.png"
    # image_and_mask_floder = "imageAndMask/090/green"
    #
    # # 生成黑色背景：
    # image_and_mask_result = os.path.join(image_and_mask_floder, "res_15_black.jpg")
    # gen_blackground(image_path, mask_path, image_and_mask_result)  #保存格式为jpg


    # image_path = "notebooks/images/090.png"
    mask_path = "result_test_png/109/"
    files_list = os.listdir(mask_path)  # 得到文件夹下的所有文件名称，存在字符串列表中
    print("files_list=", files_list, "len=", len(files_list)-1)

    result_path = "imageAndMask/109/png/"
    # 确保输出文件夹存在
    os.makedirs(result_path, exist_ok=True)
    for i in range(0, len(files_list)-1):
        # print("load image...", mask_image)
        image_path = "notebooks/images/109_green.png"
        mask_path = "result_test_png/109/mask_"+str(i + 1)+".png"
        print("mask_path=", mask_path)
        image_and_mask_result = os.path.join(result_path, f"res_mask_{i + 1}.jpg")
        gen_blackground(image_path, mask_path, image_and_mask_result)  # 保存格式为jpg
        i = i+1

    print("ending")


    # 生成透明背景：ls
    # image_and_mask_result = os.path.join(image_and_mask_floder, "green_15.png")
    # gen_foreground(image_path, mask_path, image_and_mask_result)  #注意事项：保存的结果图像应该是png格式，而不能是jpg等格式，因为只有png才存在alpha通道表透明度的概念



