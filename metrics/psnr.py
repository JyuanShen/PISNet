from skimage.metrics import peak_signal_noise_ratio
import cv2


def calculate_psnr(x_gt, x_test):
    psnr = peak_signal_noise_ratio(x_gt, x_test)
    return psnr


# if __name__ == '__main__':
#     img1 = cv2.imread("H:\\1_Projects\\7_Code\\2_python\\TF-ISNet_00\\data\\test\\y\\mag/00002.png", cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread("H:\\1_Projects\\7_Code\\2_python\\TF-ISNet_00\\data\\test\\x\\mag/00002.png", cv2.IMREAD_GRAYSCALE)
#     print(type(img1))
#     print(img1.size)
#     print(img1.dtype)
#     print(img1.shape)
#     print(calculate_psnr(img1, img2))