import numpy as np 
import cv2


def estimate_background(image, thr_low=30, thr_high=220):

    gray_image = np.mean(image * 255, axis=2)

    bkg_msk_high = np.where(gray_image > thr_high, np.ones_like(gray_image), np.zeros_like(gray_image))
    bkg_msk_low = np.where(gray_image < thr_low, np.ones_like(gray_image), np.zeros_like(gray_image))

    bkg_msk = np.bitwise_or(bkg_msk_low.astype(np.uint8), bkg_msk_high.astype(np.uint8))
    bkg_msk = cv2.medianBlur(bkg_msk, 7)
    kernel = np.ones((19, 19), np.uint8)
    bkg_msk = cv2.dilate(bkg_msk, kernel)

    bkg_msk = bkg_msk.astype(np.float32)
    return bkg_msk