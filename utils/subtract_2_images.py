import cv2
import numpy as np
import os
import sys

def subtract_2_images(img1, img2):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    assert img1.shape == img2.shape
    diff = np.abs(img1 - img2)
    return diff

def main():
    if len(sys.argv) != 4:
        print('Usage: python subtract_2_images.py <img1> <img2> <output>')
        quit()
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    diff = subtract_2_images(img1, img2)
    diff = diff / 255.
    diff = diff ** 0.5
    diff = (diff * 255).astype(np.uint8)
    cv2.imwrite(sys.argv[3], diff)

if __name__ == '__main__':
    main()
