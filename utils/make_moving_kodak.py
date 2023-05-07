import cv2
import numpy as np

img = cv2.imread('/mnt/d/Data/kodim/kodim01.png')
# Convert img to YUV
# First convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
yuv_from_rgb_matrix = np.array([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]], dtype=np.float32)
yuv_from_rgb_offset = np.array([0., 0.5, 0.5], dtype=np.float32)
yuv = np.dot(img, yuv_from_rgb_matrix) + yuv_from_rgb_offset
yuv = np.round(np.clip(yuv, 0, 1) * 255.).astype(np.uint8)

off_x = 64
off_y = 64

frame0 = yuv[off_x+0:off_x+128, off_y+0:off_y+128, :]
frame1 = yuv[off_x+0:off_x+128, off_y+1:off_y+129, :]
frame2 = yuv[off_x+1:off_x+129, off_y+1:off_y+129, :]
frame3 = yuv[off_x+2:off_x+130, off_y+2:off_y+130, :]
frames = np.stack([frame0, frame1, frame2, frame3], axis=0).astype(np.uint8)
def convert_yuv444_to_yuv420(yuv444):
    y = yuv444[:, :, 0]
    u = yuv444[:, :, 1]
    v = yuv444[:, :, 2]
    h, w = y.shape
    down_u = cv2.resize(u, (w//2, h//2), interpolation=cv2.INTER_AREA)
    down_v = cv2.resize(v, (w//2, h//2), interpolation=cv2.INTER_AREA)
    return y, down_u, down_v

vid_path = './kodim01_4frames_420.yuv'
with open(vid_path, 'wb') as f:
    for i in range(4):
        y, u, v = convert_yuv444_to_yuv420(frames[i])
        f.write(y.tobytes())
        f.write(u.tobytes())
        f.write(v.tobytes()) 


