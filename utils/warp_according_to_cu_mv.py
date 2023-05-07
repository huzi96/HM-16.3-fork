import pickle
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--to_parse_fn', type=str, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--input_yuv', type=str, required=True)
parser.add_argument('--input_type', type=str, default='same') # same, high
parser.add_argument('--output_yuv', type=str, required=True)
parser.add_argument('--frame_offset', type=int, default=0)

args = parser.parse_args()

per_frame_cu_list = pickle.load(open(args.to_parse_fn, 'rb'))

# Read YUV 420 file
height = args.height
width = args.width
frame_size = height * width
frame_offset = args.frame_offset * frame_size

with open(args.input_yuv, 'rb') as f:
    # mmap
    yuv = np.memmap(args.input_yuv, dtype=np.uint8, mode='r')
    ref_y = yuv[frame_offset:frame_offset+frame_size].reshape((height, width))
    ref_u = yuv[frame_offset+frame_size:frame_offset+frame_size+frame_size//4].reshape((height//2, width//2))
    ref_v = yuv[frame_offset+frame_size+frame_size//4:frame_offset+frame_size+frame_size//2].reshape((height//2, width//2))
    crt_y = yuv[frame_offset+frame_size+frame_size//2:frame_offset+frame_size+frame_size//2+frame_size].reshape((height, width))
    crt_u = yuv[frame_offset+frame_size+frame_size//2+frame_size:frame_offset+frame_size+frame_size//2+frame_size+frame_size//4].reshape((height//2, width//2))
    crt_v = yuv[frame_offset+frame_size+frame_size//2+frame_size+frame_size//4:frame_offset+frame_size+frame_size//2+frame_size+frame_size//2].reshape((height//2, width//2))

cu_template = {
    'frame_idx': 1,
    'x': 0,
    'y': 0,
    'width': 32,
    'height': 32,
    'mode': 0,
    'pu_list': [],
}
pu_template = {
    'ref_idx': 0,
    'x': 0,
    'y': 0,
    'width': 32,
    'height': 32,
    'y_mv': (0, 0),
    'u_mv': (0, 0),
    'v_mv': (0, 0),
}

current_frame_idx = args.frame_offset + 1
cu_list = per_frame_cu_list[current_frame_idx]
pred_y = ref_y.copy()
pred_v = ref_v.copy()
pred_u = ref_u.copy()
if args.input_type == 'same':
    up_width = width * 4
    up_height = height * 4
elif args.input_type == 'high':
    up_width = width * 2
    up_height = height * 2
inter_ref_y = cv2.resize(ref_y, (up_width, up_height), interpolation=cv2.INTER_LINEAR)
inter_ref_u = cv2.resize(ref_u, (up_width // 2, up_height // 2), interpolation=cv2.INTER_LINEAR)
inter_ref_v = cv2.resize(ref_v, (up_width // 2, up_height // 2), interpolation=cv2.INTER_LINEAR)
# pad refs
padded_inter_ref_y = np.zeros((height * 4 + 16, width * 4 + 16), dtype=np.uint8) + 128
padded_inter_ref_u = np.zeros((height * 2 + 16, width * 2 + 16), dtype=np.uint8) + 128
padded_inter_ref_v = np.zeros((height * 2 + 16, width * 2 + 16), dtype=np.uint8) + 128
padded_inter_ref_y[8:-8, 8:-8] = inter_ref_y
padded_inter_ref_u[8:-8, 8:-8] = inter_ref_u
padded_inter_ref_v[8:-8, 8:-8] = inter_ref_v
poff = 8

def motion_compensation(dst_pred, interpolated_ref, anchor, mv):
    width = dst_pred.shape[1]
    height = dst_pred.shape[0]
    interpolated_ref_anchor = (anchor[0] * 4 + mv[0], anchor[1] * 4 + mv[1])
    inter_width = width * 4
    inter_height = height * 4
    ref_crop = interpolated_ref[interpolated_ref_anchor[1]+poff:interpolated_ref_anchor[1]+inter_height+poff, interpolated_ref_anchor[0]+poff:interpolated_ref_anchor[0]+inter_width+poff]
    dst_pred[...] = cv2.resize(ref_crop, (width, height), interpolation=cv2.INTER_LINEAR)

for cu in cu_list:
    if cu['mode'] == 0:
        # 2N x 2N
        pu = cu['pu_list'][0]
        anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
        cu_width = cu['width']
        cu_height = cu['height']
        motion_compensation(
            dst_pred = pred_y[anchor[1]:anchor[1]+cu_height, anchor[0]:anchor[0]+cu_width],
            interpolated_ref = padded_inter_ref_y,
            anchor = anchor,
            mv = pu['y_mv']
        )
        motion_compensation(
            dst_pred = pred_u[anchor[1]//2:anchor[1]//2+cu_height//2, anchor[0]//2:anchor[0]//2+cu_width//2],
            interpolated_ref = padded_inter_ref_u,
            anchor = (anchor[0]//2, anchor[1]//2),
            mv = (pu['u_mv'][0]//2, pu['u_mv'][1]//2)
        )
        motion_compensation(
            dst_pred = pred_v[anchor[1]//2:anchor[1]//2+cu_height//2, anchor[0]//2:anchor[0]//2+cu_width//2],
            interpolated_ref = padded_inter_ref_v,
            anchor = (anchor[0]//2, anchor[1]//2),
            mv = (pu['v_mv'][0]//2, pu['v_mv'][1]//2)
        )
    else:
        for puid in range(2):
            pu = cu['pu_list'][puid]
            anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
            pu_width = pu['width']
            pu_height = pu['height']
            motion_compensation(
                dst_pred = pred_y[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width],
                interpolated_ref = padded_inter_ref_y,
                anchor = anchor,
                mv = pu['y_mv']
            )
            motion_compensation(
                dst_pred = pred_u[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2],
                interpolated_ref = padded_inter_ref_u,
                anchor = (anchor[0]//2, anchor[1]//2),
                mv = (pu['u_mv'][0]//2, pu['u_mv'][1]//2)
            )
            motion_compensation(
                dst_pred = pred_v[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2],
                interpolated_ref = padded_inter_ref_v,
                anchor = (anchor[0]//2, anchor[1]//2),
                mv = (pu['v_mv'][0]//2, pu['v_mv'][1]//2)
            )

# calculate mse
y_mse = np.mean(np.square(pred_y - crt_y))
u_mse = np.mean(np.square(pred_u - crt_u))
v_mse = np.mean(np.square(pred_v - crt_v))
y_psnr = 10 * np.log10(255 * 255 / y_mse)
u_psnr = 10 * np.log10(255 * 255 / u_mse)
v_psnr = 10 * np.log10(255 * 255 / v_mse)
print('Y PSNR: {:.2f}'.format(y_psnr))
print('U PSNR: {:.2f}'.format(u_psnr))
print('V PSNR: {:.2f}'.format(v_psnr))

# write to yuv
with open(args.output_yuv, 'wb') as f:
    f.write(pred_y.tobytes())
    f.write(pred_u.tobytes())
    f.write(pred_v.tobytes())
