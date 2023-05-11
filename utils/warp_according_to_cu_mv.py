import pickle
import numpy as np
import cv2
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--to_parse_fn', type=str, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--input_yuv', type=str, required=True)
parser.add_argument('--gt_yuv', type=str, required=True)
parser.add_argument('--input_type', type=str, default='same') # same, high
parser.add_argument('--output_yuv', type=str, required=True)
parser.add_argument('--num_frame', type=int, default=4)
parser.add_argument('--pad_size', type=int, default=64)
parser.add_argument('--output_mask', type=str, default='mask.npy')

args = parser.parse_args()

per_frame_cu_list = pickle.load(open(args.to_parse_fn, 'rb'))

# Read YUV 420 file
height = args.height
width = args.width
frame_size = height * width

if args.input_type == 'same':
    up_width = width * 4
    up_height = height * 4
    fct = 1
elif args.input_type == 'high':
    up_width = width * 4
    up_height = height * 4
    fct = 2

def read_yuv_file(fn, frame_size, height, width):
    with open(fn, 'rb') as f:
        # mmap
        yuv = np.memmap(fn, dtype=np.uint8, mode='r')
        # read into memory
        y = [yuv[i:i+frame_size] for i in range(0, len(yuv), frame_size + frame_size // 2)]
        u = [yuv[i:i+frame_size//4] for i in range(frame_size, len(yuv), frame_size + frame_size // 2)]
        v = [yuv[i:i+frame_size//4] for i in range(frame_size + frame_size // 4, len(yuv), frame_size + frame_size // 2)]
        # convert to numpy
        y = np.stack(y, axis=0).reshape((-1, height, width))
        u = np.stack(u, axis=0).reshape((-1, height // 2, width // 2))
        v = np.stack(v, axis=0).reshape((-1, height // 2, width // 2))
    return y, u, v

y, u, v = read_yuv_file(args.input_yuv, frame_size, height, width)
gt_y, gt_u, gt_v = read_yuv_file(args.gt_yuv, frame_size, height, width)

def get_ref_list(y, u, v, crt_frame_idx):
    start = max(0, crt_frame_idx - 4)
    end = crt_frame_idx
    # reverse arrange
    ref_y = np.array(y[start:end][::-1])
    ref_u = np.array(u[start:end][::-1])
    ref_v = np.array(v[start:end][::-1])
    return ref_y, ref_u, ref_v

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

pred_y_list = []
pred_u_list = []
pred_v_list = []
mask_y_list = []
mask_uv_list = []

for current_frame_idx in range(1, args.num_frame):
    cu_list = per_frame_cu_list[current_frame_idx]
    # ref_y shape: (num_ref, height, width)
    ref_y, ref_u, ref_v = get_ref_list(y, u, v, current_frame_idx)
    pred_y = ref_y[0].copy()
    pred_v = ref_v[0].copy()
    pred_u = ref_u[0].copy()
    mask_y = np.zeros_like(pred_y, dtype=np.uint8)
    mask_uv = np.zeros_like(pred_u, dtype=np.uint8)
    # if args.input_type == 'high':
    #     pred_y = cv2.resize(pred_y, (up_width // 4, up_height // 4), interpolation=cv2.INTER_LINEAR)
    #     pred_u = cv2.resize(pred_u, (up_width // 8, up_height // 8), interpolation=cv2.INTER_LINEAR)
    #     pred_v = cv2.resize(pred_v, (up_width // 8, up_height // 8), interpolation=cv2.INTER_LINEAR)

    # inter_ref_y = cv2.resize(ref_y, (up_width, up_height), interpolation=cv2.INTER_LINEAR)
    # inter_ref_u = cv2.resize(ref_u, (up_width // 2, up_height // 2), interpolation=cv2.INTER_LINEAR)
    # inter_ref_v = cv2.resize(ref_v, (up_width // 2, up_height // 2), interpolation=cv2.INTER_LINEAR)
    with torch.no_grad():
        # Use torch batch resize instead of cv2.resize
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # inter_ref_y shape: (num_ref, 1, height, width)
        inter_ref_y = torch.from_numpy(ref_y).to(device).float().unsqueeze(1)
        inter_ref_u = torch.from_numpy(ref_u).to(device).float().unsqueeze(1)
        inter_ref_v = torch.from_numpy(ref_v).to(device).float().unsqueeze(1)
        inter_ref_y = torch.nn.functional.interpolate(inter_ref_y, size=(up_height, up_width), mode='bilinear', align_corners=False)
        inter_ref_u = torch.nn.functional.interpolate(inter_ref_u, size=(up_height // 2, up_width // 2), mode='bilinear', align_corners=False)
        inter_ref_v = torch.nn.functional.interpolate(inter_ref_v, size=(up_height // 2, up_width // 2), mode='bilinear', align_corners=False)

        # pad refs with torch
        pad_size = args.pad_size
        padded_inter_ref_y = torch.nn.functional.pad(inter_ref_y, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        padded_inter_ref_u = torch.nn.functional.pad(inter_ref_u, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        padded_inter_ref_v = torch.nn.functional.pad(inter_ref_v, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

        # convert to numpy
        padded_inter_ref_y = padded_inter_ref_y.squeeze(1).cpu().numpy()
        padded_inter_ref_u = padded_inter_ref_u.squeeze(1).cpu().numpy()
        padded_inter_ref_v = padded_inter_ref_v.squeeze(1).cpu().numpy()

    def motion_compensation(dst_pred, interpolated_ref, anchor, mv, ref_idx, pad_size):
        width = dst_pred.shape[1]
        height = dst_pred.shape[0]
        interpolated_ref_anchor = (anchor[0] * 4 + mv[0], anchor[1] * 4 + mv[1])
        inter_width = width * 4
        inter_height = height * 4
        ref_crop = interpolated_ref[
            ref_idx,
            interpolated_ref_anchor[1]+pad_size:interpolated_ref_anchor[1]+inter_height+pad_size,
            interpolated_ref_anchor[0]+pad_size:interpolated_ref_anchor[0]+inter_width+pad_size]
        try:
            dst_pred[...] = cv2.resize(ref_crop, (width, height), interpolation=cv2.INTER_LINEAR)
        except:
            print('Error in MC')
            import IPython
            IPython.embed()
            quit()
    for cu in cu_list:
        if cu['mode'] == 0:
            # 2N x 2N
            pu = cu['pu_list'][0]
            anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
            cu_width = cu['width']
            cu_height = cu['height']
            if args.input_type == 'high':
                anchor = (anchor[0] * 2, anchor[1] * 2)
                cu_width *= 2
                cu_height *= 2
            motion_compensation(
                dst_pred=pred_y[anchor[1]:anchor[1]+cu_height, anchor[0]:anchor[0]+cu_width],
                interpolated_ref=padded_inter_ref_y,
                anchor=anchor,
                mv=(pu['y_mv'][0] * fct, pu['y_mv'][1] * fct),
                ref_idx=pu['ref_idx'],
                pad_size=pad_size
            )
            motion_compensation(
                dst_pred=pred_u[anchor[1]//2:anchor[1]//2+cu_height//2, anchor[0]//2:anchor[0]//2+cu_width//2],
                interpolated_ref=padded_inter_ref_u,
                anchor=(anchor[0]//2, anchor[1]//2),
                mv=(pu['u_mv'][0]//2 * fct, pu['u_mv'][1]//2 * fct),
                ref_idx=pu['ref_idx'],
                pad_size=pad_size
            )
            motion_compensation(
                dst_pred=pred_v[anchor[1]//2:anchor[1]//2+cu_height//2, anchor[0]//2:anchor[0]//2+cu_width//2],
                interpolated_ref=padded_inter_ref_v,
                anchor=(anchor[0]//2, anchor[1]//2),
                mv=(pu['v_mv'][0]//2 * fct, pu['v_mv'][1]//2 * fct),
                ref_idx=pu['ref_idx'],
                pad_size=pad_size
            )
            mask_y[anchor[1]:anchor[1]+cu_height, anchor[0]:anchor[0]+cu_width] = 1
            mask_uv[anchor[1]//2:anchor[1]//2+cu_height//2, anchor[0]//2:anchor[0]//2+cu_width//2] = 1
        else:
            for puid in range(2):
                pu = cu['pu_list'][puid]
                anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
                pu_width = pu['width']
                pu_height = pu['height']
                if args.input_type == 'high':
                    anchor = (anchor[0] * 2, anchor[1] * 2)
                    pu_width *= 2
                    pu_height *= 2
                motion_compensation(
                    dst_pred=pred_y[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width],
                    interpolated_ref=padded_inter_ref_y,
                    anchor=anchor,
                    mv=(pu['y_mv'][0] * fct, pu['y_mv'][1] * fct),
                    ref_idx=pu['ref_idx'],
                    pad_size=pad_size
                )
                motion_compensation(
                    dst_pred=pred_u[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2],
                    interpolated_ref=padded_inter_ref_u,
                    anchor=(anchor[0]//2, anchor[1]//2),
                    mv=(pu['u_mv'][0]//2*fct, pu['u_mv'][1]//2*fct),
                    ref_idx=pu['ref_idx'],
                    pad_size=pad_size
                )
                motion_compensation(
                    dst_pred=pred_v[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2],
                    interpolated_ref=padded_inter_ref_v,
                    anchor=(anchor[0]//2, anchor[1]//2),
                    mv=(pu['v_mv'][0]//2*fct, pu['v_mv'][1]//2*fct),
                    ref_idx=pu['ref_idx'],
                    pad_size=pad_size
                )
                mask_y[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width] = 1
                mask_uv[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2] = 1
    crt_y = gt_y[current_frame_idx]
    crt_u = gt_u[current_frame_idx]
    crt_v = gt_v[current_frame_idx]
    # calculate mse
    y_mse = np.sum(np.square(pred_y.astype(float) - crt_y.astype(float)) * mask_y) / np.sum(mask_y)
    u_mse = np.sum(np.square(pred_u.astype(float) - crt_u.astype(float)) * mask_uv) / np.sum(mask_uv)
    v_mse = np.sum(np.square(pred_v.astype(float) - crt_v.astype(float)) * mask_uv) / np.sum(mask_uv)
    y_psnr = 10 * np.log10(255 * 255 / y_mse)
    u_psnr = 10 * np.log10(255 * 255 / u_mse)
    v_psnr = 10 * np.log10(255 * 255 / v_mse)
    print(f'frame {current_frame_idx} psnr {y_psnr:.2f} {u_psnr:.2f} {v_psnr:.2f}')
    pred_y_list.append(pred_y)
    pred_u_list.append(pred_u)
    pred_v_list.append(pred_v)
    mask_y_list.append(mask_y)
    mask_uv_list.append(mask_uv)
# write to yuv
with open(args.output_yuv, 'wb') as f:
    for i in range(args.num_frame - 1):
        f.write(pred_y_list[i].tobytes())
        f.write(pred_u_list[i].tobytes())
        f.write(pred_v_list[i].tobytes())
# save mask
mask_y = np.stack(mask_y_list, axis=0)
mask_uv = np.stack(mask_uv_list, axis=0)
np.save(f'{args.output_mask}_y.npy', mask_y)
np.save(f'{args.output_mask}_uv.npy', mask_uv)


