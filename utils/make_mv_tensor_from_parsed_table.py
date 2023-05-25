import pickle
import numpy as np
import cv2
import argparse
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--table_pkl', type=str, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--output_fn', type=str, default='partition_table.npy')
parser.add_argument('--num_frames', type=int, default=1)

args = parser.parse_args()

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

per_frame_cu_list = pickle.load(open(args.table_pkl, 'rb'))
height = args.height
width = args.width
num_frames = args.num_frames
frame_size = height * width

# Dst numpy array
# Channels: [mv_x, mv_y, blk_0, blk_1, inter/intra]
dst = np.zeros((num_frames, height, width, 5), dtype=np.uint8)
blk_h_tmpl = np.array(
    [[-1, -1], [1, 1]],
    dtype=np.float32
)
blk_w_tmpl = np.array(
    [[-1, 1], [-1, 1]],
    dtype=np.float32
)

def draw_block_info(dst, anchor, width, height, pu):
    # Inter mask
    dst[frame_idx,
        anchor[1]:anchor[1]+height,
        anchor[0]:anchor[0]+width, 4] = 1
    # MV
    dst[frame_idx,
        anchor[1]:anchor[1]+height,
        anchor[0]:anchor[0]+width, :2] = pu['y_mv']
    # Block
    blk_h_pattern = cv2.resize(blk_h_tmpl, (width, height), interpolation=cv2.INTER_LINEAR)
    blk_w_pattern = cv2.resize(blk_w_tmpl, (width, height), interpolation=cv2.INTER_LINEAR)
    blk_h_pattern = np.round((blk_h_pattern + 1) * 127.5).astype(np.uint8)
    blk_w_pattern = np.round((blk_w_pattern + 1) * 127.5).astype(np.uint8)
    # with torch.no_grad():
    #     blk_h_pattern = torch.nn.functional.interpolate(
    #         torch.from_numpy(blk_h_tmpl).unsqueeze(0).unsqueeze(0),
    #         size=(height, width),
    #         mode='bilinear',
    #         align_corners=True).squeeze().numpy()
    #     blk_h_pattern = np.round((blk_h_pattern + 1) * 127.5).astype(np.uint8)
    #     blk_w_pattern = torch.nn.functional.interpolate(
    #         torch.from_numpy(blk_w_tmpl).unsqueeze(0).unsqueeze(0),
    #         size=(height, width),
    #         mode='bilinear',
    #         align_corners=True).squeeze().numpy()
    #     blk_w_pattern = np.round((blk_w_pattern + 1) * 127.5).astype(np.uint8)
    dst[frame_idx,
        anchor[1]:anchor[1]+height,
        anchor[0]:anchor[0]+width, 2] = blk_h_pattern
    dst[frame_idx,
        anchor[1]:anchor[1]+height,
        anchor[0]:anchor[0]+width, 3] = blk_w_pattern
    return dst[frame_idx,
               anchor[1]:anchor[1]+height,
               anchor[0]:anchor[0]+width]

for frame_idx in tqdm.tqdm(range(num_frames)):
    if frame_idx == 0:
        # intra frame, all zeros
        continue
    cu_list = per_frame_cu_list[frame_idx]
    for cu in cu_list:
        if cu['mode'] == 0:
            pu = cu['pu_list'][0]
            anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
            cu_width = cu['width']
            cu_height = cu['height']
            draw_block_info(
                dst, anchor, cu_width, cu_height, pu)
        else:
            for puid in range(2):
                pu = cu['pu_list'][puid]
                anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
                pu_width = pu['width']
                pu_height = pu['height']
                draw_block_info(
                    dst, anchor, pu_width, pu_height, pu)

np.save(args.output_fn, dst)
# Visualize a frame
frame_idx = 8
blk_img_x = (np.array(dst[frame_idx, :, :, 2]) + 1) * 127.5
blk_img_x = blk_img_x.astype(np.uint8)
cv2.imwrite('blk_img_x.png', blk_img_x)
blk_img_y = (np.array(dst[frame_idx, :, :, 3]) + 1) * 127.5
blk_img_y = blk_img_y.astype(np.uint8)
cv2.imwrite('blk_img_y.png', blk_img_y)
            
        
