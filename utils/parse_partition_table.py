import re
import sys
import numpy as np
import cv2
import argparse
import pickle

# Example string
"""Current Frame Index=3
X=1408, Y=256, Width=32, Height=32
Partition Mode=0, uiWidth=32, uiHeight=32
RefIdx=2
Y-mv: 26,2, refOffset: 6, refStride: 2080, dstStride: 64, xFrac: 2, yFrac: 2, cxWidth: 32, cxHeight: 32
shiftHor: 2, shiftVer: 2

U-mv: 26,2, refOffset: 3, refStride: 1040, dstStride: 32, xFrac: 2, yFrac: 2, cxWidth: 16, cxHeight: 16
shiftHor: 3, shiftVer: 3

V-mv: 26,2, refOffset: 3, refStride: 1040, dstStride: 32, xFrac: 2, yFrac: 2, cxWidth: 16, cxHeight: 16
shiftHor: 3, shiftVer: 3
"""

def parse_partition_table(path):
    # CU template
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

    pat_cu_frame_idx = re.compile(r'Current Frame Index=(\d+)')
    pat_pu_ref_idx = re.compile(r'RefIdx=(\d+)')
    pat_cu = re.compile(r'X=(\d+), Y=(\d+), Width=(\d+), Height=(\d+)')
    pat_mode = re.compile(r'Partition Mode=(\d+), uiWidth=(\d+), uiHeight=(\d+)')
    pat_y = re.compile(r'Y-mv: (-?\d+),(-?\d+), refOffset: (-?\d+), refStride: (\d+), dstStride: (\d+), xFrac: (\d+), yFrac: (\d+), cxWidth: (\d+), cxHeight: (\d+)')
    pat_uv = re.compile(r'[UV]-mv: (-?\d+),(-?\d+), refOffset: (-?\d+), refStride: (\d+), dstStride: (\d+), xFrac: (\d+), yFrac: (\d+), cxWidth: (\d+), cxHeight: (\d+)')
    pat_shift = re.compile(r'shiftHor: (\d+), shiftVer: (\d+)')

    f = open(path, 'r')
    by_frame_cu_list = {}
    
    def parse_a_cu(str_lines, with_frame_idx=True):
        # Parse partition table
        cu_info = pat_cu.findall(str_lines)
        assert cu_info is not None and len(cu_info) == 1
        cu_info_numpy = np.array(cu_info[0], dtype=np.int32)
        cu_mode_info = pat_mode.findall(str_lines)
        assert cu_mode_info is not None and len(cu_mode_info) == 1
        cu_mode_info_numpy = np.array(cu_mode_info[0], dtype=np.int32)
        cu = {
            'x': cu_info_numpy[0],
            'y': cu_info_numpy[1],
            'width': cu_info_numpy[2],
            'height': cu_info_numpy[3],
            'mode': cu_mode_info_numpy[0],
        }
        if with_frame_idx:
            cu_current_frame_idx = pat_cu_frame_idx.findall(str_lines)
            assert cu_current_frame_idx is not None and len(cu_current_frame_idx) == 1, 'str_lines: {}'.format(str_lines)
            cu_current_frame_idx = int(cu_current_frame_idx[0])
            cu['frame_idx'] = cu_current_frame_idx
        uiWidth = cu_mode_info_numpy[1]
        uiHeight = cu_mode_info_numpy[2]
        return cu, uiWidth, uiHeight

    def parse_a_pu(str_lines, ref_cu, x_offset, y_offset, ui_width, ui_height):
        y_info = pat_y.findall(str_lines)
        assert y_info is not None and len(y_info) == 1, 'str_lines: {}'.format(str_lines)
        y_info_numpy = np.array(y_info[0], dtype=np.int32)
        y_cxWidth = y_info_numpy[7]
        y_cxHeight = y_info_numpy[8]
        assert y_cxWidth == ui_width and y_cxHeight == ui_height
        y_xFrac = y_info_numpy[5]
        y_yFrac = y_info_numpy[6]
        y_mv = (y_info_numpy[0], y_info_numpy[1])
        assert y_xFrac == y_mv[0] % 4 and y_yFrac == y_mv[1] % 4
        
        uv_info = pat_uv.findall(str_lines)
        assert uv_info is not None and len(uv_info) == 2, 'str_lines: {}'.format(str_lines)
        uv_info_numpy = np.array(uv_info, dtype=np.int32)

        pu_ref_idx = pat_pu_ref_idx.findall(str_lines)
        assert pu_ref_idx is not None and len(pu_ref_idx) == 1
        pu_ref_idx = int(pu_ref_idx[0])

        pu = {
            'ref_idx': pu_ref_idx,
            'x': x_offset,
            'y': y_offset,
            'width': y_cxWidth,
            'height': y_cxHeight,
            'y_mv': y_mv,
            'u_mv': (uv_info_numpy[0, 0], uv_info_numpy[0, 1]),
            'v_mv': (uv_info_numpy[1, 0], uv_info_numpy[1, 1]),
        }
        return pu

    while True:
        # Read 11 lines
        lines = [f.readline() for i in range(13)]
        # Check EOF
        if lines[-1] == '':
            break
        # Combine lines
        lines = ''.join(lines)

        cu, uiWidth, uiHeight = parse_a_cu(lines)
        if not cu['frame_idx'] in by_frame_cu_list:
            by_frame_cu_list[cu['frame_idx']] = []

        if cu['mode'] == 0:
            # 2Nx2N only one PU
            pu = parse_a_pu(lines, cu, 0, 0, uiWidth, uiHeight)
            cu['pu_list'] = [pu]
        else:
            # other cases
            # read another 11 lines
            lines = [f.readline() for i in range(12)]
            assert lines[-1] != '', 'Unexpected EOF'
            lines = ''.join(lines)
            sub_cu, sub_uiWidth, sub_uiHeight = parse_a_cu(lines, with_frame_idx=False)
            assert (sub_uiWidth + uiWidth) % cu['width'] == 0 and (sub_uiHeight + uiHeight) % cu['height'] == 0
            assert sub_cu['mode'] != 0
            assert sub_cu['width'] == cu['width'] and sub_cu['height'] == cu['height']
            
            pu_2 = parse_a_pu(lines, sub_cu,
                              uiWidth % cu['width'],
                              uiHeight % cu['height'],
                              sub_uiWidth, sub_uiHeight)
            cu['pu_list'] = [pu, pu_2]
        by_frame_cu_list[cu['frame_idx']].append(cu)

    f.close()
    return by_frame_cu_list

# to_parse_fn = sys.argv[1]
# cu_list = parse_partition_table(to_parse_fn)
# import IPython
# IPython.embed()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--to_parse_fn', type=str, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--output_fn', type=str, default='partition_table.pkl')

    args = parser.parse_args()

    by_frame_cu_list = parse_partition_table(args.to_parse_fn)
    cu_list = by_frame_cu_list[1]

    # visualize partition table
    img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    def draw_block_with_2pix_wide_border(height, width, color=(255, 255, 255)):
        block = np.zeros((height, width, 3), dtype=np.uint8)
        block[2:-2, 2:-2, :] = np.array(color, dtype=np.uint8)
        return block
    for cu in cu_list:
        x = cu['x']
        y = cu['y']
        width = cu['width']
        height = cu['height']
        img[y:y+height, x:x+width, :] = draw_block_with_2pix_wide_border(height, width, color=(255, 0, 0))
        if len(cu['pu_list']) == 1:
            continue
        else:
            for i, pu in enumerate(cu['pu_list']):
                pu_x = x + pu['x']
                pu_y = y + pu['y']
                pu_width = pu['width']
                pu_height = pu['height']
                img[pu_y:pu_y+pu_height, pu_x:pu_x+pu_width, :] = draw_block_with_2pix_wide_border(pu_height, pu_width, color=(0, 255*i, 255*(1-i)))
    cv2.imwrite('partition_table.png', img // 2)

    # visualize motion vectors
    # and fill another image with block of motion vectors
    img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    mv_img = np.zeros((args.height, args.width, 6), dtype=int)
    blk_img = np.zeros((args.height, args.width, 2), dtype=np.uint8)

    def draw_block_of_mv(height, width, mv, color=(255, 255, 255)):
        block = np.zeros((height, width, 3), dtype=np.uint8)
        # block[2:-2, 2:-2, :] = np.array(color, dtype=np.uint8)
        block = cv2.arrowedLine(block, (0, 0), (mv[0], mv[1]), (255, 255, 255), 1)
        return block
    def draw_block_of_grid(height, width):
        block = np.zeros((2, 2, 2), dtype=np.float32)
        block[0, :, 0] = -1
        block[1, :, 0] = 1
        block[:, 0, 1] = -1
        block[:, 1, 1] = 1
        block = cv2.resize(block, (width, height), interpolation=cv2.INTER_LINEAR)
        return block
    
    for cu in cu_list:
        x = cu['x']
        y = cu['y']
        width = cu['width']
        height = cu['height']
        if len(cu['pu_list']) == 1:
            img[y:y+height, x:x+width, :] = draw_block_of_mv(height, width, cu['pu_list'][0]['y_mv'], color=(255, 0, 0))
            mv_img[y:y+height, x:x+width, 0:2] = np.array(cu['pu_list'][0]['y_mv'], dtype=int)
            mv_img[y:y+height, x:x+width, 2:4] = np.array(cu['pu_list'][0]['u_mv'], dtype=int)
            mv_img[y:y+height, x:x+width, 4:6] = np.array(cu['pu_list'][0]['v_mv'], dtype=int)

        else:
            for i, pu in enumerate(cu['pu_list']):
                pu_x = x + pu['x']
                pu_y = y + pu['y']
                pu_width = pu['width']
                pu_height = pu['height']
                img[pu_y:pu_y+pu_height, pu_x:pu_x+pu_width, :] = draw_block_of_mv(pu_height, pu_width, pu['y_mv'], color=(0, 255*i, 255*(1-i)))
    cv2.imwrite('motion_vectors.png', img // 2)
    np.save('motion_vectors.npy', mv_img)

    # Dump cu_list
    with open(args.output_fn, 'wb') as f:
        pickle.dump(by_frame_cu_list, f)
    