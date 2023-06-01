import pickle
import numpy as np
import cv2
import argparse
import torch
from typing import Tuple, List, Iterable

class YUVFrames(object):
    """Returns mmap-ed yuv frames"""
    def __init__(self, fn, height, width):
        self.yuv = np.memmap(fn, dtype=np.uint8, mode='r')
        self.frame_size = height * width
        self.height = height
        self.width = width
    
    def __getitem__(self, idx: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = idx * self.frame_size * 3 // 2
        y = self.yuv[start:start+self.frame_size]
        u = self.yuv[start+self.frame_size:start+self.frame_size+self.frame_size//4]
        v = self.yuv[start+self.frame_size+self.frame_size//4:start+self.frame_size+self.frame_size//2]
        y = y.reshape((height, width))
        u = u.reshape((height // 2, width // 2))
        v = v.reshape((height // 2, width // 2))
        return y, u, v

class SubPixMotionCompensationEngine(object):
    """Module for MC with manual fractional interpolation.
    
    To save memory, we maintain a interpolated ref frame list.
    """
    def __init__(self,
                 per_frame_cu_list, yuv: YUVFrames, pad_size=64) -> None:
        self.per_frame_cu_list = per_frame_cu_list
        self.yuv = yuv
        self.height = yuv.height
        self.width = yuv.width
        self.ref_y_list = []
        self.ref_u_list = []
        self.ref_v_list = []
        self.pad_size = pad_size
    
    def upsample_frame(self, frame, factor=4, method=cv2.INTER_CUBIC):
        """Upsample frame to target size"""
        target_size = (frame.shape[1] * factor, frame.shape[0] * factor)
        return cv2.resize(frame, target_size, interpolation=method)
    
    def populate_ref_list(self, crt_frame_idx):
        """Populate reference list for current frame"""
        start = max(0, crt_frame_idx - 4)
        end = crt_frame_idx
        n_frames = end - start
        # append a new frame
        y_tminus1, u_tminus1, v_tminus1 = self.yuv[end - 1]
        y_up = self.upsample_frame(y_tminus1)
        u_up = self.upsample_frame(u_tminus1)
        v_up = self.upsample_frame(v_tminus1)
        # pad
        pad_size = self.pad_size
        y_up = np.pad(
            y_up,
            ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
        u_up = np.pad(
            u_up,
            ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
        v_up = np.pad(
            v_up,
            ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
        
        # append to ref list
        self.ref_y_list = [y_up] + self.ref_y_list
        self.ref_u_list = [u_up] + self.ref_u_list
        self.ref_v_list = [v_up] + self.ref_v_list

        # truncate if exceed ref buffer
        if len(self.ref_y_list) > 4:
            self.ref_y_list = self.ref_y_list[:4]
            self.ref_u_list = self.ref_u_list[:4]
            self.ref_v_list = self.ref_v_list[:4]
        
        return self.ref_y_list, self.ref_u_list, self.ref_v_list

    def block_motion_compensation(
            self, dst_pred, interpolated_ref, anchor, mv, ref_idx, pad_size):
        width = dst_pred.shape[1]
        height = dst_pred.shape[0]
        interpolated_ref_anchor = (anchor[0] * 4 + mv[0], anchor[1] * 4 + mv[1])
        inter_width = width * 4
        inter_height = height * 4
        ref_frame = interpolated_ref[ref_idx]
        ref_crop = ref_frame[
            interpolated_ref_anchor[1]+pad_size:interpolated_ref_anchor[1]+inter_height+pad_size,
            interpolated_ref_anchor[0]+pad_size:interpolated_ref_anchor[0]+inter_width+pad_size]
        try:
            dst_pred[...] = cv2.resize(ref_crop, (width, height), interpolation=cv2.INTER_LINEAR)
        except:
            print('Error in MC')
            import IPython
            IPython.embed()
            quit()
    
    def motion_compensate(
            self, crt_frame_idx,
            default_pred_y=None,
            default_pred_u=None,
            default_pred_v=None):
        """Motion compensate current frame"""
        cu_list = self.per_frame_cu_list[crt_frame_idx]
        if default_pred_y is None:
            default_pred_y = np.zeros((self.height, self.width), dtype=np.uint8)
        if default_pred_u is None:
            default_pred_u = np.zeros((self.height // 2, self.width // 2), dtype=np.uint8)
        if default_pred_v is None:
            default_pred_v = np.zeros((self.height // 2, self.width // 2), dtype=np.uint8)
        pred_y = default_pred_y
        pred_u = default_pred_u
        pred_v = default_pred_v
        # populate ref list
        # self.populate_ref_list(crt_frame_idx)

        for cu in cu_list:
            for puid in range(len(cu['pu_list'])):
                pu = cu['pu_list'][puid]
                anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
                pu_width = pu['width']
                pu_height = pu['height']
                if args.input_type == 'high':
                    anchor = (anchor[0] * 2, anchor[1] * 2)
                    pu_width *= 2
                    pu_height *= 2
                self.block_motion_compensation(
                    dst_pred=pred_y[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width],
                    interpolated_ref=self.ref_y_list,
                    anchor=anchor,
                    mv=(pu['y_mv'][0] * fct, pu['y_mv'][1] * fct),
                    ref_idx=pu['ref_idx'],
                    pad_size=self.pad_size
                )
                self.block_motion_compensation(
                    dst_pred=pred_u[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2],
                    interpolated_ref=self.ref_u_list,
                    anchor=(anchor[0]//2, anchor[1]//2),
                    mv=(pu['u_mv'][0]//2*fct, pu['u_mv'][1]//2*fct),
                    ref_idx=pu['ref_idx'],
                    pad_size=self.pad_size
                )
                self.block_motion_compensation(
                    dst_pred=pred_v[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2],
                    interpolated_ref=self.ref_v_list,
                    anchor=(anchor[0]//2, anchor[1]//2),
                    mv=(pu['v_mv'][0]//2*fct, pu['v_mv'][1]//2*fct),
                    ref_idx=pu['ref_idx'],
                    pad_size=self.pad_size
                )
                # mask_y[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width] = 1
                # mask_uv[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2] = 1
        return pred_y, pred_u, pred_v


class MotionCompensationEngine(object):
    def __init__(self, per_frame_cu_list, yuv: YUVFrames) -> None:
        self.per_frame_cu_list = per_frame_cu_list
        self.yuv = yuv
        self.height = yuv.height
        self.width = yuv.width

    def populate_ref_list(self, crt_frame_idx):
        """Populate reference list for current frame"""
        start = max(0, crt_frame_idx - 4)
        end = crt_frame_idx
        n_frames = end - start
        ref_y_list = []
        ref_u_list = []
        ref_v_list = []
        for i in range(n_frames):
            y, u, v = self.yuv[end - 1 - i]
            ref_y_list.append(y)
            ref_u_list.append(u)
            ref_v_list.append(v)
        return ref_y_list, ref_u_list, ref_v_list
    
    def make_optical_flow_from_mv(self, cu_list):
        """Make optical flow from CU list"""
        flow = np.zeros((height, width, 2), dtype=np.float32)
        for cu in cu_list:
            for puid in range(len(cu['pu_list'])):
                # 2N x 2N
                pu = cu['pu_list'][puid]
                anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
                anchor = (anchor[0] * 2, anchor[1] * 2)
                pu_width = pu['width']
                pu_height = pu['height']
                pu_width *= 2
                pu_height *= 2
            
                # Calculate absolute resample coordinate
                abs_resample_x = np.arange(
                    anchor[0], anchor[0] + pu_width, dtype=np.float32)
                abs_resample_y = np.arange(
                    anchor[1], anchor[1] + pu_height, dtype=np.float32)
                # Add rescaled motion vector
                abs_resample_x += pu['y_mv'][0] / 2
                abs_resample_y += pu['y_mv'][1] / 2
                grid = np.meshgrid(abs_resample_x, abs_resample_y)
                flow[
                    anchor[1]:anchor[1]+pu_height,
                    anchor[0]:anchor[0]+pu_width, 0] = grid[0]
                flow[
                    anchor[1]:anchor[1]+pu_height,
                    anchor[0]:anchor[0]+pu_width, 1] = grid[1]
        # Normalize flow to (-1, 1)
        flow[:, :, 0] /= width
        flow[:, :, 1] /= height
        flow = flow * 2 - 1
        # Downsample flow to half
        down_flow = cv2.resize(
            flow, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        return flow, down_flow

    def motion_compensate(
            self, crt_frame_idx,
            default_pred_y=None,
            default_pred_u=None,
            default_pred_v=None):
        """Motion compensation for current frame"""
        crt_frame_cu_list = self.per_frame_cu_list[crt_frame_idx]
        ref_y_list, ref_u_list, ref_v_list = self.populate_ref_list(
            crt_frame_idx)
        if default_pred_y is None:
            assert default_pred_u is None
            assert default_pred_v is None
            pred_y = np.zeros((height, width), dtype=np.uint8)
            pred_u = np.zeros((height // 2, width // 2), dtype=np.uint8)
            pred_v = np.zeros((height // 2, width // 2), dtype=np.uint8)
        else:
            pred_y = default_pred_y
            pred_u = default_pred_u
            pred_v = default_pred_v
        flow, down_flow = self.make_optical_flow_from_mv(crt_frame_cu_list)
        for cu in crt_frame_cu_list:
            for puid in range(len(cu['pu_list'])):
                pu = cu['pu_list'][puid]
                anchor = (cu['x'] + pu['x'], cu['y'] + pu['y'])
                anchor = (anchor[0] * 2, anchor[1] * 2)
                pu_width = pu['width']
                pu_height = pu['height']
                pu_width *= 2
                pu_height *= 2

                # Warp
                ref_idx = pu['ref_idx']
                warped_y_block = torch.nn.functional.grid_sample(
                    torch.from_numpy(ref_y_list[ref_idx]).unsqueeze(0).unsqueeze(0).float(),
                    torch.from_numpy(flow[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width]).unsqueeze(0).float(),
                    mode='bilinear', padding_mode='border', align_corners=False)
                warped_u_block = torch.nn.functional.grid_sample(
                    torch.from_numpy(ref_u_list[ref_idx]).unsqueeze(0).unsqueeze(0).float(),
                    torch.from_numpy(down_flow[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2]).unsqueeze(0).float(),
                    mode='bilinear', padding_mode='border', align_corners=False)
                warped_v_block = torch.nn.functional.grid_sample(
                    torch.from_numpy(ref_v_list[ref_idx]).unsqueeze(0).unsqueeze(0).float(),
                    torch.from_numpy(down_flow[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2]).unsqueeze(0).float(),
                    mode='bilinear', padding_mode='border', align_corners=False)
                warped_y_block = warped_y_block.squeeze().numpy()
                warped_u_block = warped_u_block.squeeze().numpy()
                warped_v_block = warped_v_block.squeeze().numpy()

                # Paste
                pred_y[anchor[1]:anchor[1]+pu_height, anchor[0]:anchor[0]+pu_width] = warped_y_block
                pred_u[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2] = warped_u_block
                pred_v[anchor[1]//2:anchor[1]//2+pu_height//2, anchor[0]//2:anchor[0]//2+pu_width//2] = warped_v_block
        return pred_y, pred_u, pred_v
        


if __name__ == '__main__':

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


    # y, u, v = read_yuv_file(args.input_yuv, frame_size, height, width)
    # gt_y, gt_u, gt_v = read_yuv_file(args.gt_yuv, frame_size, height, width)
    yuv = YUVFrames(args.input_yuv, height, width)
    gt_yuv = YUVFrames(args.gt_yuv, height, width)

    mc_engine = SubPixMotionCompensationEngine(per_frame_cu_list, yuv)

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
        ref_y, ref_u, ref_v = mc_engine.populate_ref_list(
            current_frame_idx)
        prev_y, prev_u, prev_v = gt_yuv[current_frame_idx - 1]
        default_y = prev_y.copy()
        default_u = prev_u.copy()
        default_v = prev_v.copy()
        pred_y, pred_u, pred_v = mc_engine.motion_compensate(
            current_frame_idx,
            default_pred_y=default_y,
            default_pred_u=default_u,
            default_pred_v=default_v,)
        crt_y, crt_u, crt_v = gt_yuv[current_frame_idx]
        # calculate mse
        mask_y = np.ones_like(crt_y)
        mask_uv = np.ones_like(crt_u)
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


