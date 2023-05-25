import os
import sys
import argparse
import subprocess as sp
import multiprocessing as mp

# Encode and decode a sequence
# Encode using HM
# Decode using HM that dumps CU info to stderr

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input sequence')
parser.add_argument('--qp_list', type=str, help='QP list')
parser.add_argument('--width', type=int, help='Width')
parser.add_argument('--height', type=int, help='Height')
parser.add_argument('--num_frames', type=int, help='Number of frames')
parser.add_argument('--decode', action='store_true', help='Decode')

args = parser.parse_args()

input_fn = args.input
qp_list = args.qp_list.split(',')
width = args.width
height = args.height
num_frames = args.num_frames

def run_encode(width, height, fn, qp):
    file_path = '/'.join(fn.split('/')[:-1])
    fid = fn.split('/')[-1][:-4]
    cmd = [
        '../bin/TAppEncoderStatic',
        '-c', '../cfg/encoder_lowdelay_P_1234_main.cfg',
        '-c', './Common_120F.cfg',
        f'--SourceWidth={args.width}',
        f'--SourceHeight={args.height}',
        f'--InputFile={args.input}',
        f'--QP={qp}',
        f'--ReconFile={file_path}/{fid}_qp{qp}.yuv',
        f'--BitstreamFile={file_path}/{fid}_qp{qp}.265',
        f'--FramesToBeEncoded={args.num_frames}',
    ]
    print(' '.join(cmd))
    c = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    ret = c.wait()
    stdout, stderr = c.communicate()
    with open(f'{file_path}/{fid}_qp{qp}.log', 'w') as f:
        f.write(stdout)
    if ret != 0:
        print(stderr.decode())
        sys.exit(1)

# Encode
# Create pool
pool = mp.Pool(mp.cpu_count())
for qp in qp_list:
    pool.apply_async(run_encode, args=(width, height, input_fn, qp))
pool.close()
pool.join()

if args.decode:
    # Decode
    raise NotImplementedError()
    for qp in qp_list:
        file_path = '/'.join(input_fn.split('/')[:-1])
        fid = input_fn.split('/')[-1][:-4]
        cmd = [
            './TAppDecoderStaticPrintPT',
            '-b', f'{fid}_qp{qp}.265',
            '-o', f'{fid}_qp{qp}.yuv',
        ]
        print(' '.join(cmd))
        c = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        ret = c.wait()
        stdout, stderr = c.communicate()
        if ret != 0:
            print(stderr.decode())
            sys.exit(1)


