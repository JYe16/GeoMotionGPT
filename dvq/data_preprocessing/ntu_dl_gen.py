import sys
project_root = '../'
sys.path.append(project_root)
import argparse
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from utils.process_video import process_video_llava
from multiprocessing import Pool


training_subjects_60 = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras_60 = [2, 3]
training_subjects_120 = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]
training_setups_120 = [
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
]


def gendata(data_path, out_path, benchmark):
    multiprocess_args = []

    for filename in os.listdir(data_path):
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xsub':
            istraining = (subject_id in training_subjects_60)
        elif benchmark == 'xview':
            istraining = (camera_id in training_cameras_60)
        else:
            raise ValueError()

        save_path = out_path + '/train/' if istraining else out_path + '/test/'
        save_path = save_path + filename[:-4] + ".npy"

        multiprocess_args.append([filename, data_path, save_path])
    with Pool(16) as p:
        for _ in tqdm(p.imap_unordered(process_and_save, multiprocess_args), total=len(multiprocess_args)):
            pass


def process_and_save(args):
    filename, data_path, save_path = args
    # 自动创建目标文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video_data = process_video_llava(os.path.join(data_path, filename))
    np.save(save_path, video_data)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../data/ntu/skl_vid/')
    parser.add_argument('--out-folder', type=str, default='../data/ntu/skl_vid_dl_full/')
    args = parser.parse_args()

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']

    for b in benchmark:
        out_path = osp.join(args.out_folder, b)
        if not osp.exists(out_path):
            os.makedirs(out_path)
        gendata(data_path=args.data_path, out_path=out_path, benchmark=b)
