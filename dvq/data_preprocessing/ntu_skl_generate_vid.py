import os
import shutil
import cv2
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')     # 防止多进程下找不到 X11
from tqdm.contrib.concurrent import process_map   # NEW

opt_path = '/data/jackieye/ntu/vid_2/'

# ============================================================
# NTU-RGB+D Skeleton 关节索引说明
# ------------------------------------------------------------
# index_0  index_1  英文名             中文注释
# ------------------------------------------------------------
#    0        1     SpineBase          脊柱底端（骨盆中央）
#    1        2     SpineMid           脊柱中段
#    2        3     Neck               颈部
#    3        4     Head               头部
#    4        5     ShoulderLeft       左肩
#    5        6     ElbowLeft          左肘
#    6        7     WristLeft          左腕
#    7        8     HandLeft           左手掌
#    8        9     ShoulderRight      右肩
#    9       10     ElbowRight         右肘
#   10       11     WristRight         右腕
#   11       12     HandRight          右手掌
#   12       13     HipLeft            左髋
#   13       14     KneeLeft           左膝
#   14       15     AnkleLeft          左踝
#   15       16     FootLeft           左脚
#   16       17     HipRight           右髋
#   17       18     KneeRight          右膝
#   18       19     AnkleRight         右踝
#   19       20     FootRight          右脚
#   20       21     SpineShoulder      上胸椎（肩胛骨间）
#   21       22     HandTipLeft        左手指尖
#   22       23     ThumbLeft          左拇指尖
#   23       24     HandTipRight       右手指尖
#   24       25     ThumbRight         右拇指尖
# ============================================================

# ================== 关节分组（0-based 索引）==================
JOINT_GROUPS = {
    "torso":       [0, 1, 2, 3, 20],                    # 躯干
    "left_arm":    [4, 5, 6, 7, 21, 22],                # 左臂＋手
    "right_arm":   [8, 9, 10, 11, 23, 24],              # 右臂＋手
    "left_leg":    [12, 13, 14, 15],                    # 左腿
    "right_leg":   [16, 17, 18, 19],                    # 右腿
}

# ================== 每个部位对应的颜色 ==================
COLOR_MAP = {
    "torso":      "#FDB813",   # 金黄
    "left_arm":   "#1F77B4",   # 蓝
    "right_arm":  "#D62728",   # 红
    "left_leg":   "#2CA02C",   # 绿
    "right_leg":  "#9467BD",   # 紫
}

# 反向映射：关节索引 → 颜色（便于快速查询）
JOINT_COLOR = {}
for part, joints in JOINT_GROUPS.items():
    for j in joints:
        JOINT_COLOR[j] = COLOR_MAP[part]

def visualize_graph(seq, rotate, output_path, file):
    adj_list = [[1, 2], [2, 21], [3, 21], [4, 3], [5, 21], [6, 5], [7, 6],
                [8, 7], [9, 21], [10, 9], [11, 10], [12, 11], [13, 1],
                [14, 13], [15, 14], [16, 15], [17, 1], [18, 17], [19, 18],
                [20, 19], [22, 23], [23, 8], [24, 25], [25, 12]]
    output_path = output_path + "/" + file + "/"
    if os.path.exists(output_path) is True:
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    num_ppl = len(seq[0]) // 25

    for i in range(len(seq)):
        x = []
        y = []
        z = []
        ex = []
        ey = []
        ez = []
        for point in seq[i]:
            x.append(point[2])
            y.append(point[0])
            z.append(point[1])

        for j in range(0, len(adj_list), 1):
            for k in range(num_ppl):
                ex.append([seq[i][adj_list[j][0] + k * 25 - 1][2], seq[i][adj_list[j][1] + k * 25 - 1][2]])
                ey.append([seq[i][adj_list[j][0] + k * 25 - 1][0], seq[i][adj_list[j][1] + k * 25 - 1][0]])
                ez.append([seq[i][adj_list[j][0] + k * 25 - 1][1], seq[i][adj_list[j][1] + k * 25 - 1][1]])

        plt.figure(figsize=(3.6, 3.6))
        ax = plt.axes(projection='3d')
        if rotate:
            ax.view_init(20, 2 * i)
        else:
            ax.view_init(10, 90)
        # ---------- 画散点：按关节各自颜色 ----------
        for idx in range(len(seq[i])):  # idx = 0~24*(人数)
            part_color = JOINT_COLOR[idx % 25]  # 多人序列也能循环使用同一套颜色
            ax.scatter3D([seq[i][idx][2]],
                         [seq[i][idx][0]],
                         [seq[i][idx][1]],
                         color=part_color,
                         s=15)  # 点大小可自行调

        # ---------- 画骨架连线：用起点的颜色 ----------
        for edge in adj_list:  # edge 类似 [1, 2]（1-based）
            for k in range(num_ppl):
                a = edge[0] - 1 + k * 25  # 转 0-based + 人体偏移量
                b = edge[1] - 1 + k * 25
                color = JOINT_COLOR[a % 25]  # 取起点所属部位颜色
                ax.plot([seq[i][a][2], seq[i][b][2]],
                        [seq[i][a][0], seq[i][b][0]],
                        [seq[i][a][1], seq[i][b][1]],
                        color=color,
                        linewidth=2)

        filename = str(i)
        while len(filename) != 4:
            filename = "0" + filename
        filename = file + "_" + filename
        # Remove grid
        ax.grid(False)

        # Remove tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Hide axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        # Hide spines (bounding box)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Remove axis lines
        ax.xaxis.line.set_color((1, 1, 1, 0))  # Fully transparent
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))
        plt.savefig(f"{output_path}/{filename}.jpg")
        plt.close('all')
    generate_video(output_path, f"{file}.avi", opt_path)
    shutil.rmtree(output_path)


def generate_video(image_folder, video_name, output_path):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(str(os.path.join(output_path, video_name)), 0, 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def sample_frame(seq, num_frame):
    num_joint = len(seq[0])
    # using scipy to sample the number of frames
    original_frame = len(seq)
    original_frames = np.linspace(0, original_frame - 1, original_frame)  # [0, 1, ..., 70]
    target_frames = np.linspace(0, original_frame - 1, num_frame)  # Resampled to 32 frames
    # Initialize interpolated array
    interpolated_sequence = np.empty((num_frame, len(seq[0]), 3), dtype=np.float32)

    # Apply interpolation for each joint and coordinate
    for joint in range(num_joint):
        for coord in range(3):
            interp_func = interp1d(original_frames, seq[:, joint, coord], kind='linear', axis=0)
            interpolated_sequence[:, joint, coord] = interp_func(target_frames)
    return interpolated_sequence

def load_and_generate_vid(filename):
    path = '/data/jackieye/ntu/skl_raw_all/NTU60_skl/'

    sequence = read_skeleton(os.path.join(path, filename + '.skeleton'))
    # sequence = sample_frame(sequence, 8)
    visualize_graph(seq=sequence, rotate=False, output_path='/data/jackieye/Vis/t/', file=filename)
    # print(f"File {filename} has been generated.")
    return 1

def read_skeleton(file_path):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])
    persons = []
    for i in range(0, max_body):
        persons.append(np.zeros(shape=(nframe, njoints, 3)))
    body_counts = []
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        body_counts.append(bodycount)
        if bodycount == 0:
            continue
            # skip the empty frame
        for body in range(bodycount):
            cursor += 1
            person = persons[body]
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                person[frame, joint] = jointinfo[:3]
    # prune the abundant bodys
    result = np.empty((nframe, 0, 3))
    num_ppl = max(body_counts)
    for idx in range(num_ppl):
        result = np.concatenate((result, np.asarray(persons[idx])), axis=1)
    # if num_ppl > 2:
    #     print(f"{file_path}: {num_ppl}")
    return result

# def get_num_bodies()

if __name__ == '__main__':
    path = '/data/jackieye/ntu/skl_raw_all/NTU60_skl/'
    # ① 先收集待处理文件列表
    with open('/data/jackieye/ntu/skl_raw_all/NTU_RGBD_samples_with_missing_skeletons.txt') as f:
        missing_files = set(line.strip() for line in f)

    files = [f[:-9]  # 去掉 "_skeleton"
             for f in os.listdir(path)
             if f.endswith('.skeleton')
             and f[:-9] + '.avi' not in os.listdir(opt_path)
             and f[:-9] not in missing_files]

    print(f'{len(files)} files to be processed…')

    # ② 一行代码，多进程并带统一进度条
    #    max_workers = 24 等价于你的 Pool(24)
    process_map(load_and_generate_vid, files,
                max_workers=24, chunksize=1)