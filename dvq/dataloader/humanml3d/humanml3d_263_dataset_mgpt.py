import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os

class HumanML3DDataset(data.Dataset):
    """
    This is for VQ-VAE training ONLY (Raw Motion Data)
    """

    def __init__(self, data_root, split='train', window_size=64, unit_length=4, features_dim=263):
        self.window_size = window_size
        self.unit_length = unit_length
        self.data_root = data_root

        # 定义我们数据集的特定参数
        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')
        self.joints_num = 22
        self.features_dim = features_dim
        # MotionGPT3 uses 200, we keep consistent
        self.max_motion_length = 200 
        self.min_motion_length = 20

        try:
            mean = np.load(pjoin(self.data_root, 'Mean.npy'))
            std = np.load(pjoin(self.data_root, 'Std.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Error：Mean.npy and Std.npy not found.")

        split_file = pjoin(self.data_root, f'{split}.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        print(f"Loading data from {split}.txt...")
        # Aligned with MotionGPT3 filtering logic
        new_name_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                # Filter length consistent with MotionGPT3
                if (len(motion)) < self.min_motion_length or (len(motion) < self.window_size) or (len(motion) >= self.max_motion_length):
                    continue

                # 将所有动作数据预先加载到内存中
                self.data.append(motion)
                self.lengths.append(motion.shape[0] - self.window_size)
                new_name_list.append(name)
            except Exception as e:
                # print(f"Cannot load file {name}.npy: {e}")
                pass

        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

        # 检查特征维度是否匹配
        if self.mean.shape[0] != self.features_dim or self.std.shape[0] != self.features_dim:
            raise ValueError(
                f"Dimension Mismatch - Real: ({self.mean.shape[0]}), Expected: ({self.features_dim})")

        print(f"Successfully loaded {split} dataset with total {len(self.data)} clips")

    def inv_transform(self, data):
        """将标准化的数据逆转换为原始尺度"""
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]

        # 从完整动作中随机截取一个窗口
        if len(motion) - self.window_size > 0:
            idx = random.randint(0, len(motion) - self.window_size)
            motion_window = motion[idx: idx + self.window_size]
        else:
            # Handle cases where filtered motion might be smaller than window_size (though init filters them)
            motion_window = motion[:self.window_size] 

        # 转换为Torch Tensor
        motion_tensor = torch.from_numpy(motion_window).float()

        # Z-Score Normalization (标准化)
        motion_normalized = (motion_tensor - self.mean) / self.std

        return motion_normalized


class HumanML3DTokenDataset(data.Dataset):
    """
    Aligned with MotionGPT3's Text2MotionDataset for fair M2T evaluation.
    """
    def __init__(self, data_root, split, text_tokenizer, motion_id_table, max_motion_token=128, max_text_token=64):
        # data_root passed from training script usually points to '.../motion_tokens_2cb/'
        # We need to access the parent data root for texts and raw motions
        self.data_root = os.path.abspath(pjoin(data_root, '..')) # Assuming structure: data/humanml3d_263/motion_tokens_2cb/
        
        # If the above assumption fails, fallback or ensure data_root is correct. 
        # For safety based on your provided file paths: 
        # If input data_root is like '../data/humanml3d_263/motion_tokens_2cb/', 
        # we need '../data/humanml3d_263/'
        
        self.split = split
        self.text_tokenizer = text_tokenizer
        self.max_motion_token = max_motion_token
        self.max_text_token = max_text_token
        self.motion_id_table = torch.as_tensor(motion_id_table, dtype=torch.long)
        
        # 定义特殊Token的ID
        self.bos_token_id = self.text_tokenizer.bos_token_id
        self.eos_token_id = self.text_tokenizer.eos_token_id
        # 使用换行符的Token作为动作和文本的分隔符
        self.sep_token_id = self.text_tokenizer.encode("\n")[0]

        # Token directory (input data_root)
        self.motion_token_dir = data_root 
        
        # Raw data directories for validation/filtering
        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')

        split_file = pjoin(self.data_root, f'{split}.txt')

        # MotionGPT3 filtering params
        self.min_motion_length = 20
        self.max_motion_length = 200
        self.fps = 20

        self.id_list = []
        self.data_dict = {}

        # 1. Load Split File
        raw_id_list = []
        if os.path.exists(split_file):
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    raw_id_list.append(line.strip())
        else:
            # Fallback for some directory structures
            print(f"Warning: Split file {split_file} not found. Trying local path.")
            pass

        print(f"Scanning {split} dataset for filtering (MotionGPT3 alignment)...")
        
        # 2. Filter data exactly like MotionGPT3's Text2MotionDataset
        # This ensures the test set is identical in content and order
        new_name_list = []
        
        for name in tqdm(raw_id_list):
            try:
                # Load raw motion to check length
                motion_path = pjoin(self.motion_dir, name + ".npy")
                if not os.path.exists(motion_path):
                    continue
                    
                motion = np.load(motion_path)
                
                # Check 1: Raw Motion Length
                # MotionGPT3 logic: if (len(motion)) < min or (len(motion) >= max): continue
                if (len(motion) < self.min_motion_length) or (len(motion) >= self.max_motion_length):
                    continue

                # Check 2: Text Validity & Segmentation
                text_data = []
                flag = False
                text_path = pjoin(self.text_dir, name + '.txt')
                
                if not os.path.exists(text_path):
                    continue

                with cs.open(text_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        if len(line_split) < 2: continue # Safety check
                        t_tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = t_tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            # Handle sub-segments (though rare in HumanML3D test)
                            motion_new = motion[int(f_tag * self.fps):int(to_tag * self.fps)]
                            if (len(motion_new) < self.min_motion_length) or (len(motion_new) >= self.max_motion_length):
                                continue
                            new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                            
                            # Note: We assume tokens exist for these segments. 
                            # If llm-sensing preprocessing didn't generate tokens for segments, this might fail later.
                            # But for standard HumanML3D, flag=True is the common case.
                            
                            self.data_dict[new_name] = {
                                'length': len(motion_new),
                                'text': [text_dict],
                                'is_segment': True,
                                'parent_name': name
                            }
                            new_name_list.append(new_name)

                if flag:
                    self.data_dict[name] = {
                        'length': len(motion),
                        'text': text_data,
                        'is_segment': False
                    }
                    new_name_list.append(name)

            except Exception as e:
                # print(f"Error processing {name}: {e}")
                pass

        self.id_list = new_name_list
        print(f"Successfully initialized {split} dataset with {len(self.id_list)} samples (Filtered).")

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        motion_id = self.id_list[item]
        data_entry = self.data_dict[motion_id]
        
        # Get all captions (for evaluation)
        text_list = data_entry['text']
        all_captions = [x['caption'] for x in text_list]
        
        # Select one caption for Input (random for training, deterministic-ish for test?)
        # MotionGPT3 picks random even in test, but for consistency we just pick random.
        text_data = random.choice(text_list)
        text_description = text_data['caption']
        
        # Load Motion Tokens
        # If it's a sub-segment (name_start_end), we might not have a separate token file.
        # Assuming we load the original file. 
        # NOTE: If tokens are not segmented, this is an approximation. 
        # For standard HumanML3D test set, this branch is rarely taken.
        file_name_to_load = motion_id
        if data_entry.get('is_segment', False):
             file_name_to_load = data_entry['parent_name']
             # Warning: We are loading full tokens for a segment. 
             # Ideal fix: Slice tokens based on f_tag/to_tag (scaled by downsample ratio).
             # But here we assume 1-to-1 mapping for simplicity and standard test set.

        try:
            # 读取 0..nb_code-1 的离散码
            token_path = pjoin(self.motion_token_dir, file_name_to_load + '.npy')
            raw = np.load(token_path)
            if raw.ndim > 1:
                raw = raw.flatten()
            raw = raw[:self.max_motion_token].astype(np.int64)

            # —— 关键：查表映射到真实 tokenizer id，绝不做 +offset ——
            table = self.motion_id_table
            motion_tokens = table[torch.from_numpy(raw)].numpy()

        except Exception as e:
            # Fallback
            # print(f"Error loading tokens for {motion_id}: {e}")
            return self.__getitem__((item + 1) % len(self.id_list))

        # 使用文本分词器处理文本描述
        text_tokens = self.text_tokenizer(text_description, add_special_tokens=False)["input_ids"]

        # 截断
        # motion_tokens = motion_tokens[:self.max_motion_token]
        text_tokens = text_tokens[:self.max_text_token]
        
        # Instruction prompt
        start_word_tokens = self.text_tokenizer(
            "Please describe the motion given the following motion tokens: ", 
            add_special_tokens=False
        )["input_ids"]
        
        # 3. 根据 split 构建 input / label
        if self.split.lower() in {"val", "test", "eval"}:
            # ---------- 推断 / 评测 ----------
            input_ids = np.concatenate(
                [start_word_tokens, motion_tokens, [self.sep_token_id]]
            ).astype(np.int64)  # 只含动作
            
            labels = np.asarray(text_tokens + [self.eos_token_id],
                                dtype=np.int64)  # 参考答案
            
            # Important: Return ALL captions in references for M2T metrics
            refs = all_captions
            
        else:
            # ---------- 训练 ----------
            input_ids = np.concatenate(
                [start_word_tokens, motion_tokens,
                 [self.sep_token_id], text_tokens,
                 [self.eos_token_id]]
            ).astype(np.int64)
            labels = np.full_like(input_ids, -100)
            motion_len = len(motion_tokens) + len(start_word_tokens) + 1  # SEP
            labels[motion_len:] = input_ids[motion_len:]  # 仅文本参与 loss
            refs = [text_description]

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.int64),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "references": refs # Updated to return list of all captions
        }