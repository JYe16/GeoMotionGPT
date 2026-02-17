import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

class Motion2TextDatasetTokenCustom(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        code_path="motion_tokens", # Default token folder name
        w_vectorizer=None,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.mean = mean
        self.std = std
        self.nfeats = int(self.mean.shape[0])
        self.split = split
        
        # Path settings
        split_file = pjoin(data_root, split + '.txt')
        self.text_dir = pjoin(data_root, 'texts')
        self.motion_dir = pjoin(data_root, 'new_joint_vecs')
        
        # Process Token path
        # If code_path is passed in config, use it; otherwise default to 'motion_tokens'
        actual_code_path = kwargs.get('code_path', code_path)
        if actual_code_path is None:
            actual_code_path = "motion_tokens"
            
        if os.path.isabs(actual_code_path):
            self.motion_token_dir = actual_code_path
        else:
            self.motion_token_dir = pjoin(data_root, actual_code_path)

        print(f"[CustomDataset] Loading tokens from: {self.motion_token_dir}")

        # Load Split file
        self.id_list = []
        if os.path.exists(split_file):
            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    self.id_list.append(line.strip())
        else:
            print(f"[Error] Split file not found: {split_file}")

        # Data filtering and loading logic
        new_name_list = []
        data_dict = {}
        
        print(f"Loading {split} dataset...")
        for name in tqdm(self.id_list):
            try:
                # 1. Check Token file
                token_file = pjoin(self.motion_token_dir, name + '.npy')
                if not os.path.exists(token_file):
                    continue
                
                # 2. Load tokens
                tokens = np.load(token_file)
                if len(tokens.shape) > 1:
                    tokens = tokens.flatten()
                
                # 3. Filter based on MOTION frame count (not token length!)
                # Need to load motion file to get actual frame count
                motion_file = pjoin(self.motion_dir, name + '.npy')
                if not os.path.exists(motion_file):
                    continue
                motion = np.load(motion_file)
                motion_length = len(motion)
                
                # Standard filtering: min=40, max<200 for HumanML3D evaluation
                if motion_length < self.min_motion_length or motion_length >= self.max_motion_length:
                    continue

                # 4. Load text
                text_path = pjoin(self.text_dir, name + '.txt')
                if not os.path.exists(text_path):
                    continue
                    
                text_data = []
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        if len(line_split) < 2: continue
                        tokens_text = line_split[1].split(' ')
                        
                        text_dict = {
                            'caption': caption,
                            'tokens': tokens_text
                        }
                        text_data.append(text_dict)

                if len(text_data) > 0:
                    data_dict[name] = {
                        'token_path': token_file,
                        'motion_path': motion_file,
                        'motion_length': motion_length,
                        'text': text_data
                    }
                    new_name_list.append(name)
            except Exception as e:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"[CustomDataset] Loaded {len(self.name_list)} samples for {split}")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        fname = self.name_list[item]
        data = self.data_dict[fname]
        
        # 1. 加载 Motion Tokens
        m_tokens = np.load(data['token_path'])
        if len(m_tokens.shape) > 1:
            m_tokens = m_tokens.flatten()
        
        # 截断处理
        m_tokens_len = len(m_tokens)
        if m_tokens_len > self.max_motion_length:
            m_tokens = m_tokens[:self.max_motion_length]
            m_tokens_len = self.max_motion_length

        # 2. 加载 Text
        text_list = data['text']
        
        # 获取所有 caption (用于评估) - 与 eval_v3 相同的处理逻辑
        all_captions = [text_dic['caption'] for text_dic in text_list]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # 随机选择一条文本
        text_data = random.choice(text_list)
        caption = text_data['caption']
        tokens = text_data['tokens']
        
        # Text processing - 与 eval_v3 完全相同
        max_text_len = 20
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # 3. 加载 Motion (用于评估)
        motion_file = pjoin(self.motion_dir, fname + '.npy')
        if os.path.exists(motion_file):
            motion = np.load(motion_file)
            
            # 对于 M2T 任务，motion 应该和 token 对应
            # Token 是从完整 motion 编码的，所以这里不做 random crop
            # 只做长度对齐和归一化
            m_length = motion.shape[0]
            
            # 对齐到 unit_length 的倍数
            m_length = (m_length // self.unit_length) * self.unit_length
            motion = motion[:m_length]

            # Z Normalization
            motion = (motion - self.mean) / self.std
        else:
            # Fallback
            motion = np.zeros((m_tokens_len * 4, 263))
            m_length = m_tokens_len * 4

        # 返回元组顺序与 eval_v3 完全相同:
        # caption, m_tokens, m_tokens_len, motion, m_length, word_embs, pos_ohot, text_len, tokens, all_captions, tasks, fname
        # 注意: 这里返回 m_tokens 和 m_tokens_len 供 M2T 任务使用
        
        # M2T 任务定义
        task = {
            "class": "m2t",
            "input": ["Generate text: <Motion_Placeholder>"],
            "output": ["<Caption_Placeholder>"]
        }
        
        return caption, m_tokens, m_tokens_len, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(
            tokens), all_captions, task, fname