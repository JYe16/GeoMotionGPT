import sys
project_root = '../../'
sys.path.append(project_root)
from utils.llm.LlavaNextVideo.modeling_llava_next_video_modified import LlavaNextVideoForConditionalGeneration
import torch
import torch.nn as nn
import os

class VideoProcessorWithClassifier(nn.Module):
    def __init__(self, base_model: LlavaNextVideoForConditionalGeneration, num_classes: int, model_dir):
        super().__init__()
        self.base = base_model

        # 2) Classification head: Two-layer MLP
        feat_dim = self.base.config.text_config.hidden_size
        classifier = nn.Sequential(
            nn.Linear(feat_dim, num_classes)
        )

        if os.path.isfile(os.path.join(model_dir, 'cls_project.pth')):
            cls_project = torch.load(os.path.join(model_dir, 'cls_project.pth'), weights_only=False)
        else:
            print("CLS Project not found. New created.")
            cls_project = nn.Linear(
                self.base.config.vision_config.hidden_size,  # 1024
                self.base.config.text_config.hidden_size,  # 4096
                bias=False
            )

        # **CRITICAL**: Cast classification head to same dtype & device as base_model
        #    Get dtype of base_model params (e.g., if it uses bfloat16 on GPU, get bfloat16 here)
        target_dtype = next(self.base.parameters()).dtype
        target_device = next(self.base.parameters()).device
        self.classifier = classifier.to(device=target_device, dtype=target_dtype)
        self.cls_project = cls_project.to(device=target_device, dtype=target_dtype)

    def forward(self, videos: torch.FloatTensor):
        B, F, C, H, W = videos.shape

        # Extract video features, list of patch features per frame
        _, cls_token = self.base.get_video_features(
            videos,
            self.base.config.vision_feature_layer,
            self.base.config.vision_feature_select_strategy
        )

        # Restore frame dimension and average over time
        D = cls_token.size(-1)  # 1024
        cls_token = cls_token.view(B, F, D).mean(dim=1)  # [B, D]
        cls_token = self.cls_project(cls_token)
        logits = self.classifier(cls_token)
        return logits