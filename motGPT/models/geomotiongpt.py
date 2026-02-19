"""
MotGPT model variant with orthogonality regularization for motion token embeddings.

This module extends the base MotGPT model to include orthogonality loss
that encourages motion token embeddings to be mutually orthogonal.
"""

import numpy as np
import os
import random
import torch
import time
from motGPT.config import instantiate_from_config
from os.path import join as pjoin
from motGPT.losses.motgpt import MotLosses
from motGPT.losses.orthogonal import (
    MotionOrthogonalityLoss,
    get_embedding_weight_from_model
)
from motGPT.losses.pairwise_cosine_distance import MotionPairwiseCosineDistanceLoss
from motGPT.models.base import BaseModel
import json
from motGPT.utils.render_utils import render_motion


def sig(x):
    s = 1./(1+np.exp(-x))
    return s


class GeoMotionGPT(BaseModel):
    """
    MotGPT with Orthogonality Regularization for Motion Token Embeddings.
    
    This variant adds an orthogonality loss term to encourage motion token
    embeddings in the LLM's embedding space to be mutually orthogonal.
    
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrain
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 fps=20,
                 guidance_scale=1.0,
                 # Geometry regularization parameters
                 lambda_geo: float = 0.1,
                 geo_loss_type: str = 'ortho',
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        self.njoints = self.datamodule.njoints
        self.fps = self.datamodule.fps
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            motion_vae['params']['datatype'] = self.datamodule.name
            self.vae = instantiate_from_config(motion_vae)

        self.vae_latent_channels = self.vae.latent_dim

        # Instantiate motion-language model
        lm['params']['vae_latent_channels'] = self.vae_latent_channels
        lm['params']['vae_latent_size'] = self.vae.latent_size if hasattr(
            self.vae,'latent_size') else None
        self.lm = instantiate_from_config(lm)

        # Freeze the motion tokenizer for lm training
        if 'adaptor' in self.hparams.stage:
            self.vae.training = False
            self.lm.language_model.eval()
            self.lm.language_model.training = False
            self.lm.tokenizer.training = False
            
            for p in self.vae.parameters():
                p.requires_grad = False
            for p in self.lm.language_model.parameters():
                p.requires_grad = False
        elif 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False
        self.model_dir = cfg.FOLDER_EXP
        self.vis_num = 2

        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: MotLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # =====================================================
        # NEW: Geometry regularization for motion token embeddings
        # =====================================================
        legacy_lambda_ortho = kwargs.pop('lambda_ortho', None)
        if legacy_lambda_ortho is not None:
            lambda_geo = legacy_lambda_ortho

        self.lambda_geo = lambda_geo
        self.lambda_ortho = self.lambda_geo  # backward-compatible alias
        self.geo_loss_type = str(geo_loss_type).strip().lower()
        motion_codebook_size = lm.get('params', {}).get('motion_codebook_size', 512)
        if self.geo_loss_type == 'ortho':
            self.geo_loss = MotionOrthogonalityLoss(
                motion_codebook_size=motion_codebook_size,
                lambda_ortho=self.lambda_geo
            )
        elif self.geo_loss_type == 'pairwise_cosine':
            self.geo_loss = MotionPairwiseCosineDistanceLoss(
                motion_codebook_size=motion_codebook_size,
                lambda_pairwise=self.lambda_geo
            )
        else:
            raise ValueError(
                f"Unsupported geo_loss_type='{geo_loss_type}'. "
                "Expected one of ['ortho', 'pairwise_cosine']."
            )
        
        # Data transform
        self.feats2joints = datamodule.feats2joints

    def compute_geo_loss(self) -> torch.Tensor:
        """
        Compute the geometry regularization loss for motion token embeddings.
        
        Returns:
            Scalar tensor representing the geometry regularization loss.
        """
        try:
            embedding_weight = get_embedding_weight_from_model(self.lm.language_model)
            device = next(self.lm.language_model.parameters()).device
            loss = self.geo_loss(embedding_weight, self.lm.tokenizer, device)
            
            # Debug: Check for NaN
            if torch.isnan(loss):
                print(f"[DEBUG] Geometry regularization loss is NaN!")
                print(f"[DEBUG] Embedding weight shape: {embedding_weight.shape}")
                print(f"[DEBUG] Embedding weight dtype: {embedding_weight.dtype}")
                print(f"[DEBUG] Embedding weight has NaN: {torch.isnan(embedding_weight).any()}")
                print(f"[DEBUG] Embedding weight has Inf: {torch.isinf(embedding_weight).any()}")
                return torch.zeros((), device=device, dtype=embedding_weight.dtype)
            
            return loss
        except Exception as e:
            # Print the error for debugging
            print(f"[DEBUG] compute_geo_loss failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return zero loss if something goes wrong
            device = next(self.lm.language_model.parameters()).device
            return torch.zeros((), device=device)

    def compute_ortho_loss(self) -> torch.Tensor:
        """Backward-compatible wrapper for old call sites."""
        return self.compute_geo_loss()

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]
        if task in ['inbetween']:
            lengths = lengths_ref
        else:
            lengths = [random.randint(20,35)*4 for l in lengths_ref]
        motion_tokens_input = batch['motion_tokens_input']

        if task in ['t2m', 'pred', 'prediction', 'inbetween']:
            outputs = self.lm.generate_direct_motion(
                    texts,
                    motion_tokens=motion_tokens_input,
                    )
            sampled_token_latents, motion_mask = self.lm.sample_tokens(
                outputs, self.lm.device, 
                temperature=1.0, cfg=self.guidance_scale, 
                vae_mean_std_inv=self.vae.mean_std_inv)
            sampled_token_latents = sampled_token_latents.reshape(len(lengths), self.vae.latent_size, -1).permute(1,0,2)
            feats_rst = self.vae.decode(sampled_token_latents, lengths=lengths)

            joints_rst = self.feats2joints(feats_rst)
            feats_rst = self.datamodule.denormalize(feats_rst)
            gen_texts = ['<Motion_Placeholder>' for t in texts]
            outputs = {
                "texts": gen_texts,
                "feats": feats_rst,
                "joints": joints_rst,
                "length": lengths
            }
        elif task in ['m2t', 't2t']:
            outputs_tokens, cleaned_text = self.lm.generate_direct(
                texts,
                motion_tokens=motion_tokens_input,
                max_length=40,
                gen_mode='text',
                bad_words_ids=[[self.lm.som_id], [self.lm.eom_id]],
            )
            gen_texts = cleaned_text
            outputs = {
                "texts": gen_texts,
                "feats": [None]*len(gen_texts),
                "joints": [None]*len(gen_texts),
                "length": [None]*len(gen_texts),
            }
        else:
            assert False, f'{task} Not implemented yet'
            
        return outputs

    def train_lm_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]

        try:
            outputs = self.lm(texts, feats_ref, self.vae, lengths, tasks)
        except TypeError:
            if "m_tokens" in batch:
                motion_tokens = batch["m_tokens"].long()
                lm_lengths = batch.get("m_tokens_len", lengths)
                outputs = self.lm(texts, motion_tokens, lm_lengths, tasks)
            else:
                raise

        # Debug code commented out
        # if hasattr(outputs, 'loss'):
        #     if outputs.loss is not None:
        #         loss_val = outputs.loss
        #         if torch.isnan(loss_val):
        #             print(f"[DEBUG] NaN loss detected!")
        #         else:
        #             print(f"[DEBUG] outputs.loss = {loss_val.item():.4f}")

        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2t_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = [{
                'class': 't2t',
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)
        
        with torch.no_grad():
            outputs = self.lm.generate_conditional(texts,
                                                lengths=lengths,
                                                stage='test',
                                                task='t2t',
                                                tasks=tasks,
                                                )

        rs_set = {
            "m_ref": feats_ref,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set
    
    @torch.no_grad()
    def val_t2m_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            instructions = pjoin(self.datamodule.hparams.data_root,
                                 'template_t2m_instructions.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        with torch.no_grad():
            outputs = self.lm.generate_conditional(texts,
                                                lengths=lengths,
                                                stage='test',
                                                tasks=tasks,
                                                )
            
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch.get("tasks", None)
        all_captions = batch['all_captions']
        motion_tokens = batch['m_tokens']
        motion_tokens_len = batch.get('m_tokens_len', None)

        if tasks is None:
            tasks = [{
                    'class': 'm2t',
                    'input': ['<Motion_Placeholder>'],
                    'output': ['']
                }] * len(texts)

        with torch.no_grad():
            outputs = self.lm.generate_conditional(texts,
                                                motion_tokens=motion_tokens,
                                                lengths=motion_tokens_len if motion_tokens_len is not None else lengths,
                                                stage='test',
                                                task='m2t',
                                                tasks=tasks)
        
        # IMPORTANT: renorm4t2m converts from training Mean/Std to evaluator Mean/Std
        # This is required for correct R-precision computation with the T2M evaluator
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])
            lengths_tokens.append(motion_token.shape[1])

        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        
        motion_z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(motion_z, lengths)
        recons_z, _ = self.vae.encode(feats_rst, lengths)
        
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        
        if dist_m is not None:
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        feats_ref = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        motion_z, dist_m = self.vae.encode(feats_ref, lengths)
        feats_rst = self.vae.decode(motion_z, lengths)

        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "lat_m": motion_z.permute(1, 0, 2),
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        """
        Main training/validation step with orthogonality loss.
        """
        loss = None
        
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_finetune", "lm_adaptor_pretrain", "token_custom"
                                    ] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
            
            # =====================================================
            # NEW: Add geometry regularization during training
            # =====================================================
            if self.lambda_geo > 0:
                geo_loss = self.compute_geo_loss()
                loss = loss + geo_loss
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics (validation/test)
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_finetune", "lm_rl", "lm_adaptor_pretrain", "token_custom"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task == "t2t":
                    rs_set = self.val_t2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)

            if self.hparams.task not in ["m2t", 't2t']:
                if batch_idx == 0 and self.global_rank == 0:
                    lengths = batch['length']
                    feats_ref, joints_ref = rs_set['m_ref'], rs_set['joints_ref']
                    feats_rst, joints_rst = rs_set['m_rst'], rs_set['joints_rst']
                    rand_save_idx = random.sample(range(feats_ref.shape[0]), self.vis_num)
                    for idd in rand_save_idx:
                        idx = idd % len(lengths)
                        output_dir = os.path.join(self.model_dir, 'validate_motion', f'epoch_{self.current_epoch}')
                        os.makedirs(output_dir, exist_ok=True)
                        keyid = (batch['fname'][idx]).split('/')[-1]
                        motion = batch['motion'][idx]
                        joint_ref = self.feats2joints(motion)
                        feat_ref, joint_ref = feats_ref[idx][:lengths[idx]], joints_ref[idx][:lengths[idx]]
                        feat_rst, joint_rst = feats_rst[idx][:lengths[idx]], joints_rst[idx][:lengths[idx]]
                        render_motion(joint_ref, joint_ref.cpu().numpy(), output_dir=output_dir, fname=f'{keyid}_gt', method='fast', fps=self.fps)
                        render_motion(joint_rst, joint_rst.cpu().numpy(), output_dir=output_dir, fname=f'{keyid}', method='fast')
                        np.savetxt(os.path.join(output_dir, f'{keyid}.txt'), [batch['text'][idx]], fmt='%s')

                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

                for metric in metrics_dicts:
                    lengths = batch['length']
                    if metric == "TemosMetric":
                        getattr(self.metrics, metric).update(rs_set["joints_rst"], rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl", "lm_adaptor_pretrain"]:
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            if self.trainer.datamodule.is_mm:
                                word_embs = word_embs.repeat_interleave(self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
                                text_lengths = text_lengths.repeat_interleave(self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "TMRMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            texts=batch["text"]
                        )
                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(self.metrics, metric).update(rs_set["joints_rst"], rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(self.metrics, metric).update(rs_set["joints_rst"], rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        getattr(self.metrics, metric).update(
                            feats_rst=rs_set["m_rst"],
                            lengths_rst=rs_set['length'])
            else:
                # M2T or T2T metrics
                for metric in self.hparams.metrics_dict:
                    lengths = batch['length']
                    if metric == "M2TMetrics":
                        word_embs = batch['word_embs']
                        pos_ohot = batch['pos_ohot']
                        text_lengths = batch['text_len']
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=rs_set["t_ref"],
                            lengths=lengths,
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths
                        )

        if split in ["val", "test"]:
            if self.hparams.task == "m2t":
                # Return full data for M2T CSV saving: (t_pred, length, m_ref, t_ref, fname)
                return (rs_set["t_pred"], rs_set["length"], rs_set.get("m_ref"), 
                        rs_set.get("t_ref"), batch.get("fname", []))
            return rs_set["t_pred"], rs_set["length"]

        return loss
