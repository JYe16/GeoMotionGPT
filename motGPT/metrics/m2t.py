from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
from .utils import *
from bert_score import score as score_bert
import spacy
from motGPT.config import instantiate_from_config

class M2TMetrics(Metric):

    def __init__(self,
                 cfg,
                 w_vectorizer,
                 dataname='humanml3d',
                 top_k=3,
                 bleu_k=4,
                 R_size=32,
                 max_text_len=40,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 unit_length=4,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.w_vectorizer = w_vectorizer
        self.name = "matching, fid, and diversity scores"
        # self.text = True if cfg.TRAIN.STAGE in ["diffusion","t2m_gpt"] else False
        self.max_text_len = max_text_len
        self.top_k = top_k
        self.bleu_k = bleu_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.unit_length = unit_length

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []

        # Matching scores
        self.add_state("Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"gt_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.Matching_metrics)

        # NLG
        # for k in range(1, top_k + 1):
        #     self.add_state(
        #         f"Bleu_{str(k)}",
        #         default=torch.tensor(0.0),
        #         dist_reduce_fx="sum",
        #     )
        #     self.metrics.append(f"Bleu_{str(k)}")

        self.add_state("ROUGE_L",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("ROUGE_L")

        self.add_state("CIDEr",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("CIDEr")

        # Chached batches
        self.pred_texts = []
        self.gt_texts = []
        self.add_state("predtext_embeddings", default=[])
        self.add_state("gttext_embeddings", default=[])
        self.add_state("gtmotion_embeddings", default=[])

        # T2M Evaluator
        self._get_t2m_evaluator(cfg)

        self.nlp = spacy.load('en_core_web_sm')

        if self.cfg.model.params.task == 'm2t':
            from nlgmetricverse import NLGMetricverse, load_metric
            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            self.nlg_evaluator = NLGMetricverse(metrics)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_textencoder)
        self.t2m_moveencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_moveencoder)
        self.t2m_motionencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_motionencoder)


        # load pretrianed
        if self.dataname == "kit":
            dataname = "kit"
        elif self.dataname == 'tomato':
            dataname ='motionx/vector_313'
        elif self.dataname == 'motionx':
            # dataname = "t2m"
            cfg.METRIC.TM2T.t2m_moveencoder.params['input_size'] = 263-4
            # dataname ='motionx/vector_623'
            dataname ='motionx/x-vector_263'
        else:
            dataname = "t2m"

        t2m_checkpoint = torch.load(os.path.join(
            cfg.METRIC.TM2T.t2m_path, dataname, "text_mot_match/model/finest.tar"),
                                    map_location='cpu')
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def _process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN'
                    or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def _get_text_embeddings(self, texts):
        word_embs = []
        pos_ohot = []
        text_lengths = []
        for i, sentence in enumerate(texts):
            word_list, pos_list = self._process_text(sentence.strip())
            t_tokens = [
                '%s/%s' % (word_list[i], pos_list[i])
                for i in range(len(word_list))
            ]

            if len(t_tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'
                                   ] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = t_tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(torch.tensor(pos_oh).float()[None])
                word_embeddings.append(torch.tensor(word_emb).float()[None])
            text_lengths.append(sent_len)
            pos_ohot.append(torch.cat(pos_one_hots, dim=0)[None])
            word_embs.append(torch.cat(word_embeddings, dim=0)[None])

        word_embs = torch.cat(word_embs, dim=0).to(self.Matching_score)
        pos_ohot = torch.cat(pos_ohot, dim=0).to(self.Matching_score)
        text_lengths = torch.tensor(text_lengths).to(self.Matching_score)

        align_idx = np.argsort(text_lengths.data.tolist())[::-1].copy()

        # get text embeddings
        text_embeddings = self.t2m_textencoder(word_embs[align_idx],
                                               pos_ohot[align_idx],
                                               text_lengths[align_idx])

        original_text_embeddings = text_embeddings.clone()

        for idx, sort in enumerate(align_idx):
            original_text_embeddings[sort] = text_embeddings[idx]

        return original_text_embeddings

    @torch.no_grad()
    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(count_seq)
        all_predtexts = torch.cat(self.predtext_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]

        if len(self.gtmotion_embeddings) > 0:
            all_motions = torch.cat(self.gtmotion_embeddings,
                                    axis=0).cpu()[shuffle_idx, :]
            all_gttexts = torch.cat(self.gttext_embeddings,
                                    axis=0).cpu()[shuffle_idx, :]

        # Compute r-precision
        if len(self.gtmotion_embeddings) > 0:
            assert count_seq >= self.R_size
            top_k_mat = torch.zeros((self.top_k, ))
            for i in range(count_seq // self.R_size):
                # [bs=32, 1*256]
                group_texts = all_predtexts[i * self.R_size:(i + 1) * self.R_size]
                # [bs=32, 1*256]
                group_motions = all_motions[i * self.R_size:(i + 1) * self.R_size]
                # [bs=32, 32]
                dist_mat = euclidean_distance_matrix(group_texts,
                                                     group_motions).nan_to_num()
                # print(dist_mat[:5])
                self.Matching_score += dist_mat.trace()
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

            R_count = count_seq // self.R_size * self.R_size
            metrics["Matching_score"] = self.Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

            # Compute r-precision with gt
            assert count_seq >= self.R_size
            top_k_mat = torch.zeros((self.top_k, ))
            for i in range(count_seq // self.R_size):
                # [bs=32, 1*256]
                group_texts = all_gttexts[i * self.R_size:(i + 1) * self.R_size]
                # [bs=32, 1*256]
                group_motions = all_motions[i * self.R_size:(i + 1) * self.R_size]
                # [bs=32, 32]
                dist_mat = euclidean_distance_matrix(group_texts,
                                                     group_motions).nan_to_num()
                # match score
                self.gt_Matching_score += dist_mat.trace()
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
            
            metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # NLP metrics
        print(f"Computing NLG metrics for {len(self.pred_texts)} samples...")
        import sys
        sys.stdout.flush()  # Force print output immediately
        
        try:
            scores = self.nlg_evaluator(predictions=self.pred_texts,
                                        references=self.gt_texts)
            
            for key in scores.keys():
                if 'bleu' in key:
                    metrics[key] = torch.tensor(scores[key]['score'],device=self.device)
            # for k in range(1, self.bleu_k + 1):
            #     metrics[f"Bleu_{str(k)}"] = torch.tensor(scores[f'bleu_{str(k)}'],
            #                                              device=self.device)
            
            # Handle ROUGE score - format may vary depending on nlgmetricverse version
            if "rouge" in scores:
                rouge_data = scores["rouge"]
                if isinstance(rouge_data, dict):
                    if "rougeL" in rouge_data:
                        rouge_l_val = rouge_data["rougeL"]
                    elif "score" in rouge_data:
                        rouge_l_val = rouge_data["score"]
                    else:
                        # Try to find any rouge-L related key
                        rouge_l_val = rouge_data.get("rouge-l", rouge_data.get("rougeLsum", 0.0))
                        if isinstance(rouge_l_val, dict):
                            rouge_l_val = rouge_l_val.get("score", rouge_l_val.get("fmeasure", 0.0))
                else:
                    rouge_l_val = float(rouge_data) if rouge_data else 0.0
                metrics["ROUGE_L"] = torch.tensor(rouge_l_val, device=self.device)
            else:
                print("Warning: 'rouge' not found in scores, setting ROUGE_L to 0")
                metrics["ROUGE_L"] = torch.tensor(0.0, device=self.device)
            
            # Handle CIDEr score
            if "cider" in scores:
                cider_data = scores["cider"]
                if isinstance(cider_data, dict) and "score" in cider_data:
                    cider_val = cider_data["score"]
                else:
                    cider_val = float(cider_data) if cider_data else 0.0
                metrics["CIDEr"] = torch.tensor(cider_val, device=self.device)
            else:
                print("Warning: 'cider' not found in scores, setting CIDEr to 0")
                metrics["CIDEr"] = torch.tensor(0.0, device=self.device)
            print("NLG metrics computed.")
            sys.stdout.flush()
        except Exception as e:
            print(f"Warning: NLG metrics computation failed: {e}")
            print("Setting default values for NLG metrics...")
            sys.stdout.flush()
            metrics["bleu_1"] = torch.tensor(0.0, device=self.device)
            metrics["bleu_4"] = torch.tensor(0.0, device=self.device)
            metrics["ROUGE_L"] = torch.tensor(0.0, device=self.device)
            metrics["CIDEr"] = torch.tensor(0.0, device=self.device)

        # Bert metrics - use GPU for faster computation
        # Skip BERTScore during training to avoid slowdown and potential hangs
        skip_bert_score = os.environ.get('SKIP_BERT_SCORE', '0') == '1'
        if skip_bert_score:
            print("Skipping BERTScore computation (SKIP_BERT_SCORE=1)")
            metrics["Bert_F1"] = torch.tensor(0.0, device=self.device)
        else:
            try:
                # Limit samples to avoid very long computation times
                max_samples = min(len(self.pred_texts), 500)
                P, R, F1 = score_bert(self.pred_texts[:max_samples],
                                      self.gt_texts[:max_samples],
                                      lang='en',
                                      rescale_with_baseline=True,
                                      idf=True,
                                      device=self.device,
                                      verbose=False,
                                      nthreads=1)  # Disable multiprocessing to avoid SIGTERM
                metrics["Bert_F1"] = F1.mean().to(self.device)
            except Exception as e:
                print(f"Warning: BERTScore computation failed: {e}")
                metrics["Bert_F1"] = torch.tensor(0.0, device=self.device)

        # Reset
        self.reset()
        self.gt_texts = []
        self.pred_texts = []

        return {**metrics}

    @torch.no_grad()
    def update(self,
               feats_ref: Tensor,
               pred_texts: List[str],
               gt_texts: List[str],
               lengths: List[int],
               word_embs: Tensor = None,
               pos_ohot: Tensor = None,
               text_lengths: Tensor = None):

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # Get motion embeddings (no sorting needed, pack_padded_sequence uses enforce_sorted=False)
        predtext_emb = self._get_text_embeddings(pred_texts)
        predtext_embeddings = torch.flatten(predtext_emb, start_dim=1).detach()
        self.predtext_embeddings.append(predtext_embeddings)

        if word_embs is not None:
            gtmotion_embeddings = self.get_motion_embeddings(feats_ref, lengths)
            self.gtmotion_embeddings.append(gtmotion_embeddings)

            # text encoder (no sorting needed)
            gttext_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                              text_lengths)
            gttext_embeddings = torch.flatten(gttext_emb, start_dim=1).detach()
            self.gttext_embeddings.append(gttext_embeddings)

        self.pred_texts.extend(pred_texts)
        self.gt_texts.extend(gt_texts)

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        m_lens = torch.tensor(lengths)
        # 只除一次 UNIT_LEN，原代码除了两次是 bug
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")
        if self.dataname == 'tomato':
            mov = self.t2m_moveencoder(feats[...,4:]).detach()
        elif self.dataname == 'motionx':
            njoints=52
            feats_22 = torch.cat((feats[..., :4+21*3], feats[..., 4+(njoints-1)*3:4+(njoints-1)*3+21*6],
                               feats[..., 4+(njoints-1)*9:4+(njoints-1)*9+22*3]), dim=-1).to(feats)
            mov = self.t2m_moveencoder(feats_22).detach()
        else:
            mov = self.t2m_moveencoder(feats[...,:-4]).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        return torch.flatten(emb, start_dim=1).detach()
