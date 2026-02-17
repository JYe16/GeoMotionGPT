import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import spacy
from omegaconf import OmegaConf

from motGPT.config import parse_args, instantiate_from_config
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.metrics.utils import euclidean_distance_matrix, calculate_top_k
from motGPT.utils.load_checkpoint import load_pretrained


def _to_device_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _normalize_refs(refs: List[str]) -> List[str]:
    refs = [r if isinstance(r, str) else str(r) for r in refs if r is not None]
    refs = [r.strip() for r in refs]
    refs = [r for r in refs if len(r) > 0]
    if len(refs) == 0:
        return ["", "", ""]
    if len(refs) == 1:
        return [refs[0], refs[0], refs[0]]
    if len(refs) == 2:
        return [refs[0], refs[1], refs[0]]
    return refs[:3]


class TM2TCompatEvaluator:
    def __init__(self, cfg, dataname: str, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.unit_length = int(cfg.TM2T_COMPAT.UNIT_LEN)

        if dataname == "kit":
            eval_dataset_name = "kit"
        else:
            eval_dataset_name = "t2m"

        self.text_encoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_textencoder).to(device)
        self.move_encoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_moveencoder).to(device)
        self.motion_encoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_motionencoder).to(device)

        ckpt_path = os.path.join(
            cfg.METRIC.TM2T.t2m_path,
            eval_dataset_name,
            "text_mot_match/model/finest.tar",
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.text_encoder.load_state_dict(checkpoint["text_encoder"])
        self.move_encoder.load_state_dict(checkpoint["movement_encoder"])
        self.motion_encoder.load_state_dict(checkpoint["motion_encoder"])

        self.text_encoder.eval()
        self.move_encoder.eval()
        self.motion_encoder.eval()
        for module in (self.text_encoder, self.move_encoder, self.motion_encoder):
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        word_embs = word_embs.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        cap_lens = cap_lens.detach().to(self.device).long()
        motions = motions.detach().to(self.device).float()

        if not torch.is_tensor(m_lens):
            m_lens = torch.tensor(m_lens, device=self.device)
        else:
            m_lens = m_lens.to(self.device)

        align_idx = np.argsort(m_lens.detach().cpu().numpy().tolist())[::-1].copy()
        align_idx_t = torch.from_numpy(align_idx).to(self.device)

        motions = motions[align_idx_t]
        m_lens_sorted = m_lens[align_idx_t] // self.unit_length

        movements = self.move_encoder(motions[..., :-4]).detach()
        motion_embedding = self.motion_encoder(movements, m_lens_sorted)

        # TM2T text encoder uses pack_padded_sequence (enforce_sorted=True),
        # so we sort by cap_lens here and then restore original batch order.
        text_sort_idx = torch.argsort(cap_lens, descending=True)
        text_unsort_idx = torch.argsort(text_sort_idx)
        text_embedding_sorted = self.text_encoder(
            word_embs[text_sort_idx],
            pos_ohot[text_sort_idx],
            cap_lens[text_sort_idx],
        )
        text_embedding = text_embedding_sorted[text_unsort_idx]
        text_embedding = text_embedding[align_idx_t]
        return text_embedding, motion_embedding


class PredTextEncoder:
    def __init__(self, w_vectorizer, max_text_len: int):
        self.w_vectorizer = w_vectorizer
        self.max_text_len = max_text_len
        self.nlp = spacy.load("en_core_web_sm")

    def _process_text(self, sentence: str) -> Tuple[List[str], List[str]]:
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        words, poss = [], []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if token.pos_ in ["NOUN", "VERB"] and word != "left":
                words.append(token.lemma_)
            else:
                words.append(word)
            poss.append(token.pos_)
        return words, poss

    def encode_batch(self, texts: List[str], device: torch.device):
        word_embs = []
        pos_ohots = []
        lengths = []

        for sentence in texts:
            sentence = sentence if isinstance(sentence, str) else str(sentence)
            w_list, p_list = self._process_text(sentence.strip())
            text_tokens = [f"{w_list[i]}/{p_list[i]}" for i in range(len(w_list))]

            if len(text_tokens) < self.max_text_len:
                tokens = ["sos/OTHER"] + text_tokens + ["eos/OTHER"]
                sent_len = len(tokens)
                tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = text_tokens[: self.max_text_len]
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)

            pos_one_hots = []
            word_embeddings = []
            for tk in tokens:
                word_emb, pos_oh = self.w_vectorizer[tk]
                word_embeddings.append(torch.tensor(word_emb, dtype=torch.float32)[None, :])
                pos_one_hots.append(torch.tensor(pos_oh, dtype=torch.float32)[None, :])

            word_embs.append(torch.cat(word_embeddings, dim=0))
            pos_ohots.append(torch.cat(pos_one_hots, dim=0))
            lengths.append(sent_len)

        lengths = torch.tensor(lengths, dtype=torch.long).to(device)
        word_embs = torch.stack(word_embs, dim=0).to(device)
        pos_ohots = torch.stack(pos_ohots, dim=0).to(device)
        return word_embs, pos_ohots, lengths


def _compute_matching(
    evaluator: TM2TCompatEvaluator,
    text_emb_batches: List[torch.Tensor],
    motion_emb_batches: List[torch.Tensor],
    top_k: int = 3,
    group_size: int = 32,
):
    if len(text_emb_batches) == 0:
        return 0.0, [0.0] * top_k

    all_text = torch.cat(text_emb_batches, dim=0)
    all_motion = torch.cat(motion_emb_batches, dim=0)

    if group_size <= 0:
        group_size = 32
    valid_size = (all_text.shape[0] // group_size) * group_size
    if valid_size == 0:
        return 0.0, [0.0] * top_k

    all_text = all_text[:valid_size]
    all_motion = all_motion[:valid_size]

    top_k_count = np.zeros((top_k,), dtype=np.float64)
    matching_score_sum = 0.0

    num_groups = valid_size // group_size
    for i in range(num_groups):
        t_group = all_text[i * group_size:(i + 1) * group_size]
        m_group = all_motion[i * group_size:(i + 1) * group_size]
        dist_mat = euclidean_distance_matrix(m_group, t_group).detach().cpu().numpy()
        matching_score_sum += float(np.trace(dist_mat))

        argsmax = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k(torch.from_numpy(argsmax), top_k=top_k).cpu().numpy()
        top_k_count += top_k_mat.sum(axis=0)

    return matching_score_sum / valid_size, (top_k_count / valid_size).tolist()


def _compute_nlg_metrics(pred_texts: List[str], gt_caps: List[List[str]], compute_bert: bool):
    ref_list = [list(refs) for refs in zip(*gt_caps)]
    scores = {}

    try:
        import importlib

        NLGEval = importlib.import_module("nlgeval").NLGEval

        nlg_eval = NLGEval(
            metrics_to_omit=[
                "EmbeddingAverageCosineSimilarity",
                "SkipThoughtCS",
                "VectorExtremaCosineSimilarity",
                "GreedyMatchingScore",
            ]
        )
        nlg_scores = nlg_eval.compute_metrics(ref_list, pred_texts)
        scores["bleu_1"] = float(nlg_scores.get("Bleu_1", 0.0))
        scores["bleu_2"] = float(nlg_scores.get("Bleu_2", 0.0))
        scores["bleu_3"] = float(nlg_scores.get("Bleu_3", 0.0))
        scores["bleu_4"] = float(nlg_scores.get("Bleu_4", 0.0))
        scores["METEOR"] = float(nlg_scores.get("METEOR", 0.0))
        scores["ROUGE_L"] = float(nlg_scores.get("ROUGE_L", 0.0))
        scores["CIDEr"] = float(nlg_scores.get("CIDEr", 0.0))
    except Exception:
        from nlgmetricverse import NLGMetricverse, load_metric

        nlg = NLGMetricverse(
            metrics=[
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_2", compute_kwargs={"max_order": 2}),
                load_metric("bleu", resulting_name="bleu_3", compute_kwargs={"max_order": 3}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("meteor"),
                load_metric("rouge"),
                load_metric("cider"),
            ]
        )
        nlg_scores = nlg(predictions=pred_texts, references=gt_caps)
        scores["bleu_1"] = float(nlg_scores.get("bleu_1", {}).get("score", 0.0))
        scores["bleu_2"] = float(nlg_scores.get("bleu_2", {}).get("score", 0.0))
        scores["bleu_3"] = float(nlg_scores.get("bleu_3", {}).get("score", 0.0))
        scores["bleu_4"] = float(nlg_scores.get("bleu_4", {}).get("score", 0.0))
        meteor_raw = nlg_scores.get("meteor", {})
        scores["METEOR"] = float(meteor_raw.get("score", 0.0) if isinstance(meteor_raw, dict) else meteor_raw)
        rouge_raw = nlg_scores.get("rouge", {})
        if isinstance(rouge_raw, dict):
            scores["ROUGE_L"] = float(rouge_raw.get("rougeL", rouge_raw.get("score", 0.0)))
        else:
            scores["ROUGE_L"] = float(rouge_raw)
        cider_raw = nlg_scores.get("cider", {})
        scores["CIDEr"] = float(cider_raw.get("score", 0.0) if isinstance(cider_raw, dict) else cider_raw)

    for key in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "METEOR", "ROUGE_L", "CIDEr"):
        scores.setdefault(key, 0.0)

    if compute_bert:
        try:
            from bert_score import score as bert_score

            primary_refs = [refs[0] for refs in gt_caps]
            _, _, f1 = bert_score(
                pred_texts,
                primary_refs,
                lang="en",
                rescale_with_baseline=True,
                idf=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
                nthreads=1,
            )
            scores["Bert_F1"] = float(f1.mean().item())
        except Exception:
            scores["Bert_F1"] = 0.0
    else:
        scores["Bert_F1"] = 0.0

    return scores


def main():
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER
    model_name = cfg.model.target.split(".")[-2]
    cfg.FOLDER_EXP = str(Path(cfg.FOLDER) / model_name / cfg.NAME)
    cfg.TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if not hasattr(cfg, "TM2T_COMPAT"):
        raise ValueError("Missing TM2T_COMPAT section in config.")

    if not bool(cfg.TM2T_COMPAT.ENABLED):
        print("TM2T_COMPAT.ENABLED is false, nothing to run.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datamodule = build_data(cfg, phase="test")
    datamodule.setup("test")

    model = build_model(cfg, datamodule)
    if cfg.TEST.CHECKPOINTS:
        load_pretrained(cfg, model, phase="test")
    model = model.to(device)
    model.eval()

    dataname = str(getattr(cfg, "DATASET_NAME", "humanml3d")).lower()
    evaluator = TM2TCompatEvaluator(cfg, dataname=dataname, device=device)
    pred_encoder = PredTextEncoder(
        w_vectorizer=datamodule.hparams.w_vectorizer,
        max_text_len=int(cfg.TM2T_COMPAT.MAX_TEXT_LEN),
    )

    pred_text_emb_batches, gt_text_emb_batches, motion_emb_batches = [], [], []
    all_pred_texts: List[str] = []
    all_gt_captions: List[List[str]] = []

    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            cpu_batch = batch
            batch = _to_device_batch(batch, device)

            rs_set = model.val_m2t_forward(batch)
            pred_texts = rs_set["t_pred"]
            feats_ref = rs_set["m_ref"] if bool(cfg.TM2T_COMPAT.APPLY_RENORM4T2M) else batch["motion"]
            lengths = batch["length"]

            gt_caps_batch = [_normalize_refs(caps) for caps in cpu_batch["all_captions"]]

            pred_word, pred_pos, pred_lens = pred_encoder.encode_batch(pred_texts, device)
            pred_text_emb, motion_emb = evaluator.get_co_embeddings(
                word_embs=pred_word,
                pos_ohot=pred_pos,
                cap_lens=pred_lens,
                motions=feats_ref,
                m_lens=lengths,
            )

            gt_text_emb, _ = evaluator.get_co_embeddings(
                word_embs=batch["word_embs"],
                pos_ohot=batch["pos_ohot"],
                cap_lens=batch["text_len"],
                motions=feats_ref,
                m_lens=lengths,
            )

            pred_text_emb_batches.append(pred_text_emb.detach().cpu())
            gt_text_emb_batches.append(gt_text_emb.detach().cpu())
            motion_emb_batches.append(motion_emb.detach().cpu())

            all_pred_texts.extend([str(t) for t in pred_texts])
            all_gt_captions.extend(gt_caps_batch)

    matching_pred, r_pred = _compute_matching(
        evaluator,
        pred_text_emb_batches,
        motion_emb_batches,
        top_k=int(cfg.TM2T_COMPAT.TOP_K),
        group_size=int(getattr(cfg.TM2T_COMPAT, "R_SIZE", 32)),
    )
    matching_gt, r_gt = _compute_matching(
        evaluator,
        gt_text_emb_batches,
        motion_emb_batches,
        top_k=int(cfg.TM2T_COMPAT.TOP_K),
        group_size=int(getattr(cfg.TM2T_COMPAT, "R_SIZE", 32)),
    )

    nlg_scores = _compute_nlg_metrics(
        pred_texts=all_pred_texts,
        gt_caps=all_gt_captions,
        compute_bert=bool(cfg.TM2T_COMPAT.COMPUTE_BERT_SCORE),
    )

    results = {
        "protocol": "tm2t_m2t_compat",
        "dataset": dataname,
        "num_samples": len(all_pred_texts),
        "Matching_score": float(matching_pred),
        "gt_Matching_score": float(matching_gt),
        "R_precision_top_1": float(r_pred[0]) if len(r_pred) > 0 else 0.0,
        "R_precision_top_2": float(r_pred[1]) if len(r_pred) > 1 else 0.0,
        "R_precision_top_3": float(r_pred[2]) if len(r_pred) > 2 else 0.0,
        "gt_R_precision_top_1": float(r_gt[0]) if len(r_gt) > 0 else 0.0,
        "gt_R_precision_top_2": float(r_gt[1]) if len(r_gt) > 1 else 0.0,
        "gt_R_precision_top_3": float(r_gt[2]) if len(r_gt) > 2 else 0.0,
        **nlg_scores,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    output_root = Path(cfg.FOLDER) / cfg.model.target.split(".")[-2].lower() / cfg.NAME
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_root / f"tm2t_compat_metrics_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved TM2T-compatible metrics to: {output_path}")


if __name__ == "__main__":
    main()
