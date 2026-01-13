import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def humanml3d_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][7] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[7], reverse=True)

    adapted_batch = {}
    # Motion only
    if notnone_batches[0][3] is not None:
        adapted_batch.update({
            "motion":
            collate_tensors([torch.tensor(b[3]).float() for b in notnone_batches]),
            "length": [b[4] for b in notnone_batches],
            })
    if notnone_batches[0][1] is not None:
        adapted_batch.update({
            "m_tokens":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "m_tokens_len": [b[2] for b in notnone_batches],
        })

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[9] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[5]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[6]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[7]) for b in notnone_batches]),
            "tokens": [b[8] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) > 10:
        adapted_batch.update({"tasks": [b[10] for b in notnone_batches]})
    # file_name
    if len(notnone_batches[0]) > 11:
        adapted_batch.update({"fname": [b[11] for b in notnone_batches]})
    # text, m_tokens, m_tokens_len, motion, length, word_embs, pos_ohot, text_len, tokens, all_captions, tasks, fname

    return adapted_batch


def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
