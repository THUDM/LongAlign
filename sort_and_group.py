from transformers import AutoTokenizer, AutoModel
import copy
import torch
import json, os
import multiprocessing
from tqdm import tqdm
import numpy as np
import random
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--group_size', default=8, type=int)
    parser.add_argument('--train_file', type=str)
    return parser.parse_args(args)

def main(args):
    filepath = args.train_file
    PAD_ID = 0
    group_size = args.group_size

    # sort
    input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs.npy')))
    labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels.npy')))

    eos_indices = input_ids.argmin(dim=1) - 1
    eos_indices, indices = torch.sort(eos_indices)

    input_ids, labels = input_ids[indices], labels[indices]
    group_num = input_ids.size(0) // group_size
    input_ids = input_ids[:group_num*group_size].view(group_num, group_size, -1)
    labels = labels[:group_num*group_size].view(group_num, group_size, -1)
    idx = torch.randperm(group_num)
    input_ids = input_ids[idx].view(group_num*group_size, -1)
    labels = labels[idx].view(group_num*group_size, -1)
    np.save(os.path.join(filepath, 'inputs_sort.npy'), input_ids.numpy().astype(np.int64))
    np.save(os.path.join(filepath, 'labels_sort.npy'), labels.numpy().astype(np.int64))

    # pack
    input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs.npy')))
    labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels.npy')))
    num, max_length = input_ids.shape
    max_length = 65536
    new_inputs = []
    new_labels = []
    new_weights = []
    attention_masks = []
    tmp_input = torch.full((max_length,), PAD_ID, dtype=torch.int64)
    tmp_label = torch.full((max_length,), -100, dtype=torch.int64)
    tmp_weight = torch.full((max_length,), 0., dtype=torch.float32)
    attention_mask = [0]
    curr_idx = 0
    idx = 0
    total_len = []
    while idx < num:
        input_id, label = input_ids[idx], labels[idx]
        eos_indice = input_id.argmin().item() - 1
        if eos_indice + 1 >= max_length or eos_indice < 0: # exceed max length
            idx += 1
            continue
        elif curr_idx + eos_indice + 1 >= max_length: # full, start new pack
            total_len.append(len(attention_mask))
            new_inputs.append(tmp_input)
            new_labels.append(tmp_label)
            new_weights.append(tmp_weight)
            attention_masks.append(attention_mask+[max_length])
            curr_idx = 0
            tmp_input = torch.full((max_length,), PAD_ID, dtype=torch.int64)
            tmp_label = torch.full((max_length,), -100, dtype=torch.int64)
            tmp_weight = torch.full((max_length,), 0., dtype=torch.float32)
            attention_mask = [0]
        else: # pack in
            tmp_input[curr_idx: curr_idx+eos_indice+1] = input_id[:eos_indice+1]
            tmp_label[curr_idx: curr_idx+eos_indice+1] = label[:eos_indice+1]
            weight = torch.where(label[:eos_indice+1] == -100, 0, 1)
            if weight.sum() > 0.5:
                weight = weight / weight.sum()
            tmp_weight[curr_idx: curr_idx+eos_indice+1] = weight
            curr_idx += (eos_indice+1)
            attention_mask.append(curr_idx)
            idx += 1
    input_ids = torch.stack(new_inputs, dim=0)
    labels = torch.stack(new_labels, dim=0)
    weights = torch.stack(new_weights, dim=0)

    np.save(os.path.join(filepath, 'inputs_pack.npy'), input_ids.numpy().astype(np.int64))
    np.save(os.path.join(filepath, 'labels_pack.npy'), labels.numpy().astype(np.int64))
    np.save(os.path.join(filepath, 'weights_pack.npy'), weights.numpy())
    json.dump(attention_masks, open(os.path.join(filepath, 'attention_masks_pack.json'), 'w'))
    print(np.mean(total_len))

if __name__ == '__main__':
    main(parse_args())