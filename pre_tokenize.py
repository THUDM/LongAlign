from transformers import AutoTokenizer, AutoModel, LlamaTokenizer
import copy
import torch
import json, os, random
import multiprocessing
from tqdm import tqdm
import traceback
import numpy as np
import argparse

BEGIN_TOKEN, END_TOKEN = '<<BEGIN>>', '<<END>>'
max_length = 65536
PAD_ID = 0
EOS_ID = 2
skip_exceed_length_case = False
truncate_side = 'right'

PROC_NUM = 128
save_dir = 'multiprocess_data'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default="chatglm", type=str)
    parser.add_argument('--datanum', default="10k", type=str)
    return parser.parse_args(args)

def process_file(lines, rank, args):
    def build_input(conversations, tokenizer, args):
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"] # filter null characters
        for conv in conversations:
            if conv['role'] == "assistant":
                for char in zero_width_chars:
                    conv['content'] = conv['content'].replace(char, '')

        if len(conversations) % 2 != 0:
            conversations = conversations[:-1]
        if len(conversations) == 0:
            return None

        l = []
        for i in range(0, len(conversations), 2):
            local_rank = i // 2
            ipt, ans = conversations[i]['content'], conversations[i + 1]['content']
            if ipt == None:
                ipt = ''
            if ans == None:
                ans = ''
            ipt, ans = ipt, BEGIN_TOKEN + ans + END_TOKEN
            if model == "chatglm": # "[Round 1]\n\n问：aaaa\n\n答：bbbb\n\n[Round 2]\n\n问：cccc\n\n答："
                l.append(
                    '[Round {}]\n\n问：{}\n\n答：{}'.format(local_rank + 1, ipt, ans)
                )
            else: # "[INST]aaaa[/INST]bbbb\n\n[INST]cccc[/INST]"
                l.append(
                    '[INST]{}[/INST]{}'.format(ipt, ans)
                )
        ipt = '\n\n'.join(l)
        inputs = tokenizer(ipt, return_tensors='pt')['input_ids'][0]
        inputs = torch.cat([inputs, torch.tensor([EOS_ID])], dim=0)
        labels = torch.full_like(inputs, -100)
        begin_ids = torch.nonzero(inputs == BEGIN_ID).squeeze().tolist()
        end_ids = torch.nonzero(inputs == END_ID).squeeze().tolist()
        if isinstance(begin_ids, int):
            begin_ids = [begin_ids]
            end_ids = [end_ids]
        assert len(begin_ids) == len(end_ids)
        for begin_id, end_id in zip(begin_ids, end_ids):
            labels[begin_id: end_id + 1] = inputs[begin_id: end_id + 1]
            labels[end_id + 1] = EOS_ID
        del_list = torch.tensor([BEGIN_ID, END_ID])
        inputs = inputs[~torch.isin(inputs, del_list)]
        labels = labels[~torch.isin(labels, del_list)]
        labels[-1] = EOS_ID
        for i in range(1, len(labels)): # mask out empty responses
            if labels[i - 1] == -100 and labels[i] == 2:
                labels[i] = -100
        if inputs.shape[0] < max_length:
            inputs = torch.cat([inputs, torch.full((max_length - inputs.shape[0],), PAD_ID, dtype=torch.int64)], dim=0)
            labels = torch.cat([labels, torch.full((max_length - labels.shape[0],), -100, dtype=torch.int64)], dim=0)
        else:
            print("exceed_length")
            if skip_exceed_length_case:
                return None
            cut_num = inputs.shape[0] - max_length
            if truncate_side == 'right':
                inputs = inputs[:-cut_num]
                labels = labels[:-cut_num]
            elif truncate_side == 'left':
                inputs = torch.cat([inputs[:2], inputs[2 + cut_num:]], dim=0)
                labels = torch.cat([labels[:2], labels[2 + cut_num:]], dim=0)
            else:
                raise ValueError('truncate_side must be "right" or "left"')
        return inputs, labels

    try:
        final_inputs = torch.zeros((len(lines), max_length), dtype=torch.int64)
        final_labels = torch.zeros((len(lines), max_length), dtype=torch.int64)
        if model == 'chatglm':
            tokenizer = AutoTokenizer.from_pretrained("THUDM/LongAlign-6B-64k", trust_remote_code=True)
        else:
            tokenizer = LlamaTokenizer.from_pretrained("THUDM/LongAlign-7B-64k", trust_remote_code=True)
        tokenizer.add_special_tokens({'cls_token': BEGIN_TOKEN})
        tokenizer.add_special_tokens({'sep_token': END_TOKEN})
        BEGIN_ID, END_ID = tokenizer(BEGIN_TOKEN)['input_ids'][-1], tokenizer(END_TOKEN)['input_ids'][-1]
        pass_data_num = 0

        for line in tqdm(lines):
            conversations = json.loads(line)['messages']
            tmp = build_input(conversations, tokenizer, args)
            if tmp is None:
                continue
            inputs, labels = tmp
            final_inputs[pass_data_num] = inputs
            final_labels[pass_data_num] = labels
            pass_data_num += 1
        final_inputs = final_inputs[:pass_data_num]
        final_labels = final_labels[:pass_data_num]
        torch.save(final_inputs, os.path.join(save_dir, f'inputs_{rank}.pt'))
        torch.save(final_labels, os.path.join(save_dir, f'labels_{rank}.pt'))
    except Exception:
        with open('error.txt', 'a') as f:
            traceback.print_exc(file=f)

def main(args):
    final_dir = f'data/{args.model}/{args.datanum}'
    os.system('rm -r {}'.format(save_dir))
    os.makedirs(final_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    manager = multiprocessing.Manager()
    lines = manager.list()

    short = open('data/raw/sharegpt.jsonl', encoding='utf-8').readlines()
    long = open('data/raw/long.jsonl', encoding='utf-8').readlines()
    if args.datanum == '10k':
        lines = short + long
    elif args.datanum == '0k':
        lines = short
    elif args.datanum == '5k':
        lines = short + long[:5000]
    lines = short + long
    random.shuffle(lines)
    total_lines = len(lines)

    pool = multiprocessing.Pool(processes=PROC_NUM)
    lines_per_process = total_lines // PROC_NUM

    for i in range(PROC_NUM):
        start_line = i * lines_per_process
        end_line = None if i == PROC_NUM - 1 else (i + 1) * lines_per_process
        pool.apply_async(process_file, args=(lines[start_line:end_line], i, args))

    pool.close()
    pool.join()

    inputs, labels = [], []
    for i in tqdm(range(PROC_NUM)):
        inputs.append(torch.load(os.path.join(save_dir, f'inputs_{i}.pt')))
        labels.append(torch.load(os.path.join(save_dir, f'labels_{i}.pt')))
    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)

    input_ids = inputs.numpy().astype(np.int64)
    labels = labels.numpy().astype(np.int64)
    filtered_rows = np.where(~np.all(labels == -100, axis=1))[0]
    input_ids = input_ids[filtered_rows]
    labels = labels[filtered_rows]

    print(labels.shape)
    np.save(os.path.join(final_dir, 'inputs.npy'), input_ids)
    np.save(os.path.join(final_dir, 'labels.npy'), labels)


if __name__ == '__main__':
    main(parse_args())