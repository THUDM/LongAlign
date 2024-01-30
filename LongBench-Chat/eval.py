import requests
import time, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
import numpy as np
import random
import codecs
import argparse
from copy import deepcopy
from tqdm import tqdm
import traceback
import re
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

os.makedirs("predictions", exist_ok=True)
os.makedirs("scores", exist_ok=True)
test_cases = json.load(open("test_cases.json", "r"))
few_shots = json.load(open('few_shots.json', 'r', encoding='utf-8'))
template = open("prompt_fs.txt", encoding="utf-8").read()
system_prompt = "You're a good assistant at evaluating the quality of texts."
GPT_MODEL = 'gpt-4'
api_key = '' # Enter your openai api key here
    
def query_gpt4(prompt):
    msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    tries = 0
    while tries < 5:
        tries += 1
        try:
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": GPT_MODEL,
                "messages": msg,
                "temperature": 0.
            }, headers=headers, timeout=120)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return
    
    return resp

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def chat(model, path, tokenizer, prompt, device, history=[], max_new_tokens=1024, temperature=1.0):
    valid_path = path.lower()
    if "internlm" in valid_path or "chatglm" in valid_path:
        response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature)
        return response, history
    elif "llama" in valid_path:
        if history == []:
            prompt = f"[INST]{prompt}[/INST]"
        else:
            prompt = history+"\n\n"+f"[INST]{prompt}[/INST]"
    elif "mistral" in valid_path or "mixtral" in valid_path:
        if history == []:
            prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            prompt = history+f"</s> [INST] {prompt} [/INST]"
    elif "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]
    output = model.generate(
        **input,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        temperature=temperature,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred.strip(), prompt + pred.strip()

def load_model_and_tokenizer(path, device):
    valid_path = path.lower()
    if "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import load_model
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(path, device='cpu', num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "mistral" in valid_path or "mixtral" in valid_path:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, use_flash_attention_2=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    return model, tokenizer

def get_predictions(path, max_length):
    save_name = path.replace("/", "\\")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(path, device=device)
    with open(f"predictions/{save_name}.txt", "w", encoding='utf-8') as f:
        for case in test_cases:
            seed_everything(42)
            history = []
            prompt = case["prompt"]
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            print(case["idx"], len(tokenized_prompt))
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            response, history = chat(model, path, tokenizer, prompt, device=device, history=history, max_new_tokens=1024, temperature=1.0)
            line = response.strip().replace('\n', ' ') + '\n'
            f.write(line)
            f.flush()
    
def get_score(path):
    save_name = path.replace("/", "\\")
    predictions = []
    with open(f"predictions/{save_name}.txt", "r", encoding='utf-8') as f:
        for line in f:
            predictions.append(line.strip())
    assert len(predictions) == len(test_cases)
    result = []
    lines = []
    scores = []
    total_tokens = 0
    for case, prediction in tqdm(zip(deepcopy(test_cases), predictions)):
        question, answer = case["query"], case["answer"]
        few_shot_answers = [x['answer'] for x in few_shots[case['idx'] - 1]]
        few_shot_scores = [x['score'] for x in few_shots[case['idx'] - 1]]
        few_shot_ans_scores = []
        for k in range(len(few_shot_answers)):
            few_shot_ans_scores.append(few_shot_answers[k])
            few_shot_ans_scores.append(few_shot_scores[k])
        prompt = template.format(question, answer, *few_shot_ans_scores, prediction)
        score = "none"
        trys = 0
        while (score == "none") and (trys < 5):
            response = query_gpt4(prompt)
            try:
                num_tokens = response["usage"]["total_tokens"]
                response = response["choices"][0]["message"]["content"]
                score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
                matches = re.findall(r"\d+\.\d+|\d+", score)
                score = matches[0]
            except:
                trys += 1
                num_tokens = 0
                score = "none"
        total_tokens += num_tokens
        scores.append(score)
        lines.append(prediction + '\t' + score + '\n')
        case.update({
            "prediction": prediction,
            "gpt_analysis": response,
            "score": score,
            "used_tokens": num_tokens
        })
        case.pop("prompt")
        result.append(case)
    try:
        scores = [float(score) for score in scores]
        total_score = sum(scores)
    except Exception as e:
        traceback.print_exc()
        total_score = "none"
    
    result.append({
        "total_score": total_score,
        "total_tokens": total_tokens,
    })
    
    with codecs.open(f"scores/{save_name}.json", 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=False)
    parser.add_argument("--max_length", default=64000, type=int)
    args = parser.parse_args()

    get_predictions(args.model_path, args.max_length)
    get_score(args.model_path)