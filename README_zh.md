# LongAlign: A Recipe for Long Context Alignment of LLMs

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongAlign-10k" target="_blank">HF 仓库</a> • 📃 <a href="https://arxiv.org/abs/2401.18058" target="_blank">论文</a>
</p>

Read this in [English](README.md).

**LongAlign** 是首个大语言模型（LLM）针对长上下文对齐的全面方法。我们提出了 **LongAlign-10k** 数据集，包含 10,000 条长指令数据，长度在 8k-64k 之间。我们研究了训练策略，即 **packing (with loss weighting)** 和 **sorted batching**，这些都实现在了我们的代码中。为了评估真实世界中模型长上下文的性能，我们引入了 **LongBench-Chat**，它能够评估LLM对 10k-100k 长度任务的指令遵循能力。

## 🔍 目录
- [⚙️ 数据准备](#data-preparation)
- [🖥️ LongAlign 训练](#longalign-training)
- [📊 评测](#longbench-chat-evaluation)
- [📝 引用](#citation)

<a name="data-preparation"></a>

## ⚙️ 数据准备

您可以通过 Hugging Face datasets 下载并保存 **LongAlign-10k** 数据（[🤗 HF 仓库](https://huggingface.co/datasets/THUDM/LongAlign-10k)）：
```python
dataset = load_dataset('THUDM/LongAlign-10k')
for split, split_dataset in dataset.items():
    split_dataset.to_json("data/raw/long.jsonl")
```
ShareGPT 数据可以从[这里](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset)下载。我们参考了 [open-instruct](https://github.com/allenai/open-instruct) 仓库处理 ShareGPT 数据。请将数据文件保存在 `data/raw/sharegpt.jsonl`。您可以使用其他数据作为一般指令数据的来源，但请按以下格式处理好您的数据：
```json
{
    "messages": [{"role": "user", "content": "..."}, 
                 {"role": "assistant", "content": "..."}, ...]
    }
```

<a name="longalign-training"></a>
## 🖥️ LongAlign 训练

### 环境设置
使用 pip 安装所需环境：`pip install -r requirements.txt`。对于基于 Llama 的模型，我们推荐使用 FlashAttention 2 进行优化并节省 GPU 内存。相关依赖项可以根据 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的代码库进行安装。

### 数据预处理

首先，使用模型的分词器对原始文本数据进行分词。例如，训练 ChatGLM 时：
```bash
python pre_tokenize.py --model chatglm --datanum 10k
```
这里的 `--datanum` 参数指的是您想要在训练数据集中混入的长数据的数据量（我们的论文研究了 0k、5k 和 10k）。分词后的数据将保存在 `./data/chatglm/10k` 下。

对于 packing 和 sorted batching 策略，我们将数据处理成训练所需格式：
```bash
python sort_and_group.py --group_size 8 --train_file ./data/chatglm/10k
```
您应该将 `--group_size` 参数设置为训练期间的 GPU 数量。我们建议至少使用 8 个 80G GPU 进行模型训练，否则 64k 长度可能会导致内存溢出。

### 模型训练

我们在 `scripts/` 下提供了 ChatGLM3 和 Llama-2 模型系列的训练脚本。请调整 `--model_name_or_path`、`--train_file` 和 `--output_dir` 以匹配您的模型路径、数据路径和输出路径。请使用至少有 64k 上下文窗口长度的基座模型。我们发布了三个 **基座模型**，上下文窗口扩展到 64k：[LongAlign-6B-64k-base](https://huggingface.co/THUDM/LongAlign-6B-64k-base)、[LongAlign-7B-64k-base](https://huggingface.co/THUDM/LongAlign-7B-64k-base) 和 [LongAlign-13B-64k-base](https://huggingface.co/THUDM/LongAlign-13B-64k-base)。

对于 packing 训练，请修改*注意力计算*以支持传入标记了每个序列在 pack 中起止位置的 1D 注意力掩码，以及*模型前向计算*函数以支持 loss weighting。我们为 ChatGLM3 模型提供了此类修改的示例，在 [modeling_chatglm.py](https://github.com/THUDM/LongAlign/blob/main/modeling_chatglm.py) 中的 `CoreAttention.forward` 和 `ChatGLMForConditionalGeneration.forward`。您可以直接使用此文件作为 ChatGLM 训练中的模型文件。我们也提供了 Llama 的训练代码，如果要复现我们的结果，请使本 Repo 中的 [modeling_llama.py](https://github.com/THUDM/LongAlign/blob/main/modeling_llama.py) 作为模型文件。根据我们论文中的实验结果，我们推荐对 ChatGLM 训练使用 *packing+loss weighting*，对 Llama 训练使用 *sorted batching*。

### 模型部署
我们发布了四个使用 LongAlign 训练的 **chat 模型**：[LongAlign-6B-64k](https://huggingface.co/THUDM/LongAlign-6B-64k)（基于 *ChatGLM3-6B*）、[LongAlign-7B-64k](https://huggingface.co/THUDM/LongAlign-7B-64k)（基于 *Llama-2-7B*）、[LongAlign-13B-64k](https://huggingface.co/THUDM/LongAlign-13B-64k)（基于 *Llama-2-13B*）和 [ChatGLM3-6B-128k](https://huggingface.co/THUDM/chatglm3-6b-128k)。您可以用这个 demo 代码来尝试使用模型来总结我们的论文，或询问有关的任何问题：
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("THUDM/LongAlign-6B-64k", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/LongAlign-6B-64k", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = open("assets/paper.txt").read() + "\n\n请总结这篇论文。"
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=512, temperature=1)
print(response)
```
对于基于 Llama 的模型，我们还提供了 [llama_flash_attn_monkey_patch.py](https://github.com/THUDM/LongAlign/blob/main/LongBench_Chat/llama_flash_attn_monkey_patch.py)，以便在长序列推理时利用 FlashAttention-2 以节省显存。

### 所有可用模型

这里是我们发布的开源模型的完整列表：

| 模型                      | HF 仓库                                                      | 描述                                                     |
| ------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| **LongAlign-6B-64k-base** | [🤗 HF 仓库](https://huggingface.co/THUDM/LongAlign-6B-64k-base) | **ChatGLM3-6B** 上下文窗口扩展到 64k                     |
| **LongAlign-6B-64k**      | [🤗 HF 仓库](https://huggingface.co/THUDM/LongAlign-6B-64k)   | 基于 LongAlign 在 LongAlign-6B-64k-base 上训练的 chat 模型 |
| **LongAlign-7B-64k-base** | [🤗 HF 仓库](https://huggingface.co/THUDM/LongAlign-7B-64k-base) | **Llama-2-7B** 上下文窗口扩展到 64k                      |
|**LongAlign-7B-64k**| [🤗 HF 仓库](https://huggingface.co/THUDM/LongAlign-7B-64k) | 基于 LongAlign 在 LongAlign-7B-64k-base 上训练的 chat 模型 |
|**LongAlign-13B-64k-base**| [🤗 HF 仓库](https://huggingface.co/THUDM/LongAlign-13B-64k-base) | **Llama-2-13B** 上下文窗口扩展到 64k |
|**LongAlign-13B-64k**| [🤗 HF 仓库](https://huggingface.co/THUDM/LongAlign-13B-64k) | 基于 LongAlign 在 LongAlign-13B-64k-base 上训练的 chat 模型 |
|**ChatGLM3-6B-128k**| [🤗 HF 仓库](https://huggingface.co/THUDM/chatglm3-6b-128k) | **ChatGLM3-6B** 上下文窗口扩展到 128k|

<a name="longbench-chat-evaluation"></a>
## 📊 评测

### LongBench-Chat 评测
LongBench-Chat 是首个用于评估长上下文对齐的基准测试，问题都来源于真实用户提问，测试数据长度在 10k-100k 之间。数据集和评估代码在 `LongBench_Chat/` 下。记得在 `eval.py` 中配置您的 OpenAI API 密钥，因为我们采用 GPT-4 作为评估器。运行
```bash
python eval.py --model {model_path} --max_length {max_length}
```
`model_path` 可以是您的本地模型路径或 Hugging Face 模型路径。这是 LongBench-Chat 上的排行榜：

![](assets/leaderboard.png)

我们也欢迎您提交您的模型预测结果或测试结果。我们在计划发布一个更加正式的排行榜。

### 大海捞针试验评测
我们还提供了在“大海捞针”测试下评估HuggingFace模型的代码，位于`Needle_test/`目录下。有关更多信息，请参阅其 [README.md](https://github.com/THUDM/LongAlign/blob/main/Needle_test/README.md)。

*为了复现我们在其他基准测试上的结果，我们推荐使用 [LongBench](https://github.com/THUDM/LongBench)、[FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) 和 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 中的代码来评估 LongBench、MT-Bench 和 Open LLM Leaderboard 中的任务。*

<a name="citation"></a>
## 📝 引用

如果您认为我们的工作有用，请考虑引用 LongAlign：

```
@article{bai2024longalign,
  title={LongAlign: A Recipe for Long Context Alignment of Large Language Models},
  author={Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, Juanzi Li},
  journal={arXiv preprint arXiv:2401.18058},
  year={2024}
}
```
