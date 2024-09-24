# <div align="center">🔥Toward General Instruction-Following Alignment for Retrieval-Augmented Generation<div>


<div align="center">
<a href="" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://github.com/dongguanting/IF-RAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>



We propose a instruction-following alignement pipline named **VIF-RAG framework** and auto-evaluation Benchmark named **FollowRAG**:

- **IF-RAG:** It is the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. VIF-RAG integrates a verification process at each step of data augmentation and combination. We begin by manually creating a minimal set of atomic instructions (<100) and then apply steps including instruction composition, quality verification, instruction-query combination, and dual-stage verification to generate a large-scale, high-quality VIF-RAG-QA dataset (>100K). 

- **FollowRAG:** To address the gap in instruction-following auto-evaluation for RAG systems, we introduce FollowRAG Benchmark, which includes approximately 3K test samples, covering 22 categories of general instruction constraints and 4 knowledge-intensive QA datasets. Due to its robust pipeline design, FollowRAG can seamlessly integrate with different RAG benchmarks




<p align="center">
🤖️ <a href="https://followrag.github.io/" target="_blank">Website</a> • 🤗 <a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K" target="_blank">VIF-RAG-QA-110K</a> • 👉 <a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K" target="_blank">VIF-RAG-QA-20K</a> • 📖 <a href="https://arxiv.org/pdf/2308.07074.pdf" target="_blank">Paper</a>  <br>
</p>

---

## News

- [10/2024] 🔥 We released an our SFT datasets named VIF-RAG-QA for deployments. Download [VIF-RAG-QA-110K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K) and [VIF-RAG-QA-20K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K). 

- [10/2024] 🔥 We released our instruction-following auto-evaluation benchmark named ***FollowRAG***. Please follow [outlines](#FollowRAG) for testing.

- [10/2024] 🔥 We introduced ***VIF-RAG***, the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. Check out the [paper](). 

---

## Contents

- [VIF-RAG](#VIF-RAG)
- [FollowRAG](#FollowRAG)
- [Citation](#citation)

---

## VIF-RAG


We broke down the VIF-RAG data synthesis process into steps and provided 10-20 samples for each step to assist with your reproduction. Be sure to replace these with your own input.

<img width="1243" alt="image" src="https://github.com/user-attachments/assets/d38871d3-d29d-425b-a7d5-d8a7081a110d">



### :wrench: Dependencies
General Setup Environment:
- Python 3.9
- [PyTorch](http://pytorch.org/) (currently tested on version 2.1.2+cu121)
- [Transformers](http://huggingface.co/transformers/) (version 4.41.2, unlikely to work lower than this version)

```bash
cd ./VIF-RAG/
pip install -r requirements.txt
```

### :rocket: How to Perform *VIF-RAG* Data Synthesis?


Follow the interactive Jupyter notebook VIF-RAG on ``vifrag.ipynb`` to reproduce our synthesize dataset.


### 🎯 Training

We use the version of [LlaMA-Factory v0.6.3](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.6.3). Thanks for their excellent work.

we also release our SFT version dataset as strong baseline in Table1:
- **SFT Version:** To make a fair comparison with VIF-RAG, we use the same amount of [🤗ShareGPT](https://huggingface.co/datasets/dongguanting/ShareGPT-12K) and [🤗RAG-QA-40K](https://huggingface.co/datasets/dongguanting/RAG-QA-40K) as in VIF-RAG’s data synthesis process, mixing them together to fine-tune (SFT) different baseline models.

- **VIF-RAG-QA:** We release our SFT datasets, including [🤗VIF-RAG-QA-110K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K) and [🤗VIF-RAG-QA-20K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K).


- **SFT bash:**
  
```bash
deepspeed --num_gpus=8 train_bash.py \
        --deepspeed $deepspeed_zero3_config_path \
        --stage sft \
        --do_train \
        --use_fast_tokenizer \
        --flash_attn \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --model_name_or_path $MODEL_PATH \
        --dataset $dataset \
        --template $Template \
        --finetuning_type full \
        --output_dir $OUTPUT_PATH \
        --overwrite_cache \
        --overwrite_output_dir \
        --warmup_steps 20 \
        --weight_decay 0.1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --ddp_timeout 9000 \
        --learning_rate 7e-6 \
        --lr_scheduler_type "linear" \
        --logging_steps 1 \
        --cutoff_len 8192 \
        --save_steps 200 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --bf16 
```

---

## FollowRAG

### inference
You first need to perform inference on followRAG, and the pseudocode is as follows:
```python
followRAG_full=load_json('followRAG/followRAG_full.json')
data_inferenced=[]
for dp in followRAG_full:
    response=llm.inference(dp['prompt'])
    dp['response']=response
    data_inferenced.append(dp)
save_jsonl(data_inferenced,'results/finish_inference/data_inferenced.jsonl')
```

### eval
After completing the inference, run the evaluation script:
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_API_BASE=https://api.openai.com/v1
python eval/main_eval.py \
    --input_file_path results/finish_inference/data_inferenced.jsonl \
    --output_file_path results/finish_eval/data_evaled.jsonl \
    --rag_eval_type mini \
    --result_log_file_path results/logs/results_log.jsonl
```
Our evaluation of instruction-following part largely draws on the [IFEval code repository](https://github.com/google-research/google-research/tree/master/instruction_following_eval). We appreciate their excellent work!





## Citation 

Please cite our work if you find the repository helpful.

```

```


