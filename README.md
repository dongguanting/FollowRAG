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

## 💥 News

- [10/2024] 🔥 We released an our SFT datasets named VIF-RAG-QA for deployments. Download [VIF-RAG-QA-110K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K) and [VIF-RAG-QA-20K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K). 

- [10/2024] 🔥 We released our instruction-following auto-evaluation benchmark named ***FollowRAG***. Please follow [outlines](#FollowRAG) for testing.

- [10/2024] 🔥 We introduced ***VIF-RAG***, the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. Check out the [paper](). 

---

## Outlines
- [News](https://github.com/dongguanting/FollowRAG/blob/main/README.md#-news)
- [VIF-RAG](https://github.com/dongguanting/FollowRAG/blob/main/README.md#-vif-rag-)
- [FollowRAG](https://github.com/dongguanting/FollowRAG/blob/main/README.md#-followrag-)
- [Citation](https://github.com/dongguanting/FollowRAG/blob/main/README.md#-citation-)


---

## 🌠 VIF-RAG


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

## 🐋 FollowRAG 

FollowRAG is the first benchmark designed to comprehensively evaluate LLM’s complex instruction-following abilities in RAG. 

<img width="1070" alt="image" src="https://github.com/user-attachments/assets/91a5e7ac-d828-46f2-bcae-96886f7ef295">


### :wrench: Dependencies
General Setup Environment:
- Python 3.9

```bash
cd ./FollowRAG/
pip install -r requirements.txt
```


### 📊 Test Cases

<details>
<summary>🔍 Click here! if you are curious about FollowRAG‘s test cases.</summary>

**Key-Value Introduction:**

- **prompt:** The complete question for FollowRAG, including three parts: TopK Document + user query + instruction
- **question:** QA question (sourced from NQ)
- **answer_gold:** Reference answer (note that this is not the golden answer, as the answer needs to follow instruction constraints after adding instructions)
- **question_with_instrs:** QA question + a series of instruction constraints
- **instruction_id_list & kwargs:** Instruction types and parameters needed for evaluation calculation
- **passages:** TopK documents retrieved from Wiki using DPR



```json

    {
        "key": 0,
        "type": "ifnq",
        "prompt": "Given the following information: \nPassage-0 Title: Gravity Content: and prevents further acceleration. The force of gravity on Earth is the resultant (vector sum) of two forces: (a) The gravitational attraction in accordance with Newton's universal law of gravitation, and (b) the centrifugal force, which results from the choice of an earthbound, rotating frame of reference. The force of gravity is the weakest at the equator because of the centrifugal force caused by the Earth's rotation and because points on the equator are furthest from the center of the Earth. The force of gravity varies with latitude and increases from about 9.780 m/s at the Equator to about 9.832\nPassage-1 Title: Gravitational acceleration Content: Gravitational acceleration In physics, gravitational acceleration is the acceleration on an object caused by the force of gravitation. Neglecting friction such as air resistance, all small bodies accelerate in a gravitational field at the same rate relative to the center of mass. This equality is true regardless of the masses or compositions of the bodies. At different points on Earth, objects fall with an acceleration between and depending on altitude and latitude, with a conventional standard value of exactly 9.80665 m/s (approximately 32.174 ft/s). This does not take into account other effects, such as buoyancy or drag. Newton's law of\nPassage-2 Title: Gravity Content: Gravity Gravity (), or gravitation, is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or \"gravitate\" toward) one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming starsand for the stars to group together into galaxiesso gravity is responsible for many of the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. Gravity\n\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: What is the common name for gravitational force? In this task, repeat the exact request first, then give your response. Do not say any word before repeating the exact request. Moreover, your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>. Ensure the word disappointed appears at least twice. Finally, provide your answer with less than 200 words.",
        "question": "what is the common name for gravitational force",
        "answer_gold": "Gravity/Gravity, or gravitation",
        "question_with_instrs": "What is the common name for gravitational force? In this task, repeat the exact request first, then give your response. Do not say any word before repeating the exact request. Moreover, your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>. Ensure the word disappointed appears at least twice. Finally, provide your answer with less than 200 words.",
        "instruction_id_list": [
            "combination:repeat_prompt",
            "detectable_format:title",
            "keywords:frequency",
            "length_constraints:number_words"
        ],
        "kwargs": [
            {
                "prompt_to_repeat": "What is the common name for gravitational force?"
            },
            {},
            {
                "relation": "at least",
                "keyword": "disappointed",
                "frequency": 2
            },
            {
                "relation": "less than",
                "num_words": 200
            }
        ],
        "passages": [
            {
                "title": "Gravity",
                "content": "and prevents further acceleration. The force of gravity on Earth is the resultant (vector sum) of two forces: (a) The gravitational attraction in accordance with Newton's universal law of gravitation, and (b) the centrifugal force, which results from the choice of an earthbound, rotating frame of reference. The force of gravity is the weakest at the equator because of the centrifugal force caused by the Earth's rotation and because points on the equator are furthest from the center of the Earth. The force of gravity varies with latitude and increases from about 9.780 m/s at the Equator to about 9.832"
            },
            {
                "title": "Gravitational acceleration",
                "content": "Gravitational acceleration In physics, gravitational acceleration is the acceleration on an object caused by the force of gravitation. Neglecting friction such as air resistance, all small bodies accelerate in a gravitational field at the same rate relative to the center of mass. This equality is true regardless of the masses or compositions of the bodies. At different points on Earth, objects fall with an acceleration between and depending on altitude and latitude, with a conventional standard value of exactly 9.80665 m/s (approximately 32.174 ft/s). This does not take into account other effects, such as buoyancy or drag. Newton's law of"
            },
            {
                "title": "Gravity",
                "content": "Gravity Gravity (), or gravitation, is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or \"gravitate\" toward) one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming starsand for the stars to group together into galaxiesso gravity is responsible for many of the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. Gravity"
            }
        ]
    }
```
</details>




### 🔑 Inference
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
Please refer to the following template to prepare your result JSON files for subsequent evaluation. 
Your data_inferenced.jsonl format should be consistent with the following form:

```json

#TODO SXS

```


### 📝 Evaluation
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


## 📜 License

Our dataset are distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.



## 🎖 Citation 

Please cite our work if you find the repository helpful.

```

```


