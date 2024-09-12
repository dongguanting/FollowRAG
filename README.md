# <div align="center">⚡Toward General Instruction-Following Alignment for Retrieval-Augmented Generation<div>


<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://github.com/dongguanting/IF-RAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>



We propose a instruction-following alignement pipline named **VIF-RAG framework** and auto-evaluation Benchmark named **FollowRAG**:

- IF-RAG: It is the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. VIF-RAG integrates a verification process at each step of data augmentation and combination. We begin by manually creating a minimal set of atomic instructions (<100) and then apply steps including instruction composition, quality verification, instruction-query combination, and dual-stage verification to generate a large-scale, high-quality VIF-RAG-QA dataset (>100K). 

- FollowRAG: To address the gap in instruction-following auto-evaluation for RAG systems, we introduce FollowRAG Benchmark, which includes approximately 3K test samples, covering 22 categories of general instruction constraints and 4 knowledge-intensive QA datasets. Due to its robust pipeline design, FollowRAG can seamlessly integrate with different RAG benchmarks




<p align="center">
🤖️ <a href="https://followrag.github.io/" target="_blank">Website</a> 🤗 <a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K" target="_blank">VIF-RAG-QA-110K</a> • 👉 <a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K" target="_blank">VIF-RAG-QA-20K</a> • 📖 <a href="https://arxiv.org/pdf/2308.07074.pdf" target="_blank">Paper</a>  <br>
</p>



## News

- [10/2024] 🔥 We released an our SFT datasets named VIF-RAG-QA for deployments. Download [VIF-RAG-QA-110K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K) and [VIF-RAG-QA-20K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K). 

- [10/2024] 🔥 We released our instruction-following auto-evaluation benchmark named ***FollowRAG***. Please follow [guidelines](#FollowRAG) for testing.

- [10/2024] 🔥 We introduced ***VIF-RAG***, the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. Check out the [paper](). 



## VIF-RAG Framework

**What is *InsTag*?**

Foundation language models obtain the instruction-following ability through supervised fine-tuning (SFT).
Diversity and complexity are considered critical factors of a successful SFT dataset, while their definitions remain obscure and lack quantitative analyses.
In this work, we propose *InsTag*, an open-set fine-grained tagger, to tag samples within SFT datasets based on semantics and intentions and define instruction diversity and complexity regarding tags.
We obtain 6.6K tags to describe comprehensive user queries.
We analyze popular open-sourced SFT datasets and find that the model ability grows with more diverse and complex data.
Based on this observation, we propose a data selector based on *InsTag* to select 6K diverse and complex samples from open-source datasets and fine-tune models on *InsTag*-selected data.
These models outperform open-source models based on considerably larger SFT data evaluated by MT-Bench, echoing the importance of query diversity and complexity.

<p align="center" width="100%">
<a ><img src="assets/main_figure.png" alt="InsTag" style="width: 80%; min-width: 300px; display: block; margin: auto;"></a>
</p>




**What is *VIF-RAG*?**

we propose VIF-RAG, the first automated, scalable, and verifiable data synthesis pipeline for achieving complex instruction-following alignment in RAG scenarios. The core insight of VIF-RAG is to ensure every step of data augmentation and combination includes a proper verification process. Specifically, we start by manually crafting a minimal set of atomic instructions ($<$100) and developing combination rules to synthesize and verify complex instructions for a seed set. We then use supervised models for instruction rewriting. Motivated by tool execution studies~\citep{le2022coderl,qiao2024making}, we employ the same supervised model to generate verification code and automatically verify the quality of augmented instructions through the Python compiler's outputs. Finally, we combine these high-quality instructions with RAG datasets from various domains (each containing retrieved documents per query), performing the augmentation and dual validation process to synthesize a high-quality instruction-based RAG dataset, named VIF-RAG-QA ($>$100K samples).





## Contents

- [Model Checkpoints](#model-checkpoints)
- [Citation](#citation)

## InsTagger

InsTagger is a LLaMa-2 based SFT model trained with FastChat in the vicuna template. You can easily download weight at [HuggingFace ModelHub](https://huggingface.co/OFA-Sys/InsTagger) and then use [FastChat](https://github.com/lm-sys/FastChat) to serve or inference. Demo codes are about to be released.

## Model Checkpoints

- **InsTagger** for local query tagging:

    **InsTagger** is an tagging LLM which is fine-tuned on **InsTag**'s tagging results on open-resourced SFT data. The model is based on 7B version LLaMA-2.

    Download the model checkpoint below:

    | Model | Checkpoint | Exact Match F1 | Semantic-based Fuzzy Match F1  | License |
    | ----- |------| -------| -------| ----- |
    | LocalTagger | 🤗 <a href="" target="_blank">HF Link</a>  | **31.8%** | **73.4%**  | <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">LLaMA 2 License </a> |




- **TagLM**, fine-tuned on our SFT data sub-sampled by complexity-first diverse sampling procedure:

    With only 6k data from current open-resourced SFT dataset, **TagLM** can outperform many open-resourced LLMs on MT-Bench using GPT-4 as a judge. 

    Download the model checkpoint below:

    | Model | Checkpoint | MT-Bench  | License |
    | ----- |------| -------| ----- |
    | TagLM-13B-v1.0 | 🤗 <a href="" target="_blank">HF Hub Link</a>  |  **6.44**	  | <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">LLaMA License </a> |
    | TagLM-13B-v2.0 | 🤗 <a href="" target="_blank">HF Hub Link</a>  |  **6.55**	  | <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">LLaMA 2 License </a> |

    All models are either based on LLaMA or LLaMA-2 and should be used under their licenses accordingly. All the models are fine-tuned using [FastChat](https://github.com/lm-sys/FastChat) codebase, and we apply the system template of Vicuna V1.1. 


## Citation 

Please cite our work if you find the repository helpful.

```
@misc{lu2023instag,
      title={#InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models}, 
      author={Keming Lu and Hongyi Yuan and Zheng Yuan and Runji Lin and Junyang Lin and Chuanqi Tan and Chang Zhou and Jingren Zhou},
      year={2023},
      eprint={2308.07074},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



https://drive.google.com/drive/folders/1dCCpAVPiwPgjOhuKGcyonwgfr2kntJHZ?usp=sharing
