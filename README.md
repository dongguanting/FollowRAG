# <div align="center">🔥Toward General Instruction-Following Alignment for Retrieval-Augmented Generation<div>


<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://github.com/dongguanting/IF-RAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>



We propose a instruction-following alignement pipline named **VIF-RAG framework** and auto-evaluation Benchmark named **FollowRAG**:

- **IF-RAG:** It is the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. VIF-RAG integrates a verification process at each step of data augmentation and combination. We begin by manually creating a minimal set of atomic instructions (<100) and then apply steps including instruction composition, quality verification, instruction-query combination, and dual-stage verification to generate a large-scale, high-quality VIF-RAG-QA dataset (>100K). 

- **FollowRAG:** To address the gap in instruction-following auto-evaluation for RAG systems, we introduce FollowRAG Benchmark, which includes approximately 3K test samples, covering 22 categories of general instruction constraints and 4 knowledge-intensive QA datasets. Due to its robust pipeline design, FollowRAG can seamlessly integrate with different RAG benchmarks




<p align="center">
🤖️ <a href="https://followrag.github.io/" target="_blank">Website</a> 🤗 <a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K" target="_blank">VIF-RAG-QA-110K</a> • 👉 <a href="https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K" target="_blank">VIF-RAG-QA-20K</a> • 📖 <a href="https://arxiv.org/pdf/2308.07074.pdf" target="_blank">Paper</a>  <br>
</p>



## News

- [10/2024] 🔥 We released an our SFT datasets named VIF-RAG-QA for deployments. Download [VIF-RAG-QA-110K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K) and [VIF-RAG-QA-20K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K). 

- [10/2024] 🔥 We released our instruction-following auto-evaluation benchmark named ***FollowRAG***. Please follow [guidelines](#FollowRAG) for testing.

- [10/2024] 🔥 We introduced ***VIF-RAG***, the first automated, scalable, and verifiable data synthesis pipeline for aligning complex instruction-following in RAG scenarios. Check out the [paper](). 


## Contents

- [VIF-RAG](#VIF-RAG)
- [FollowRAG](#FollowRAG)
- [Citation](#citation)



# VIF-RAG


We broke down the VIF-RAG data synthesis process into steps and provided 10-20 samples for each step to assist with your reproduction. Be sure to replace these with your own input.

<img width="1243" alt="image" src="https://github.com/user-attachments/assets/d38871d3-d29d-425b-a7d5-d8a7081a110d">

---

### :wrench: Dependencies
General Setup Environment:
- Python 3.9
- [PyTorch](http://pytorch.org/) (currently tested on version 2.1.2+cu121)
- [Transformers](http://huggingface.co/transformers/) (version 4.41.2, unlikely to work lower than this version)

```bash
cd ./VIF-RAG/
pip install -r requirements.txt
```
---

### :rocket: How to Perform *VIF-RAG* Data Synthesis?


Follow the interactive Jupyter notebook VIF-RAG on ``vifrag.ipynb`` to reproduce our experiment on WebQSP.

The script will first convert the KB relations into text sentences.
DPR is then run to select the most relevant relations for each question.
Next, the input to the FiD reader is created for each question using the most relevant relations retrieved by DPR.
Finally, a FiD model can be trained using the UniK-QA input. Our trained FiD checkpoint can be downloaded here. (Our model was trained in late 2020, so you may need to check out an older version of FiD.)


# FollowRAG






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
