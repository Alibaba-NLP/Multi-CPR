# Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval

This repo contains the annotated datasets and expriments implementation introduced in our resource paper in SIGIR2022 Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval. [[Paper]](https://arxiv.org/pdf/2203.03367.pdf).

## 📢 What's New
- 🌟 2023-01: Multiple models fine-tuned with Multi-CPR dataset are open source on the ModelScope platform. [Released Models](#rm1) [开源模型](#rm2)

## Introduction

Multi-CPR is a multi-domain Chinese dataset for passage retrieval. The dataset is collected from three different domains, including E-commerce, Entertainment video and Medical. Each dataset contains millions of passages and a certain amount of human annotated query-passage related pairs.

Examples of annotated query-passage related pairs in three different domains:

|  Domain   | Query  | Passage |
|  ----  | ----  | ---- |
| E-commerce | 尼康z62 (<font color=Blue>Nikon z62</font>) | <div style="width: 150pt"> Nikon/尼康二代全画幅微单机身Z62 Z72 24-70mm套机  (<font color=Blue>Nikon/Nikon II, full-frame micro-single camera, body Z62 Z72 24-70mm set</font>) |
| Entertainment video | 海神妈祖 (<font color=Blue>Ma-tsu, Goddess of the Sea</font>) | 海上女神妈祖 (<font color=Blue>Ma-tsu, Goddess of the Sea</font>) |
| Medical | <div style="width: 150pt"> 大人能把手放在睡觉婴儿胸口吗 (<font color=Blue>Can adults put their hands on the chest of a sleeping baby?</font>) | <div style="width: 150pt"> 大人不能把手放在睡觉婴儿胸口，对孩子呼吸不好，要注意 (<font color=Blue>Adults should not put their hands on the chest of a sleeping baby as this is not good for the baby's breathing.</font>) |

## Data Format

Datasets of each domain share a uniform format, more details can be found in our paper:

- qid: A unique id for each query that is used in evaluation
- pid: A unique id for each passaage that is used in evaluation

| File name | number of record | format |
| ---- | ---- | ---- |
| corpus.tsv | 1002822 | pid, passage content |
| train.query.txt | 100000 | qid, query content |
| dev.query.txt | 1000 | qid, query content |
| qrels.train.tsv  | 100000 | qid, '0', pid, '1' |
| qrels.dev.tsv | 1000 | qid, '0', pid, '1' |
  
## Experiments

The ```retrieval``` and ```rerank``` folders contain how to train a BERT-base dense passage retrieval and reranking model based on Multi-CPR dataset. This code is based on the previous work [tevatron](https://github.com/texttron/tevatron) and [reranker](https://github.com/luyug/Reranker) produced by [luyug](https://github.com/luyug). Many thanks to [luyug](https://github.com/luyug). 

Dense Retrieval Resutls

| Models | Datasets  | Encoder | E-commerce |             | Entertainment video |             | Medical             |             |
|:------:|-----------|---------|------------|-------------|---------------------|-------------|---------------------|-------------|
|        |           |         | MRR@10     | Recall@1000 | MRR@10              | Recall@1000 | MRR@10              | Recall@1000 |
|   DPR  | General   | BERT    | 0.2106     | 0.7750      | 0.1950              | 0.7710      | 0.2133              | 0.5220      |
|  DPR-1 | In-domain | BERT    | 0.2704     | 0.9210      | 0.2537              | 0.9340      | 0.3270              | 0.7470      |
|  DPR-2 | In-domain | BERT-CT | 0.2894     | 0.9260      | 0.2627              | 0.9350      | 0.3388              | 0.7690      |

BERT-reranking results

| Retrieval | Reranker | E-commerce | Entertainment  video | Medical |
|:---------:|:--------:|:----------:|:--------------------:|:-------:|
|           |          |   MRR@10   |        MRR@10        |  MRR@10 |
|   DPR-1   |     -    |   0.2704   |        0.2537        |  0.3270 |
|   DPR-1   |   BERT   |   0.3624   |        0.3772        |  0.3885 |
  
## Requirements
```
python=3.8
transformers==4.18.0
tqdm==4.49.0
datasets==1.11.0
torch==1.11.0
faiss==1.7.0
```

## <span id="rm1">Released Models</span>
We have uploaded some checkpoints finetuned with Multi-CPR to [ModelScope](https://modelscope.cn/home) Model hub. It should be noted that the open-source models on ModelScope are fine-tuned based on the ROM or CoROM model rather than the original BERT model. ROM is a pre-trained language model specially designed for dense passage retrieval task. More details about the ROM model, please refer to paper [ROM](https://arxiv.org/abs/2210.15133)

| Model Type 	| Domain     	| Description 	| Link                                                                                                                                             	|
|------------	|------------	|-------------	|--------------------------------------------------------------------------------------------------------------------------------------------------	|
| Retrieval  	| General    	| -           	| [nlp_corom_sentence-embedding_chinese-base](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base/summary)                 	|
| Retrieval  	| E-commerce 	| -           	| [nlp_corom_sentence-embedding_chinese-base-ecom](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base-ecom/summary)       	|
| Retrieval  	| Medical    	| -           	| [nlp_corom_sentence-embedding_chinese-base-medical](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base-medical/summary) 	|
| ReRanking  	| General    	| -           	| [nlp_rom_passage-ranking_chinese-base](https://modelscope.cn/models/damo/nlp_rom_passage-ranking_chinese-base/summary)                           	|
| ReRanking  	| E-commerce 	| -           	| [nlp_corom_passage-ranking_chinese-base-ecom](https://modelscope.cn/models/damo/nlp_corom_passage-ranking_chinese-base-ecom/summary)             	|
| ReRanking  	| Medical    	| -           	| [nlp_corom_passage-ranking_chinese-base-medical](https://modelscope.cn/models/damo/nlp_corom_passage-ranking_chinese-base-medical/summary) 

  
## <span id="rm2">开源模型</span>

基于Multi-CPR数据集训练的预训练语言模型文本表示(召回)模型、语义相关性(精排)模型已逐步通过[ModelScope平台](https://modelscope.cn/home)开源，欢迎大家下载体验。在ModelScope上开源的模型都是基于ROM或者CoROM模型为底座训练的而不是原始的BERT模型，ROM是一个专门针对文本召回任务设计的预训练语言模型，更多关于ROM模型细节可以参考论文[ROM](https://arxiv.org/abs/2210.15133)

| 模型类别  	| 领域       	| 模型描述                             	| 下载链接                                                                                                                                         	|
|-----------	|------------	|--------------------------------------	|--------------------------------------------------------------------------------------------------------------------------------------------------	|
| Retrieval 	| General    	| 中文通用领域文本表示模型(召回阶段)   	| [nlp_corom_sentence-embedding_chinese-base](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base/summary)                 	|
| Retrieval 	| E-commerce 	| 中文电商领域文本表示模型(召回阶段)   	| [nlp_corom_sentence-embedding_chinese-base-ecom](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base-ecom/summary)       	|
| Retrieval 	| Medical    	| 中文医疗领域文本表示模型(召回阶段)   	| [nlp_corom_sentence-embedding_chinese-base-medical](https://modelscope.cn/models/damo/nlp_corom_sentence-embedding_chinese-base-medical/summary) 	|
| ReRanking 	| General    	| 中文通用领域语义相关性模型(精排阶段) 	| [nlp_rom_passage-ranking_chinese-base](https://modelscope.cn/models/damo/nlp_rom_passage-ranking_chinese-base/summary)                           	|
| ReRanking 	| E-commerce 	| 中文电商领域语义相关性模型(精排阶段) 	| [nlp_corom_passage-ranking_chinese-base-ecom](https://modelscope.cn/models/damo/nlp_corom_passage-ranking_chinese-base-ecom/summary)             	|
| ReRanking 	| Medical    	| 中文医疗领域语义相关性模型(精排阶段) 	| [nlp_corom_passage-ranking_chinese-base-medical](https://modelscope.cn/models/damo/nlp_corom_passage-ranking_chinese-base-medical/summary)       	|
  
## Citing us

If you feel the datasets helpful, please cite:

```  
@inproceedings{Long2022MultiCPRAM,
  author    = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
  title     = {Multi-CPR: {A} Multi Domain Chinese Dataset for Passage Retrieval},
  booktitle = {{SIGIR}},
  pages     = {3046--3056},
  publisher = {{ACM}},
  year      = {2022}
}
```
