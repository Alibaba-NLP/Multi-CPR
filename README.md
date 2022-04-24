# Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval

This repo contains the annotated datasets and expriments implementation introduced in our resource paper in SIGIR2022 Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval. [[Paper]](https://arxiv.org/pdf/2203.03367.pdf).

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

## Citing us

If you feel the datasets helpful, please cite:

```
@article{Long2022MultiCPRAM,
  title={Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
  author={Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Rui Guo and Jianfeng Xu and Guanjun Jiang and Luxi Xing and P. Yang},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series = {SIGIR 22},
  year={2022}
}
```
