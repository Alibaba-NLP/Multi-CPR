# BERT-base Passage Reranking

Code for training bert-base passage reranking model. Our code is developed based on the [reranker](https://github.com/luyug/Reranker) framework, so, you need to put the code in [```Reranker/src/```](https://github.com/luyug/Reranker/tree/main/src) in ```'./'``` for reproduce the result.

## Training

- Data preprocessing

building train and dev dataset
```
usage: build_train.py [-h] [--tokenizer_name TOKENIZER_NAME] [--truncate TRUNCATE] [--qrel_file QREL_FILE] [--query_file QUERY_FILE] [--corpus_file CORPUS_FILE] [--retrieval_file RETRIEVAL_FILE] [--ranking_file RANKING_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --tokenizer_name TOKENIZER_NAME
                        bert tokenizer name
  --truncate TRUNCATE   bert tokenizer max sequence length
  --qrel_file QREL_FILE
                        qrels train file
  --query_file QUERY_FILE
                        query train file
  --corpus_file CORPUS_FILE
                        corpus file
  --retrieval_file RETRIEVAL_FILE
                        train query retrieval result
  --ranking_file RANKING_FILE
                        ranking train save file
```

```
usage: build_dev.py [-h] [--tokenizer_name TOKENIZER_NAME] [--truncate TRUNCATE] [--qrel_file QREL_FILE] [--query_file QUERY_FILE] [--corpus_file CORPUS_FILE] [--topk TOPK] [--retrieval_file RETRIEVAL_FILE] [--ranking_file RANKING_FILE] [--label_file LABEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --tokenizer_name TOKENIZER_NAME
                        bert tokenizer name
  --truncate TRUNCATE   bert tokenizer max sequence length
  --qrel_file QREL_FILE
                        qrels train file
  --query_file QUERY_FILE
                        query train file
  --corpus_file CORPUS_FILE
                        corpus file
  --topk TOPK           select topk result
  --retrieval_file RETRIEVAL_FILE
                        train query retrieval result
  --ranking_file RANKING_FILE
                        ranking train save file
  --label_file LABEL_FILE
                        ranking train save file
```

- Training
```
sh run_train.sh
```

## Inference

Inference the ranking score of each query-passage pair (here, we only rerank the top1000 retrieval passages for each query)
```
sh inference.sh
```

## Evaluation
```
usage: evaluate.py [-h] [--topk_path TOPK_PATH] [--qrel_path QREL_PATH] [--topk TOPK]

optional arguments:
  -h, --help            show this help message and exit
  --topk_path TOPK_PATH
  --qrel_path QREL_PATH
  --topk TOPK
```
