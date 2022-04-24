# Dense Passage Retrieval

Code for training the dense passage retrieval model, building ANN index, query retrieval process and evaluation

## Training

### Data preprocessing

```
usage: create_train.py [-h] [--qrels_file QRELS_FILE] [--query_file QUERY_FILE] [--collection_file COLLECTION_FILE] [--save_to SAVE_TO] [--tokenizer_name TOKENIZER_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --qrels_file QRELS_FILE
                        qrels file
  --query_file QUERY_FILE
                        query file
  --collection_file COLLECTION_FILE
                        collections file
  --save_to SAVE_TO     processd train json file
  --tokenizer_name TOKENIZER_NAME
                        pretrained model tokenizer
```

The format of preprocessd file
```
{"spans": [["QUERY_TOKENIZED"], ["PASSAGE_TOKENIZED"]]}
{"spans": [["QUERY_TOKENIZED"], ["PASSAGE_TOKENIZED"]]}
```

Run training
```
sh run_train.sh
```

### Encoding corpus
```
usage: encoder_corpus.py [-h] [--model_path MODEL_PATH] [--max_sequence_length MAX_SEQUENCE_LENGTH] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE] [--pool_type POOL_TYPE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        sbert or prop
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        sbert or prop
  --input_file INPUT_FILE
                        input file with raw text
  --output_file OUTPUT_FILE
                        output file save embeddings
  --pool_type POOL_TYPE
                        pool type of the final text repsesentation
```

### Searching
```
usage: retrieval.py [-h] [--model_path MODEL_PATH] [--pool_type POOL_TYPE] [--max_sequence_length MAX_SEQUENCE_LENGTH] [--index_file INDEX_FILE] [--topk TOPK] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --pool_type POOL_TYPE
                        pool type of the final text repsesentation
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        use pb model or tf checkpoint
  --index_file INDEX_FILE
  --topk TOPK
  --input_file INPUT_FILE
  --output_file OUTPUT_FILE
```

### Evaluation
```
usage: evaluate.py [-h] [--result_path RESULT_PATH] [--qrel_path QREL_PATH] [--reverse REVERSE] [--topk TOPK] [--topk_list TOPK_LIST]

optional arguments:
  -h, --help            show this help message and exit
  --result_path RESULT_PATH
                        search result
  --qrel_path QREL_PATH
  --reverse REVERSE     reverse score during sorting
  --topk TOPK
  --topk_list TOPK_LIST
```
