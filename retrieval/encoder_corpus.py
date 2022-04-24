import os
from os import listdir

import numpy as np
import json
import argparse
import pickle
import torch
from torch._C import device
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from modeling import TextMatchingForBert

def load_json_file(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
        config_dict = json.loads(text)
        return config_dict

class Encoder(object):
    def __init__(self, pretrained_model_path, pool_type='max', max_seq_length=40):
        self.pretrained_model_path = pretrained_model_path
        self.device = torch.device("cpu") if not  torch.cuda.is_available else torch.device("cuda:0")
        self.max_seq_length=max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path, do_lower_case=True, cache_dir=pretrained_model_path, use_fast=False)
        self.model = TextMatchingForBert.from_pretrained(pretrained_model_path, pool_type=pool_type, is_train=False)
        self.model = self.model.eval()
        self.model.to(self.device)

    def predict(self, texts):
        encoded_inputs = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
        input_ids = encoded_inputs["input_ids"]
        features = dict()
        for key in encoded_inputs:
            features[key] = torch.tensor(encoded_inputs[key], device=self.device)
        output = self.model(**features)
        embedding = output.detach().cpu().numpy()
        return embedding
    
    def predict_by_file(self, infile, outfile):
        texts = []
        docids = []
        encoded = []
        indices = []
        count = 0
        with open(infile, 'r') as f:
            for line in f:
                count += 1
                if count % 10000 == 0:
                    print ("Finish: ", count)
                line = line.strip().split('\t')
                doc_id = line[0]
                if len(line) < 2:
                    continue
                if len(line) == 2:
                    text = line[1] # csdn doc file: index \t title
                if len(line) == 3:
                    text = line[1] + ' ' + line[2]
                docids.append(doc_id)
                texts.append(text.lower())
                if len(texts) == 32:
                    embeddings = self.predict(texts)
                    encoded.append(embeddings)
                    indices.extend(docids)
                    texts = []
                    docids = []
            if len(texts) > 0:
                embeddings = self.predict(texts)
                encoded.append(embeddings)
                indices.extend(docids)
        encoded = np.concatenate(encoded)
        with open(outfile, 'wb') as fl:
            pickle.dump((encoded, indices), fl)
        print ("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output_model/video_model/bert-base-ct_cls/", type=str, help="sbert or prop")
    parser.add_argument("--max_sequence_length", default=256, type=int, help="sbert or prop")
    parser.add_argument("--input_file", default="../data/video/corpus.tsv", type=str, help="input file with raw text")
    parser.add_argument("--output_file", default="indices/video/ct_corpus.pt", type=str, help="output file save embeddings")
    parser.add_argument("--pool_type", default="cls", type=str, help="pool type of the final text repsesentation")
    args = parser.parse_args()
    print("----------------------------args: ")
    print(args)
    print(">>>>> input_file: ", args.input_file)
    print(">>>>> output_file: ", args.output_file)
 
    predictor = Encoder(args.model_path, args.pool_type, args.max_sequence_length)
    predictor.predict_by_file(args.input_file, args.output_file)  
