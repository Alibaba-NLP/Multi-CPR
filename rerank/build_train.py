from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
from collections import defaultdict
import datasets
import random
from tqdm import tqdm
import csv

class rankerTrainBuild(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer_name = args.tokenizer_name
        self.qrel_dict = self.read_qrel(args.qrel_file)
        self.query_dict = self.read_query(args.query_file)
        self.corpus_dict = self.read_corpus(args.corpus_file)
        self.retrieval_file = self.args.retrieval_file
        self.ranking_file = self.args.ranking_file        
        self.truncate = self.args.truncate
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)    
    
    def read_qrel(self, qrel_file):
        qrel = {}
        with open(qrel_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                seg = line.split('\t')
                qid, pid = seg[0], seg[2]
                qrel[qid] = pid
        print ("Finish reading qrel dict")
        return qrel

    def read_query(self, query_file):
        query_dict = {}
        with open(query_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                seg = line.split('\t')
                qid, query = seg
                query_dict[qid] = query
        print ("Finish reading query dict")
        return query_dict

    def read_corpus(self, corpus_file):
        corpus_dict = {}
        with open(corpus_file, 'r') as f:
            for line in f:
                line = line.strip()
                seg = line.split('\t')
                doc_id = seg[0]
                content = ' '.join(seg[1:]).strip()
                corpus_dict[doc_id] = content
        print ("Finish reading corpus dict")
        return corpus_dict

    def trans_to_ranking_train(self):
        fl = open(self.ranking_file, 'w')
        recall_dict = defaultdict(list)
        with open(args.retrieval_file, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                seg = line.split('\t')
                qid, pid, score = seg
                recall_dict[qid].append(pid)
                #docs = docs.split(sep_b)
                #scores = scores.split(sep_b)
        for qid, recall_list in recall_dict.items():
        
            golden_pid = self.qrel_dict[qid]
            pos_paassage = self.corpus_dict[golden_pid]
            sample_pids = random.sample(recall_list, 101)
            #sample_docs = docs
            selected_passages = []    
            for pid in sample_pids:
                if pid != golden_pid:
                    psg_content = self.corpus_dict[pid]
                    if len(selected_passages) < 100:
                        selected_passages.append([pid, psg_content])
            
            query_encoded = {}
            # {'qid': '3', 'query': [2044, 766, 13, 5, 2270, 7133, 40715, 16, 1437]}
            query_encoded['qid'] = qid
            query = self.query_dict[qid]
            encoded_query = self.tokenizer.encode(query, add_special_tokens=False, max_length=self.truncate, truncation=True)
            query_encoded['query'] = encoded_query
            
            pos_encoded = []
            encoded_pos = self.tokenizer.encode(pos_paassage, add_special_tokens=False, max_length=self.truncate, truncation=True) 
            pos_encoded.append({
                'passage': encoded_pos,
                'pid': golden_pid,
            })
            neg_encoded = []
            for neg_p in selected_passages:
                pid, passage =  neg_p
                encoded_neg = self.tokenizer.encode(passage, add_special_tokens=False, max_length=self.truncate, truncation=True)
                neg_encoded.append({
                    'passage': encoded_neg,
                    'pid': pid,
                })
            item_set = {
                'qry': query_encoded,
                'pos': pos_encoded,
                'neg': neg_encoded
            }
            fl.write(json.dumps(item_set) + '\n')
        fl.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default="bert-base-chinese", help="bert tokenizer name")
    parser.add_argument('--truncate', type=int, default=256, help="bert tokenizer max sequence length")
    parser.add_argument('--qrel_file', type=str, default="../data/video/qrels.train.tsv", help="qrels train file")
    parser.add_argument('--query_file', type=str, default="../data/video/train.query.txt", help="query train file")
    parser.add_argument('--corpus_file', type=str, default="../data/video/corpus.tsv", help="corpus file")
    parser.add_argument('--retrieval_file', type=str, default="../retrieval/output/video/bert-base-chinese_cls.train.l2.out", help="train query retrieval result")
    parser.add_argument('--ranking_file', type=str, default="train/video/train.group.json", help="ranking train save file")
    args = parser.parse_args()
    rankerTrainBuild = rankerTrainBuild(args)
    rankerTrainBuild.trans_to_ranking_train()
