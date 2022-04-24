from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
from collections import defaultdict
import datasets
import random
from tqdm import tqdm
import csv

class rankerDevBuild(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer_name = args.tokenizer_name
        self.qrel_dict = self.read_qrel(args.qrel_file)
        self.query_dict = self.read_query(args.query_file)
        self.corpus_dict = self.read_corpus(args.corpus_file)
        self.retrieval_file = self.args.retrieval_file
        self.ranking_file = self.args.ranking_file 
        self.label_file = self.args.label_file       
        self.truncate = self.args.truncate
        self.topk = self.args.topk
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

    def trans_to_ranking_dev(self):
        fl = open(self.ranking_file, 'w')
        fd = open(self.label_file, 'w')
        recall_dict = defaultdict(list)
        with open(args.retrieval_file, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                seg = line.split('\t')
                qid, pid, score = seg
                recall_dict[qid].append([pid, score])
 
        for qid, recall_list in recall_dict.items():
            query = self.query_dict[qid]
            golden_pid = self.qrel_dict[qid]
            encoded_query = self.tokenizer.encode(query, add_special_tokens=False, max_length=self.truncate, truncation=True)
            for i, item in enumerate(recall_list[0: self.topk]):
                pid, score = item
                psg_content = self.corpus_dict[pid]
                encoded_psg = self.tokenizer.encode(psg_content, add_special_tokens=False, max_length=self.truncate, truncation=True)
                item_set = {
                    'qid': qid,
                    'pid': pid,
                    'qry': encoded_query,
                    'psg': encoded_psg
                }
                if pid == golden_pid:
                    label = '1'
                else:
                    label = '0'
                tmp = [qid, 'Q0', pid, str(i+1), score, label]
                fd.write(' '.join(tmp).strip() + '\n')
                fl.write(json.dumps(item_set) + '\n')

        fl.close()
        fd.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default="bert-base-chinese", help="bert tokenizer name")
    parser.add_argument('--truncate', type=int, default=256, help="bert tokenizer max sequence length")
    parser.add_argument('--qrel_file', type=str, default="../data/video/qrels.dev.tsv", help="qrels train file")
    parser.add_argument('--query_file', type=str, default="../data/video/dev.query.txt", help="query train file")
    parser.add_argument('--corpus_file', type=str, default="../data/video/corpus.tsv", help="corpus file")
    parser.add_argument('--topk', type=int, default=1000, help="select topk result")
    parser.add_argument('--retrieval_file', type=str, default="../retrieval/output/video/bert-base-chinese_cls.dev.l2.out", help="train query retrieval result")
    parser.add_argument('--ranking_file', type=str, default="dev/video/dev.top1000.json", help="ranking train save file")
    parser.add_argument('--label_file', type=str, default="dev/video/dev.top1000.label.txt", help="ranking train save file")
    args = parser.parse_args()
    rankerDevBuild = rankerDevBuild(args)
    rankerDevBuild.trans_to_ranking_dev()
