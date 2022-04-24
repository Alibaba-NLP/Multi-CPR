import faiss
import json
import argparse
import numpy as np
import time
import os
import pickle
import torch
from itertools import chain
from tqdm import tqdm
from faiss import normalize_L2

from collections import defaultdict
from encoder_corpus import Encoder 

class Searcher(object):
    def __init__(self, encoder, index_file, topk):
        self.index_file = index_file
        self.encoder = encoder
        self.topk = topk
        self.build_index()

    def build_index(self):
        p_reps_0, p_lookup_0 = self.pickle_load(self.index_file)
        #faiss.normalize_L2(p_reps_0)
        self.look_up = p_lookup_0 
        index = faiss.IndexFlatL2(p_reps_0.shape[1])
        #p_reps_0 = p_reps_0 / np.linalg.norm(p_reps_0, ord=2, axis=1, keepdims=True)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, index) # index
        #shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, []))
        #for p_reps, p_lookup in shards:
        self.add(p_reps_0)        
        
    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def pickle_load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def searcher(self, query):
        qvec = self.encoder.predict([query])
        scores, indices = self.index.search(qvec, self.topk)
        return scores[0], [self.look_up[x] for x in indices[0]]

    def read_query_file(self, input_file):
        query_lst = []
        with open(input_file) as f:
            for line in f:
                line = line.strip().split("\t")
                query_id = line[0]
                query = line[1]
                query_lst.append((query_id, query))
        return query_lst

    def process(self, inputFile, outputFile):
        query_lst = self.read_query_file(inputFile)
        print(">>> Now is searching ... ")
        fw = open(outputFile, "w")
        for queryl in tqdm(query_lst):
            query_id, query = queryl
            scores, indices = self.searcher(query)
            #print ("scores: ", scores)
            #print ("indices: ", indices)
            for score, indice in zip(scores, indices):
                fw.write('\t'.join([query_id, indice, str(score)]).strip() + '\n')
        fw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output_model/video_model/bert-base-ct_cls/", type=str)
    parser.add_argument("--pool_type", default="cls", type=str, help="pool type of the final text repsesentation")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="use pb model or tf checkpoint")
    parser.add_argument("--index_file", default="indices/video/ct_corpus.pt", type=str)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--input_file", default="../data/video/dev.query.txt", type=str)
    parser.add_argument("--output_file", default="output/video/bert-base-ct_cls.dev.l2.out", type=str)
    args = parser.parse_args()
    print("args: ", args)
    encoder = Encoder(args.model_path, args.pool_type, args.max_sequence_length)
    searcher = Searcher(encoder, args.index_file, args.topk)
    searcher.process(args.input_file, args.output_file) 
