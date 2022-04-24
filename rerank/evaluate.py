import argparse
from collections import defaultdict

class evaluation():
    def __init__(self, qrel_path, result_path, reverse=True):
        self.qrel_path = qrel_path
        self.result_path = result_path
        self.qrels = self.read_qrels(self.qrel_path)
        self.results = self.read_results(self.result_path)
        self.reverse = reverse

    def read_qrels(self, qrel_path):
        qrels = {} 
        with open(qrel_path) as f:
            for line in f:
                line = line.strip().split('\t')
                qid = line[0]
                pid = line[2]
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(pid)
        return qrels

    def read_results(self, result_path):
        results = {}
        with open(result_path) as f:
            for line in f:
                line = line.strip().split()
                # 200000 Q0 220237 0 -3.8359375 deepct1
                qid, _, pid, _, score, _ = line
                if qid not in results:
                    results[qid] = [(pid, float(score))]
                else:
                    results[qid].append((pid, float(score)))
                #for pid, score in zip(pids, scores):
                #    results[qid].append((pid, float(score)))
        return results

    def eval(self, topk):
        mrr = 0
        count = 0
        no_judged = 0
        for qid in self.results:
            if qid not in self.qrels:
                no_judged += 1
                continue
            res = self.results[qid]
            ar = 0
            sorted_res = sorted(res, key = lambda x:x[1], reverse=self.reverse)
            # print(sorted_res[:10])
            for i, ele in enumerate(sorted_res):
                pid = ele[0]
                if i >= topk:
                    break
                if pid in self.qrels[qid]:
                    ar = 1.0 / (i+1)
                    break
            mrr += ar
            if ar > 0:
                count += 1
        tot = len(self.results) - no_judged
        return mrr / tot, float(count) / tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default="result/video_bert_base_rank_res", type=str, help="search result")
    parser.add_argument('--qrel_path', default='../data/video/qrels.dev.tsv', type=str)
    parser.add_argument('--reverse', default=True, type=bool, help="reverse score during sorting, true for sparse and false for dense")
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--topk_list', default=[], type=list) 
    args = parser.parse_args()
    evaluation = evaluation(args.qrel_path, args.result_path, args.reverse)
    mrr, recall = evaluation.eval(args.topk)
    print (args.topk, mrr, recall)
