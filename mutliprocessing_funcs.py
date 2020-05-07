import os
import json
from others.tokenization import BertTokenizer
import lzma
import os
import json
import re

from pandas.io.json import json_normalize
import pandas as pd
from bs4 import BeautifulSoup
import subprocess
import torch
import lxml

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

def load_json(case_id):
    source = []
    tgt = []
    source_path = os.path.join('presumm_data/tokenized_text/opinions',f'{case_id}.txt.json')
    target_path = os.path.join('presumm_data/tokenized_text/headnotes',f'{case_id}.txt.json')
    for sent in json.load(open(source_path,encoding='utf-8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        tokens = [t.lower() for t in tokens]
        source.append(tokens)
    for sent in json.load(open(target_path,encoding='utf-8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        tokens = [t.lower() for t in tokens]
        tgt.append(tokens)


    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


import re


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    # abstract = abstract_sent_list
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    # print(evaluated_1grams)
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []

    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1

        for i in range(len(sents)):
            if (i in selected):
                continue

            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

max_src_nsents =10000
class BertData():
    def __init__(self, min_src_ntokens_per_sent=5,
                max_src_ntokens_per_sent=200,
                max_src_nsents=max_src_nsents,
                min_src_nsents=1,
                max_tgt_ntokens=500,
                min_tgt_ntokens=5):
        self.min_src_ntokens_per_sent = min_src_ntokens_per_sent
        self.max_src_ntokens_per_sent = max_src_ntokens_per_sent
        self.max_src_nsents = max_src_nsents
        self.min_src_nsents = min_src_nsents
        self.max_tgt_ntokens = max_tgt_ntokens
        self.min_tgt_ntokens = min_tgt_ntokens
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.max_src_nsents]
        sent_labels = sent_labels[:self.max_src_nsents]

        if ((not is_test) and len(src) < self.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

def generate_bert_data(case_id):
    source, tgt = load_json(case_id)
    sent_labels = greedy_selection(source[:max_src_nsents], tgt, 5)
    source = [' '.join(s).lower().split() for s in source]
    tgt = [' '.join(s).lower().split() for s in tgt]
    bert = BertData()
    b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=True,
                                     is_test=False)
    if b_data is not None:
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                               "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                               'src_txt': src_txt, "tgt_txt": tgt_txt}
        return (case_id,b_data_dict)
