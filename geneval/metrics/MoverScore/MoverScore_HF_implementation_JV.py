# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#imports for MoverScore
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
import os
from pyemd import emd, emd_with_flow
from torch import nn
from math import log
from itertools import chain, zip_longest

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial


from transformers import AutoTokenizer, AutoModel

from typing import List, Union, Iterable



# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {MoverScore},
authors={Zhao, Wei
, Peyrard, Maxime
, Liu, Fei
, Gao, Yang
, Meyer, Christian M. 
, Eger, Steffen },
year={2021}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
MoverScore is built upon a combination of  (i) contextualized representations of system and 
reference texts and (ii) a distance between these representations measuring the semantic distance 
between system outputs and references
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    Moverscore: Value in the interval [0,1]
Examples:
    refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
            ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
        
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    
    >>> MoverScore = datasets.load_metric("HF_MoverScore.py")
    >>> results = MoverScore._compute(sys,refs)
    >>> print(results)
    0.7390671673481917
"""

# create Moverscore metric
import datasets

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
model.eval()

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MoverScore(datasets.Metric):
    """Moverscore assigns a single holistic score to any system-generated text by comparing
       it against human references for content matching. MoverScore is built upon a combination of 
       (i) contextualized representations of system and reference texts and (ii) a distance between 
       these representations measuring the semantic distance between system outputs and references
    """

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="https://github.com/drehero/gen-eval-metrics",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )
    
    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        pass
    
    def truncate(self, tokens):
        if len(tokens) > tokenizer.model_max_length - 2:
            tokens = tokens[0:(tokenizer.model_max_length - 2)]
        return tokens

    def process(self, a):
        a = ["[CLS]"]+self.truncate(tokenizer.tokenize(a))+["[SEP]"]
        a = tokenizer.convert_tokens_to_ids(a)
        return set(a)


    def get_idf_dict(self, arr, nthreads=4):
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(self.process)

        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

        idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
        idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def padding(self, arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask

    def bert_encode(self, model, x, attention_mask):
        model.eval()
        with torch.no_grad():
            result = model(x, attention_mask = attention_mask)
        if model_name == 'distilbert-base-uncased':
            return result[1] 
        else:
            return result[2] 

    #with open('stopwords.txt', 'r', encoding='utf-8') as f:
    #    stop_words = set(f.read().strip().split(' '))

    def collate_idf(self, arr, tokenize, numericalize, idf_dict,
                    pad="[PAD]"):
        
        tokens = [["[CLS]"]+self.truncate(tokenize(a))+["[SEP]"] for a in arr]  
        arr = [numericalize(a) for a in tokens]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]
        
        pad_token = numericalize([pad])[0]

        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding(idf_weights, pad_token, dtype=torch.float)

        return padded, padded_idf, lens, mask, tokens

    def get_bert_embedding(self, all_sens, model, tokenizer, idf_dict,
                           batch_size=-1):

        padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens,
                                                          tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                          idf_dict)

        if batch_size == -1: batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode(model, padded_sens[i:i+batch_size],
                                              attention_mask=mask[i:i+batch_size])
                batch_embedding = torch.stack(batch_embedding)
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    def _safe_divide(self, numerator, denominator):
        return numerator / (denominator + 1e-30)

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.baddbmm(
            x2_norm.transpose(-2, -1),
            x1,
            x2.transpose(-2, -1),
            alpha=-2
        ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
        return res

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords = True, batch_size=256):
        preds = []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start:batch_start+batch_size]
            batch_hyps = hyps[batch_start:batch_start+batch_size]
            
            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs, model, tokenizer, idf_dict_ref)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, model, tokenizer, idf_dict_hyp)

            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]
            
            batch_size = len(ref_tokens)
            for i in range(batch_size):  
                ref_ids = [k for k, w in enumerate(ref_tokens[i]) 
                                    if w in stop_words or '##' in w 
                                    or w in set(string.punctuation)]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) 
                                    if w in stop_words or '##' in w
                                    or w in set(string.punctuation)]
              
                ref_embedding[i, ref_ids,:] = 0                        
                hyp_embedding[i, hyp_ids,:] = 0
                
                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0
                
            raw = torch.cat([ref_embedding, hyp_embedding], 1)
                                 
            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
            
            distance_matrix = self.batched_cdist_l2(raw, raw).double().cpu().numpy()
                    
            for i in range(batch_size):  
                c1 = np.zeros(raw.shape[1], dtype=np.float)
                c2 = np.zeros(raw.shape[1], dtype=np.float)
                c1[:len(ref_idf[i])] = ref_idf[i]
                c2[len(ref_idf[i]):] = hyp_idf[i]
                
                c1 = self._safe_divide(c1, np.sum(c1))
                c2 = self._safe_divide(c2, np.sum(c2))
                
                dst = distance_matrix[i]
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=np.float32)
                score = 1./(1. + np.sum(flow * dst))#1 - np.sum(flow * dst)
                preds.append(score)

        return preds
    def sentence_score(self, hypothesis: str, references: List[str], trace=0):
    
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
    
        hypothesis = [hypothesis] * len(references)
    
        sentence_score = 0 

        scores = self.word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
        sentence_score = np.mean(scores)
    
        if trace > 0:
            print(hypothesis, references, sentence_score)
            
        return sentence_score

    def corpus_score(self, sys_stream: List[str],
                     ref_streams:Union[str, List[Iterable[str]]], trace=0):

        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        fhs = [sys_stream] + ref_streams
    
        corpus_score = 0
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")
            
            hypo, *refs = lines
            corpus_score += self.sentence_score(hypo, refs, trace=0)
        
        corpus_score /= len(sys_stream)

        return corpus_score

    def _compute(self, predictions, references):

        return self.corpus_score(predictions, references)