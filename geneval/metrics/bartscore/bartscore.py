"""BARTScore: Evaluating Generated Text as Text Generation"""

import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
import datasets


_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2106.11520,
  doi = {10.48550/ARXIV.2106.11520},
  url = {https://arxiv.org/abs/2106.11520},
  author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {BARTScore: Evaluating Generated Text as Text Generation},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Zero v1.0 Universal}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    sources: list of source documents for each prediction. Each
        reference should be a string with tokens separated by spaces.
    model_type: model checkpoint (default: facebook/bart-large-cnn)
    max_length: maximum sequence length (default: 1024)
    batch_size: batch size (default: 4)
Returns:
    bart score
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> bart_score = datasets.load_metric("bart_score")
    >>> results = bart_score.compute(['This is interesting.'], ['This is fun.'], batch_size=4) 
    >>> print(results)
    [-2.510652780532837]
"""

class BARTScorer:
    def __init__(self,  max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    #def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
    #    # Assert we have the same number of references
    #    ref_nums = [len(x) for x in tgts]
    #    if len(set(ref_nums)) > 1:
    #        raise Exception("You have different number of references per test sample.")

    #    ref_num = len(tgts[0])
    #    score_matrix = []
    #    for i in range(ref_num):
    #        curr_tgts = [x[i] for x in tgts]
    #        scores = self.score(srcs, curr_tgts, batch_size)
    #        score_matrix.append(scores)
    #    if agg == "mean":
    #        score_list = np.mean(score_matrix, axis=0)
    #    elif agg == "max":
    #        score_list = np.max(score_matrix, axis=0)
    #    else:
    #        raise NotImplementedError
    #    return list(score_list)




@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NewMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'sources': datasets.Value('string'),
                'model_type': dataset.Value('int'),
                'max_length': datasets.Value('int'),
                'batch_size': datasets.Value('int'),
            }),
            # Homepage of the metric for documentation
            homepage="https://github.com/neulab/BARTScore",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/neulab/BARTScore"],
            reference_urls=["https://arxiv.org/abs/2106.11520"]
        )

    def _compute(
        self,
        predictions,
        sources,
        model_type="facebook/bart-large-cnn",
        max_length=1024,
        batch_size=4
    ):
        """Returns the scores"""
        scorer = BARTScorer(checkpoint=model_type, max_length=max_length)
        return scorer.score(srcs=sources, tgts=predictions)
