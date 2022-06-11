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

from typing import List, Union, Iterable
from itertools import zip_longest
#import sacrebleu
from moverscore_v2 import word_mover_score
from collections import defaultdict
import numpy as np

def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score


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
    
    def _compute(sys_stream: List[str],
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
            corpus_score += sentence_score(hypo, refs, trace=0)
        
        corpus_score /= len(sys_stream)

        return corpus_score