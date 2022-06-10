"""SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization"""

import functools

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize 
from gensim.parsing.preprocessing import strip_tags

import datasets


_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2005.03724,
  doi = {10.48550/ARXIV.2005.03724},
  url = {https://arxiv.org/abs/2005.03724},
  author = {Gao, Yang and Zhao, Wei and Eger, Steffen},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""


_DESCRIPTION = """\
Unsupervised multi-document summarization evaluation metric.
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: nested list, where each entry is a list of strings containing at least
        one summary of the respective source document(s)
    source_documents: nested_list, where each entry is a list of strings containing at
        least one source document of the respective summaries
    model_type: sbert model to use (default: "bert-large-nli-stsb-mean-tokens")
    top_n: number of first n sentence which are used as pseudo reference (default: 15)
Returns:
    nested list of the supert scores for each summary
Examples:
    >>> supert = datasets.load_metric("supert")
    >>> source_documents = [
        [
            "Long source document about a topic.",
            "Another long source document about the same topic."
        ],
        [
            "Long source document about another topic.",
            "Another document about the second topic."
        ]
    ]
    >>> predictions = [
        ["A summary of the documents about the first topic."],
        [
            "A summary of the documents about the second topic."
            "Another summary of the documents about the second topic."
        ]
    ]
    >>> results = supert.compute(source_documents=source_documents, predictions=predictions)
    >>> print(results)
    [[0.3677717150290458], [0.6020039781738254, 0.6592496765919262]]
"""


LANGUAGE = "english"
    

def get_ref_sents(source_docs, top_n):
    ref_sents = []
    for doc in source_docs:
        ref_sents.append(doc[:top_n])
    return ref_sents

def get_token_vecs(model, sents, remove_stopwords=True):
    vecs, tokens = model.encode(sents, token_vecs=True)
    vecs = functools.reduce(lambda a, b: a+b.tolist(), vecs, [])
    tokens = functools.reduce(lambda a, b: a+b, tokens)
    assert len(vecs) == len(tokens)
    if remove_stopwords:
        clean_vecs = []
        clean_tokens = []
        mystopwords = list(set(stopwords.words(LANGUAGE)))
        mystopwords.extend(["[cls]","[sep]"])
        for i, t in enumerate(tokens):
            if t.lower() not in mystopwords: 
                clean_vecs.append(vecs[i])
                clean_tokens.append(t)
        assert len(clean_vecs) == len(clean_tokens)
        return np.array(clean_vecs)
    return np.array(vecs)

def get_sbert_score(ref_token_vecs, summ_token_vecs):
    f1_list = []
    for i, rvecs in enumerate(ref_token_vecs):
        r_f1_list = []
        for j, svecs in enumerate(summ_token_vecs):
            sim_matrix = cosine_similarity(rvecs, svecs)
            recall = np.mean(np.max(sim_matrix, axis=1))
            precision = np.mean(np.max(sim_matrix, axis=0))
            f1 = 2. * recall * precision / (recall + precision)
            r_f1_list.append(f1)
        f1_list.append(r_f1_list)
    f1_list = np.array(f1_list)
    scores = []
    for i in range(len(summ_token_vecs)):
        scores.append(np.mean(f1_list[:,i]))
    return scores


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Supert(datasets.Metric):
    """SUPERT: Unsupervised multi-document summarization evaluation metric"""

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Value("string")),
                "source_documents": datasets.Sequence(datasets.Value("string"))
            }),
            # Homepage of the metric for documentation
            homepage="https://github.com/yg211/acl20-ref-free-eval",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/yg211/acl20-ref-free-eval"],
            reference_urls=["https://arxiv.org/abs/2005.03724"]
        )

    def _download_and_prepare(self, dl_manager):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    def _compute(self, predictions, source_documents, model_type="bert-large-nli-stsb-mean-tokens", top_n=15):
        """Returns the scores"""
        from sentence_transformers import SentenceTransformer

        assert len(predictions) == len(source_documents), "predictions and source documents need to be nested list of same length"

        predictions = [list(map(lambda x: sent_tokenize(x, LANGUAGE), s)) for s in predictions]
        source_documents = [list(map(lambda x: sent_tokenize(x, LANGUAGE), s)) for s in source_documents]

        model = SentenceTransformer(model_type)
                   
        scores = []
        for i, source_docs in enumerate(tqdm(source_documents)):
            summaries = predictions[i]
            ref_sents = get_ref_sents(source_docs, top_n)
            ref_vecs = []
            for ref in ref_sents:
                ref_vecs.append(get_token_vecs(model, ref))
            summ_vecs = []
            for summ in summaries:
                summ_vecs.append(get_token_vecs(model, summ))
            scores.append(get_sbert_score(ref_vecs, summ_vecs))
        return scores
