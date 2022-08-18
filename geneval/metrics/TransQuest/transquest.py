import datasets

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2106.00143,
  doi = {10.48550/ARXIV.2106.00143},
  
  url = {https://arxiv.org/abs/2106.00143},
  
  author = {Ranasinghe, Tharindu and Orasan, Constantin and Mitkov, Ruslan},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {An Exploratory Analysis of Multilingual Word-Level Quality Estimation with Cross-Lingual Transformers},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

_DESCRIPTION = """\
Sentence-level quality estimation of machine translations with cross-lingual transformers.
"""

_KWARGS_DESCRIPTION = """
Calculates QE of a translation.
Args:
    source: string of words separated by spaces, representing the original text
    target: string with words separated by spaces, representing the translation
Returns:
    transquest QE score
Examples:
    >>> transquest = datasets.load_metric("transquest")
    >>> source = "Reducerea acestor conflicte este importantă pentru conservare."
    >>> target = "Reducing these conflicts is not important for preservation."
    >>> score = supert.compute(source, target)
    >>> print(score)
    array(0.81065178)
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TransQuest(datasets.Metric):
    """TransQuest: Word-level quality estimation of machine translations with cross-lingual transformers"""

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'source': datasets.Value('string'),
                'translation': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self):
        pass

    def _compute(self, source, target):
        """Returns the score"""
        from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

        model = MonoTransQuestModel("xlmroberta", "TransQuest/monotransquest-da-multilingual", num_labels=1, use_cuda=True)

        predictions, raw_outputs = model.predict([[source, target]])

        return predictions