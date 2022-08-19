import datasets

_CITATION = """\
@inproceedings{stanchev-etal-2019-eed,
    title = "{EED}: Extended Edit Distance Measure for Machine Translation",
    author = "Stanchev, Peter  and
      Wang, Weiyue  and
      Ney, Hermann",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5359",
    doi = "10.18653/v1/W19-5359",
    pages = "514--520",
    abstract = "Over the years a number of machine translation metrics have been developed in order to evaluate the accuracy and quality of machine-generated translations. Metrics such as BLEU and TER have been used for decades. However, with the rapid progress of machine translation systems, the need for better metrics is growing. This paper proposes an extension of the edit distance, which achieves better human correlation, whilst remaining fast, flexible and easy to understand.",
}
"""

_DESCRIPTION = """\
EED utilises the Levenshtein distance and extends it by adding an additional jump operation.
"""

_KWARGS_DESCRIPTION = """
Calculates EED of a machine translation compared to a reference.
Args:
    predictions: string of words separated by spaces, representing the hypothesis by a machine
    references: string with words separated by spaces, representing the reference/true text
Returns:
    EED score
Examples:
    >>> eed = datasets.load_metric("eed")
    >>> predictions = "Reducing these conflicts is not important for preservation."
    >>> references = "Reducing these conflicts is very important for preservation."
    >>> score = eed.compute(predictions, references)
    >>> print(score)
    0.10094637223974764
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EED(datasets.Metric):
    """EED: The metric utilises the Levenshtein distance and extends it by adding an additional jump operation."""

    def _info(self):
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
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        pass

    def _compute(self, predictions, references):
        """Returns the score"""

        def distance(refWord, hypWord):
            if refWord == hypWord:
                return 0
            else:
                return 1

        def preprocess_en(s):
            import re
            if isinstance(s, list):
                if len(s) > 1:
                    s = "".join(s)
                else:
                    s = s[0]

            s = s.rstrip()  # trail space, tab, newlineの削除

            s = s.replace('.', ' .')
            s = s.replace('!', ' !')
            s = s.replace('?', ' ?')
            s = s.replace(',', ' ,')

            s = re.sub(r'\s+', r' ', s)  # スペースの個数正規化
            s = re.sub(r'(\d) ([.,]) (\d)', r'\1\2\3', s)  # 0 . 1 -> 0.1
            s = re.sub(r'(Dr|Jr|Prof|Rev|Gen|Mr|Mt|Mrs|Ms) .', r'\1.', s)  # Mr . -> Mr.
            s = s.replace(u'e . g .', u'e.g.')
            s = s.replace(u'i . e .', u'e.g.')
            s = s.replace(u'U . S .', u'U.S.')
            return s

        scores = []
        for pred, ref in zip(predictions, references):
            hyp = preprocess_en(pred)
            ref = preprocess_en(ref)
            hyp = list(hyp)
            ref = list(ref)

            hyp.insert(0, " ")
            hyp.append(" ")
            ref.insert(0, " ")
            ref.append(" ")
            alpha = 2.0
            deletion = 0.2
            # substitutions are implemented via the distance function
            insertion = 1.0
            rho = 0.3
            lj = [-1] * (len(hyp) + 1)
            row = [1] * (len(hyp) + 1)  # row[i] stores cost of cheapest path from (0,0) to (i,l) in CDER aligment grid.
            row[0] = 0  # CDER initialisation 0,0 = 0 rest 1
            nextRow = [float('inf')] * (len(hyp) + 1)
            for w in range(1, len(ref) + 1):
                for i in range(0, len(hyp) + 1):
                    if i > 0:
                        nextRow[i] = min(nextRow[i - 1] + deletion, row[i - 1] + distance(ref[w - 1], hyp[i - 1]),
                                         row[i] + insertion)
                    else:
                        nextRow[i] = row[i] + 1.0
                minInd = nextRow.index(min(nextRow))
                lj[minInd] += 1
                # Long Jumps
                if ref[w - 1] == " ":
                    jump = alpha + nextRow[minInd]
                    nextRow = [x if x < jump else jump for x in nextRow]
                row = nextRow
                nextRow = [float('inf')] * (len(hyp) + 1)
            coverage = rho * sum([x if x >= 0 else 1 for x in lj])
            score = min(1, (row[-1] + coverage) / (float(len(ref)) + coverage))
            scores.append(score)
        return scores
