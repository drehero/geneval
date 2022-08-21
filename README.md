# Evaluation Metrics for Natural Language Generation 

We build upon the [huggingface datasets](https://github.com/huggingface/datasets) library which already implements BERTScore, COMET, Frugalscore and BLEURT.
These metrics can be loaded using the name of the metric:

```python
import datasets

metric = datasets.load_metric("metric_name")
```

All other metrics implemented by us can be loaded using the path to the implementation script:


```python
import datasets

metric = datasets.load_metric("path/to/metric_script.py")
```

We also implement dataloaders for the WMT16-21 datasets.
Find all metrics and datasets plus detailed usage examples below.



## Metrics

### BERTScore

#### BERTScore: Evaluating Text Generation with BERT

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi

[[Paper]](https://arxiv.org/abs/1904.09675) [[Code]](https://pypi.org/project/bert-score/)

#### Usage 

```python
import datasets

bertscore = datasets.load_metric("bertscore")
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = bertscore.compute(predictions=predictions, references=references, lang="en")
```


### MoverScore

#### MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance

Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger

[[Paper]](https://arxiv.org/abs/1909.02622) [[Code]](https://github.com/AIPHES/emnlp19-moverscore)

#### Usage

```python
import datasets

moverscore = datasets.load_metric("geneval/metrics/moverscore/moverscore.py")
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = moverscore.compute(predictions=predictions, references=references)
```


### FrugalScore

#### FrugalScore: Learning Cheaper, Lighter and Faster Evaluation Metrics for Automatic Text Generation

Moussa Kamal Eddine, Guokan Shang, Antoine J.-P. Tixier, Michalis Vazirgiannis

[[Paper]](https://arxiv.org/abs/2110.08559v1) [[Code]](https://github.com/moussaKam/FrugalScore)

#### Usage

```python
import datasets

frugalscore = datasets.load_metric("frugalscore")
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = frugalscore.compute(predictions=predictions, references=references)
```


### BaryScore

#### Automatic Text Evaluation through the Lens of Wasserstein Barycenters

Pierre Colombo, Guillaume Staerman, Chloe Clavel, Pablo Piantanida

[[Paper]](https://arxiv.org/abs/2108.12463) [[Code]](https://github.com/PierreColombo/nlg_eval_via_simi_measures/)

#### Usage

```python
import datasets

baryscore = datasets.load_metric("geneval/metrics/baryscore/baryscore.py")
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = baryscore.compute(predictions=predictions, references=references)
```


### COMET


#### COMET: A Neural Framework for MT Evaluation

Ricardo Rei, Craig Stewart, Ana C Farinha, Alon Lavie

[[Paper]](https://arxiv.org/abs/2009.09025) [[Code]](https://github.com/Unbabel/COMET)

#### Usage: 

```python
import datasets

comet = datasets.load_metric('comet')
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
results = comet.compute(predictions=hypothesis, references=reference, sources=source)
```

### BLEURT

#### BLEURT: Learning Robust Metrics for Text Generation

Thibault Sellam, Dipanjan Das, Ankur P. Parikh

[[Paper]](https://arxiv.org/abs/2004.04696) [[Code]](https://github.com/google-research/bleurt)

#### Usage

```python
import datasets

bleurt = datasets.load_metric("bleurt")
redictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = bleurt.compute(predictions=predictions, references=references)
```


### RoME

#### RoMe: A Robust Metric for Evaluating Natural Language Generation

Md Rashad Al Hasan Rony, Liubov Kovriguina, Debanjan Chaudhuri, Ricardo Usbeck, Jens Lehmann

[[Paper]](https://arxiv.org/abs/2203.09183) [[Code]](https://github.com/rashad101/RoMe)

#### Usage

```python
import datasets

rome = datasets.load_metric("geneval/metrics/RoMe/rome.py")
redictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = rome.compute(predictions=predictions, references=references)
```

### TransQuest

#### TransQuest: Translation Quality Estimation with Cross-lingual Transformers

Tharindu Ranasinghe, Constantin Orasan, Ruslan Mitkov

[[Paper]](https://arxiv.org/abs/2011.01536) [[Code]](https://github.com/TharinduDR/TransQuest)

```python
import datasets

transquest = datasets.load_metric("geneval/metrics/TransQuest/transquest.py")
source = "Today the weather is very nice."
target = "Heute ist das Wetter sehr gut."
metric.add(source=source, target=target)
results = transquest.compute()
```


### SUPERT

#### SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization

Yang Gao, Wei Zhao, Steffen Eger

[[Paper]](https://arxiv.org/abs/2005.03724) [[Code]](https://github.com/yg211/acl20-ref-free-eval)

#### Usage

```python
supert = datasets.load_metric("geneval/geneval/supert/supert.py")
source_documents = [
	[
		"Long source document about a topic.",
		"Another long source document about the same topic."
	],
	[
		"Long source document about another topic.",
		"Another document about the second topic."
	]
]
predictions = [
	["A summary of the documents about the first topic."],
	[
		"A summary of the documents about the second topic."
		"Another summary of the documents about the second topic."
	]
]
results = supert.compute(source_documents=source_documents, predictions=predictions)
```


### EED

#### EED: Extended Edit Distance Measure for Machine Translation

Peter Stanchev, Weiyue Wang, Hermann Ney

[[Paper]](https://aclanthology.org/W19-5359/) [[Code]]()

#### Usage

```python
import datasets

eed = datasets.load_metric("geneval/metrics/ExtendedEditDistance/eed.py")
hypothesis = ["Today the weather is very nice."]
references = ["Today the weather is not nice."]
results = eed.compute(predictions=hypothesis, references=references)
```


### BARTScore 

#### BARTSCORE: Evaluating Generated Text as Text Generation 

Weizhe Yuan, Graham Neubig, Pengfei Liu

[[Paper]](https://arxiv.org/pdf/2106.11520.pdf) [[Code]](https://github.com/neulab/BARTScore)

#### Usage

```python
import datasets

bartscore = datasets.load_metric("geneval/metrics/bartscore/bartscore.py")
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
results = bartscore.compute(predictions=hypothesis, sources=source)
```


## Datasets

We implement the [WMT16-21](https://www.statmt.org/) datasets.
For WMT16 and WMT17 we implement segment level direct assessment scores and for WMT18-WMT21 relative rankings.

### Example Usage

Datasets with direct assessment scores:

```python
from geneval.data.wmt import WMT16

wmt16 = WMT16(
	lang_pair="cs-en",
	root="/tmp",
	download=True
)
sources = wmt16.sources
references = wmt16.references
translations = wmt16.translations
scores = wmt16.scores
```

Datasets with direct assessment relative rankings:


```python
from geneval.data.wmt import WMT18

wmt18 = WMT18(
	lang_pair="cs-en",
	root="/tmp",
	download=True
)
sources = wmt18.sources
references = wmt18.references
translations_better = wmt18.translations_better
translations_worse = wmt18.translations_worse
```
