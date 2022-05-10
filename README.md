# Evaluation Metrics Library

## Research Questions
* Can we provide a unified access point to different evaluation metrics allowing easy access for different users?
* Can we also provide the majority of relevant datasets in the same environment?
* How can we use this information for reproducibility?

## Tasks
* Search Aclanthology for latest metrics
* Collect metrics and datasets
* Conduct reproducibility study

## Literature:
* Reproducibility Issues for BERT-based Evaluation Metrics
* Quantifying Reproducibility in NLP and ML

## Metrics

### Model Augmented Relevance Score (MARS)
#### Authors 
Ruibo Liu, Jason Wei, and Soroush Vosoughi. 2021

#### Abstract
Although automated metrics are commonly used to evaluate NLG systems, they often correlate poorly with human judgements. Newer metrics such as BERTScore have addressed many weaknesses in prior metrics such as BLEU and ROUGE, which rely on n-gram matching. These newer methods, however, are still limited in that they do not consider the generation context, so they cannot properly reward generated text that is correct but deviates from the given reference. In this paper, we propose Language Model Augmented Relevance Score (MARS), a new context-aware metric for NLG evaluation. MARS leverages off-the-shelf language models, guided by reinforcement learning, to create augmented references that consider both the generation context and available human references, which are then used as additional references to score generated text. Compared with seven existing metrics in three common NLG tasks, MARS not only achieves higher correlation with human reference judgements, but also differentiates well-formed candidates from adversarial samples to a larger degree.

Paper: https://aclanthology.org/2021.acl-long.521.pdf, data can be found here:  https://aclanthology.org/2021.acl-long.521/ 
Code:


####### Metrics from the slides

### BERTScore
#### Authors
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi

#### Abstract
We propose BERTSCORE, an automatic evaluation metric for text generation.
Analogously to common metrics, BERTSCORE computes a similarity score for
each token in the candidate sentence with each token in the reference sentence.
However, instead of exact matches, we compute token similarity using contextual
embeddings. We evaluate using the outputs of 363 machine translation and image
captioning systems. BERTSCORE correlates better with human judgments and
provides stronger model selection performance than existing metrics. Finally, we
use an adversarial paraphrase detection task to show that BERTSCORE is more
robust to challenging examples when compared to existing metrics.

[[Paper]](https://arxiv.org/abs/1904.09675) [[Code]](https://pypi.org/project/bert-score/)

### MoverScore
#### Authors
Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger

#### Abstract
A robust evaluation metric has a profound impact on the development of text generation systems. A desirable metric compares system output against references based on their semantics rather than surface forms. In this paper we investigate strategies to encode system and reference texts to devise a metric that shows a high correlation with human judgment of text quality. We validate our new metric, namely MoverScore, on a number of text generation tasks including summarization, machine translation, image captioning, and data-to-text generation, where the outputs are produced by a variety of neural and non-neural systems. Our findings suggest that metrics combining contextualized representations with a distance measure perform the best. Such metrics also demonstrate strong generalization capability across tasks. For ease-of-use we make our metrics available as web service.

[[Paper]](https://arxiv.org/abs/1909.02622) [[Code]](https://github.com/AIPHES/emnlp19-moverscore)

### FrugalScore
#### Authors
Moussa Kamal Eddine, Guokan Shang, Antoine J.-P. Tixier, Michalis Vazirgiannis

#### Abstract
 Fast and reliable evaluation metrics are key to R&D progress. While traditional natural language generation metrics are fast, they are not very reliable. Conversely, new metrics based on large pretrained language models are much more reliable, but require significant computational resources. In this paper, we propose FrugalScore, an approach to learn a fixed, low cost version of any expensive NLG metric, while retaining most of its original performance. Experiments with BERTScore and MoverScore on summarization and translation show that FrugalScore is on par with the original metrics (and sometimes better), while having several orders of magnitude less parameters and running several times faster. On average over all learned metrics, tasks, and variants, FrugalScore retains 96.8% of the performance, runs 24 times faster, and has 35 times less parameters than the original metrics. We make our trained metrics publicly available, to benefit the entire NLP community and in particular researchers and practitioners with limited resources.

[[Paper]](https://arxiv.org/abs/2110.08559v1) [[Code]](https://github.com/moussaKam/FrugalScore)

### BaryScore
#### Authors
Pierre Colombo, Guillaume Staerman, Chloe Clavel, Pablo Piantanida

#### Abstract
A new metric BaryScore to evaluate text generation based on deep contextualized embeddings e.g., BERT, Roberta, ELMo) is introduced. This metric is motivated by a new framework relying on optimal transport tools, i.e., Wasserstein distance and barycenter. By modelling the layer output of deep contextualized embeddings as a probability distribution rather than by a vector embedding; this framework provides a natural way to aggregate the different outputs through the Wasserstein space topology. In addition, it provides theoretical grounds to our metric and offers an alternative to available solutions e.g., MoverScore and BertScore). Numerical evaluation is performed on four different tasks: machine translation, summarization, data2text generation and image captioning. Our results show that BaryScore outperforms other BERT based metrics and exhibits more consistent behaviour in particular for text summarization.

[[Paper]](https://arxiv.org/abs/2108.12463) [[Code]](https://github.com/PierreColombo/nlg_eval_via_simi_measures/)

### COMET
#### Authors
Ricardo Rei, Craig Stewart, Ana C Farinha, Alon Lavie

#### Abstract
We present COMET, a neural framework for training multilingual machine translation evaluation models which obtains new state-of-the-art levels of correlation with human judgements. Our framework leverages recent breakthroughs in cross-lingual pretrained language modeling resulting in highly multilingual and adaptable MT evaluation models that exploit information from both the source input and a target-language reference translation in order to more accurately predict MT quality. To showcase our framework, we train three models with different types of human judgements: Direct Assessments, Human-mediated Translation Edit Rate and Multidimensional Quality Metrics. Our models achieve new state-of-the-art performance on the WMT 2019 Metrics shared task and demonstrate robustness to high-performing systems.

[[Paper]](https://arxiv.org/abs/2009.09025) [[Code]](https://github.com/Unbabel/COMET)

### BLEURT
#### Authors
Thibault Sellam, Dipanjan Das, Ankur P. Parikh

#### Abstract
Text generation has made significant advances in the last few years. Yet, evaluation metrics have lagged behind, as the most popular choices (e.g., BLEU and ROUGE) may correlate poorly with human judgments. We propose BLEURT, a learned evaluation metric based on BERT that can model human judgments with a few thousand possibly biased training examples. A key aspect of our approach is a novel pre-training scheme that uses millions of synthetic examples to help the model generalize. BLEURT provides state-of-the-art results on the last three years of the WMT Metrics shared task and the WebNLG Competition dataset. In contrast to a vanilla BERT-based approach, it yields superior results even when the training data is scarce and out-of-distribution.

[[Paper]](https://arxiv.org/abs/2004.04696) [[Code]](https://github.com/google-research/bleurt)

### RoME
#### Authors
Md Rashad Al Hasan Rony, Liubov Kovriguina, Debanjan Chaudhuri, Ricardo Usbeck, Jens Lehmann

#### Abstract
Evaluating Natural Language Generation (NLG) systems is a challenging task. Firstly, the metric should ensure that the generated hypothesis reflects the reference's semantics. Secondly, it should consider the grammatical quality of the generated sentence. Thirdly, it should be robust enough to handle various surface forms of the generated sentence. Thus, an effective evaluation metric has to be multifaceted. In this paper, we propose an automatic evaluation metric incorporating several core aspects of natural language understanding (language competence, syntactic and semantic variation). Our proposed metric, RoMe, is trained on language features such as semantic similarity combined with tree edit distance and grammatical acceptability, using a self-supervised neural network to assess the overall quality of the generated sentence. Moreover, we perform an extensive robustness analysis of the state-of-the-art methods and RoMe. Empirical results suggest that RoMe has a stronger correlation to human judgment over state-of-the-art metrics in evaluating system-generated sentences across several NLG tasks.

[[Paper]](https://arxiv.org/abs/2203.09183) [[Code]](https://github.com/rashad101/RoMe)

### UScore
#### Authors
Jonas Belouadi, Steffen Eger

#### Abstract
 The vast majority of evaluation metrics for machine translation are supervised, i.e., (i) assume the existence of reference translations, (ii) are trained on human scores, or (iii) leverage parallel data. This hinders their applicability to cases where such supervision signals are not available. In this work, we develop fully unsupervised evaluation metrics. To do so, we leverage similarities and synergies between evaluation metric induction, parallel corpus mining, and MT systems. In particular, we use an unsupervised evaluation metric to mine pseudo-parallel data, which we use to remap deficient underlying vector spaces (in an iterative manner) and to induce an unsupervised MT system, which then provides pseudo-references as an additional component in the metric. Finally, we also induce unsupervised multilingual sentence embeddings from pseudo-parallel data. We show that our fully unsupervised metrics are effective, i.e., they beat supervised competitors on 4 out of our 5 evaluation datasets.

[[Paper]](https://arxiv.org/abs/2202.10062) [[Code]]()

### TransQuest
#### Authors
Tharindu Ranasinghe, Constantin Orasan, Ruslan Mitkov

#### Abstract
Recent years have seen big advances in the field of sentence-level quality estimation (QE), largely as a result of using neural-based architectures. However, the majority of these methods work only on the language pair they are trained on and need retraining for new language pairs. This process can prove difficult from a technical point of view and is usually computationally expensive. In this paper we propose a simple QE framework based on cross-lingual transformers, and we use it to implement and evaluate two different neural architectures. Our evaluation shows that the proposed methods achieve state-of-the-art results outperforming current open-source quality estimation frameworks when trained on datasets from WMT. In addition, the framework proves very useful in transfer learning settings, especially when dealing with low-resourced languages, allowing us to obtain very competitive results.

[[Paper]](https://arxiv.org/abs/2011.01536) [[Code]](https://github.com/TharinduDR/TransQuest)

### SUPERT
#### Authors
Yang Gao, Wei Zhao, Steffen Eger

#### Abstract
We study unsupervised multi-document summarization evaluation metrics, which require neither human-written reference summaries nor human annotations (e.g. preferences, ratings, etc.). We propose SUPERT, which rates the quality of a summary by measuring its semantic similarity with a pseudo reference summary, i.e. selected salient sentences from the source documents, using contextualized embeddings and soft token alignment techniques. Compared to the state-of-the-art unsupervised evaluation metrics, SUPERT correlates better with human ratings by 18-39%. Furthermore, we use SUPERT as rewards to guide a neural-based reinforcement learning summarizer, yielding favorable performance compared to the state-of-the-art unsupervised summarizers.

[[Paper]](https://arxiv.org/abs/2005.03724) [[Code]](https://github.com/yg211/acl20-ref-free-eval)

## Datasets

