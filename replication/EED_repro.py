import datasets, os
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

from geneval.data.wmt import WMT17, WMT18

def kendall_score(scores_better, scores_worse):
    total = len(scores_better)
    correct = np.sum(np.array(scores_better) > np.array(scores_worse))
    incorrect = total - correct
    return (correct - incorrect)/total

eed = datasets.load_metric("geneval/metrics/ExtendedEditDistance/eed.py")

#----------WMT17----------
lang_pairs = ["cs-en", "de-en", "fi-en", "lv-en", "ru-en", "tr-en", "zh-en"]
for lang_pair in lang_pairs:
    print(f"Started language pair {lang_pair}...", end="")
    # load data
    wmt = WMT17(lang_pair)
    # compute scores
    scores = eed.compute(
        predictions=wmt.translations,
        references=wmt.references
    )
    df = pd.DataFrame({
        "translation": wmt.translations,
        "references": wmt.references,
        "source": wmt.sources,
        "human": wmt.scores,
        "metric_score": scores
    })
    df.to_csv(os.path.join(os.getcwd(), "reproduction", "results", "wmt17", "eed", lang_pair + ".csv"), index=False)
    print("completed.")

results = {}
for lang_pair in lang_pairs:
    df = pd.read_csv(os.path.join(os.getcwd(), "reproduction", "results", "wmt17", "eed", lang_pair + ".csv"))
    corr = pearsonr(df["metric_score"], df["human"])[0]
    results[lang_pair] = corr
    print(f"{lang_pair}: {corr}")

#----------WMT18----------
results = {}
for lang_pair in ["cs-en", "de-en", "et-en", "fi-en", "ru-en", "tr-en", "zh-en"]:
    print(f"Started language pair {lang_pair}...")
    # load data
    wmt = WMT18(lang_pair, root="/tmp")
    # compute scores
    scores_better = eed.compute(
        predictions=wmt.translations_better,
        references=wmt.references
    )
    print("     ...scores_better completed,")
    scores_worse = eed.compute(
        predictions=wmt.translations_worse,
        references=wmt.references
    )
    print("     ...scores_worse completed,")
    ks = kendall_score(scores_worse, scores_better)
    print(f"     ...ks = {ks}")
    results[lang_pair] = ks
print(results)