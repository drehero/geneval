import numpy as np


def kendall_score(scores_better, scores_worse):
    # https://github.com/Tiiiger/bert_score/blob/master/reproduce/get_wmt18_seg_results.py#L53
    total = len(scores_better)
    correct = np.sum(np.array(scores_better) > np.array(scores_worse))
    incorrect = total - correct
    return (correct - incorrect)/total
