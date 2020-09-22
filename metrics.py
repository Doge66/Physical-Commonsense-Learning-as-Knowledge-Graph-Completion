# this script is modified from Forbes et al., 2019

import numpy as np

def cms(y_hat, y, y_labels):
    y_hat = y_hat.squeeze()
    y = y.squeeze()

    res = {}
    for i in range(len(y_labels[0].split("/"))):
        res[i] = {"overall": np.zeros((2, 2)), "per-item": {}}

    for i, y_label in enumerate(y_labels):
        want = y[i]
        got = y_hat[i]
        subgroups = y_label.split("/")
        for j, item in enumerate(subgroups):
            res[j]["overall"][want][got] += 1
            if item not in res[j]["per-item"]:
                res[j]["per-item"][item] = np.zeros((2, 2))
            res[j]["per-item"][item][want][got] += 1

    return res


def prf1(cm):
    """Returns (precision, recall, f1) from a provided 2x2 confusion matrix.

    We special case a few F1 situations where the F1 score is technically undefined or
    pathological. For example, if there are no 1s to predict, 1.0 is returned for
    p/r/f1.
    """
    # cm: cm[i][j] is number truly in group i but predicted to be in j
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    total_1s = tp + fn

    # precision undefined if tp + fp == 0, i.e. no 1s were predicted.
    if tp + fp == 0:
        # precision should not be penalized for not predicting anything.
        precision = 1.0
    else:
        # normal precision
        precision = tp / (tp + fp)

    # recall undefined if tp + fn == total_1s == 0
    if total_1s == 0:
        # couldn't have predicted any 1s because there weren't any. should not
        # penalize recall.
        recall = 1.0
    else:
        # normal recall
        recall = tp / (tp + fn)

    # f1 undefined if precision + recall == 0.
    if precision + recall == 0:
        # if precision and recall are both 0, f1 should just be 0.
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def _report(y_hat, y, y_labels, task_labels):
    # accuracy
    acc = (y_hat == y).sum() / len(y)
    txt = ["Acc: {:.3f}".format(acc)]

    # get cms
    category_cms = cms(y_hat, y, y_labels)

    # micro f1 is the same for any category, because it's the sum of the confusion
    # matrices. we pick category 0 arbitrarily.
    micro_precision, micro_recall, micro_f1 = prf1(category_cms[0]["overall"])
    txt.append("Micro F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}".format(micro_f1, micro_precision, micro_recall))

    # macro f1 is a bit more involved.
    macro_f1s = []
    for i, results in category_cms.items():
        sum_p, sum_r = 0.0, 0.0
        n = 0
        for cm in results["per-item"].values():
            # don't count "all-0" items towards the category macro. (i.e., skip if total
            # 1s = tp + fn = 0)
            if cm[1][1] + cm[1][0] == 0:
                continue
            precision, recall, _ = prf1(cm)
            sum_p += precision
            sum_r += recall
            n += 1

        macro_precision = sum_p / n
        macro_recall = sum_r / n
        macro_f1 = (
            0
            if macro_precision == 0 and macro_recall == 0
            else (
                2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
            )
        )
        # macro_f1s[task_labels[i]] = macro_f1
        macro_f1s.append(macro_f1)
        txt.append("{} macro F1: {:.3f}, macro precision: {:.3f}, macro recall: {:.3f}".format(
            task_labels[i], macro_f1, macro_precision, macro_recall))

    return txt, acc, micro_f1, macro_f1s[0], macro_f1s[1]

def report(y_hat, y, y_labels, task_labels):
    txt,_,_,_,_ = _report(y_hat, y, y_labels, task_labels)

    return txt

def report_more4cv(y_hat, y, y_labels, task_labels):
    _,acc,micro_f1,macro_f11,macro_f12 = _report(y_hat, y, y_labels, task_labels)
    return acc,micro_f1,macro_f11,macro_f12
