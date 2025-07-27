import numpy as np

def top_k_accuracy(true_labels, predicted_lists, k):
    correct = sum(1 for tl, preds in zip(true_labels, predicted_lists) if tl in preds[:k])
    return correct / len(true_labels)

def mean_reciprocal_rank(true_labels, predicted_lists):
    rr_values = []
    for tl, preds in zip(true_labels, predicted_lists):
        try:
            rank = preds.index(tl) + 1
            rr_values.append(1.0 / rank)
        except ValueError:
            rr_values.append(0.0)
    return np.mean(rr_values)
