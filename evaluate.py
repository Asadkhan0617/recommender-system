def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    if k == 0:
        return 0
    return len(set(recommended_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    if len(relevant) == 0:
        return 0
    return len(set(recommended_k) & set(relevant)) / len(relevant)


def f1_score_at_k(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)