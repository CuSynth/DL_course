def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(prediction)):
        #         positive                       true
        tp += (1 if prediction[i] and prediction[i] == ground_truth[i] else 0)
        #          positive                       false
        fp += (1 if prediction[i] and prediction[i] != ground_truth[i] else 0)
        #   ....
        tn += (1 if not prediction[i] and prediction[i] == ground_truth[i] else 0)
        #   ....
        fn += (1 if not prediction[i] and prediction[i] != ground_truth[i] else 0)

    tp = tp / len(prediction)
    fp = fp / len(prediction)
    tn = tn / len(prediction)
    fn = fn / len(prediction)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    acc = sum(prediction == ground_truth)
    return acc/len(prediction)
