from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_metrics(y_true, y_pred):
    """
    Calculate multiple evaluation metrics for classification.
    Returns a dictionary with accuracy, precision, recall, F1-score, and confusion matrix.
    """
    metrics_dict = {}

    metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metrics_dict['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics_dict['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics_dict['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    metrics_dict['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics_dict