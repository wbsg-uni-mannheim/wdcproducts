import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pdb import set_trace
    
def compute_metrics_bce(eval_pred):
    logits, labels = eval_pred

    logits = np.copy(logits)
    labels = np.copy(labels)

    logits[logits>=0.5] = 1
    logits[logits<0.5] = 0
    predictions = logits.reshape(-1)
    labels = labels.reshape(-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, pos_label=1, average='binary')
    precision = precision_score(labels, predictions, pos_label=1, average='binary')
    recall = recall_score(labels, predictions, pos_label=1, average='binary')

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics_cosine(eval_pred):
    threshold = 0.75
    logits, labels = eval_pred

    logits = np.copy(logits)
    labels = np.copy(labels)

    predictions = logits
    predictions[predictions>=threshold] = 1
    predictions[predictions<threshold] = 0
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, pos_label=1, average='binary')
    precision = precision_score(labels, predictions, pos_label=1, average='binary')
    recall = recall_score(labels, predictions, pos_label=1, average='binary')

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics_baseline(eval_pred):
    logits, labels = eval_pred

    logits = np.copy(logits)
    labels = np.copy(labels)

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, pos_label=1, average='binary')
    precision = precision_score(labels, predictions, pos_label=1, average='binary')
    recall = recall_score(labels, predictions, pos_label=1, average='binary')

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics_baseline_multiclass(eval_pred):
    logits, labels = eval_pred

    logits = np.copy(logits)
    labels = np.copy(labels)

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='micro')
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')

    return {"accuracy": accuracy, "f1_micro": f1, "precision_micro": precision, "recall_micro": recall, "f1_macro": f1_macro, "precision_macro": precision_macro, "recall_macro": recall_macro}

def compute_metrics_matrix(eval_pred):
    logits, labels = eval_pred

    logits = np.copy(logits)
    labels = np.copy(labels)

    logits = logits[logits!=-100]
    logits[logits>=0.5] = 1
    logits[logits<0.5] = 0

    labels = labels[labels!=-100]
    
    predictions = logits.reshape(-1)
    labels = labels.reshape(-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, pos_label=1, average='binary')
    precision = precision_score(labels, predictions, pos_label=1, average='binary')
    recall = recall_score(labels, predictions, pos_label=1, average='binary')

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}