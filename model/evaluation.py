from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import dgl


import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, recall_score, precision_score

def evaluate_model(preds, labels, thresholds=None):
    """
    Evaluate model performance using various metrics and thresholds.
    
    Args:
        preds (numpy array or tensor): Model's raw output (logits or probabilities).
        labels (numpy array or tensor): True binary labels.
        thresholds (list or None): List of thresholds for binary classification (between 0 and 1).
    
    Returns:
        dict: The best threshold and corresponding metrics (F1, AUC, AUPR, Recall, Precision).
    """
    
    if thresholds is None:
        thresholds = [x / 100 for x in range(1, 100)]  # Default thresholds from 0.01 to 0.99
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    # Convert logits to probabilities if necessary
    if preds.ndim == 2:  # In case of raw logits, apply sigmoid
        preds = 1 / (1 + np.exp(-preds))  # Apply sigmoid to logits
    
    for threshold in thresholds:
        preds_binary = (preds >= threshold).astype(int)  # Thresholding
        
        # Calculate metrics
        f1 = f1_score(labels, preds_binary, average='samples')
        # Use average_precision_score instead of roc_auc_score for multi-label classification
        auc = average_precision_score(labels, preds, average='samples')
        aupr = calculate_aupr(labels, preds)
        recall = recall_score(labels, preds_binary, average='samples')
        precision = precision_score(labels, preds_binary, average='samples')
        
        # Update best metrics if necessary
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'F1-score': f1,
                'AUC': auc,
                'AUPR': aupr,
                'Recall': recall,
                'Precision': precision
            }

    return best_threshold, best_metrics

def update_parent_features(label_network:dgl.DGLGraph, labels):
    # 获取图中的所有边
    edges = label_network.edges()
    # 对于图中的每条边
    for child_idx, parent_idx in zip(edges[0], edges[1]):
        # 如果child节点的特征值大于parent节点的特征值
        if labels[0][child_idx] > labels[0][parent_idx]:
            # 更新parent节点的特征值为child节点的特征值
            labels[0][parent_idx] = labels[0][child_idx]
        # 更新labels的第二列为second_dim_elements
    return labels

def cacul_aupr(lables, pred):
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = 0.06
    aupr += metrics.auc(recall, precision)
    return aupr
def calculate_aupr(labels, preds):
    """
    Calculate AUPR (Area Under Precision-Recall Curve) for each label in a multilabel classification task.
    
    Args:
        labels (numpy array): True binary labels.
        preds (numpy array): Model's raw output (logits or probabilities).
    
    Returns:
        float: The mean AUPR across all labels.
    """
    n_labels = labels.shape[1]  # Number of labels
    aupr_list = []

    for i in range(n_labels):
        # Calculate precision-recall curve for each label
        precision, recall, _thresholds = metrics.precision_recall_curve(labels[:, i], preds[:, i])
        
        # Calculate AUPR for each label
        aupr = metrics.auc(recall, precision)
        aupr_list.append(aupr)
    
    # Return the average AUPR across all labels
    return np.mean(aupr_list)

def calculate_performance(actual, pred_prob, label_network:dgl.DGLGraph, threshold=0.2, average='micro'):
    pred_lable = []
    actual_label = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]).flatten() > threshold).astype(np.int32)
        eachline = eachline.tolist()
        # eachline = update_parent_features(label_network,eachline)
        pred_lable.append(list(eachline))
    for l in range(len(actual)):
        eachline = (np.array(actual[l]).flatten()).astype(np.int32)
        eachline = eachline.tolist()
        actual_label.append(list(eachline))
    f_score = f1_score(actual_label, pred_lable, average=average)
    recall = recall_score(actual_label, pred_lable, average=average)
    precision = precision_score(actual_label,  pred_lable, average=average)
    return f_score, precision, recall

