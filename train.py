import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATv2Conv
import numpy as np
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from model.evaluation import cacul_aupr, calculate_performance
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import glob
import itertools
import json
from datetime import datetime
class ProteinGraphDataset(Dataset):
    """
    A custom Dataset class for the training script.
    It dynamically fixes incorrect file paths stored in the pickled dataset objects.
    """
    def __init__(self, data_dir):
        # This init is just a placeholder; the actual data comes from the pickled Subset object.
        self.file_paths = []

    def __getitem__(self, index):
        """
        This method will be called by the Subset object, which passes the actual data.
        We intercept the file_path and fix it before loading.
        """
        # This is a bit of a workaround. The Subset object will actually hold the
        # instance of the original ProteinGraphDataset from divide_data.py.
        # When we are here, `self` is that original instance.
        
        # Get the original, potentially incorrect, file path
        file_path = self.file_paths[index]
        
        # Dynamically fix the path if it contains '../'
        if file_path.startswith('../'):
            file_path = file_path[3:] # Remove the first 3 characters ('../')

        # Now, load the data from the corrected path
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        graph = data['graph']
        label = data['label']
        protein_id = os.path.basename(file_path).split('.')[0]

        return protein_id, graph, label

    def __len__(self):
        return len(self.file_paths)


class AttentionSumPooling(nn.Module):
    """Only attention layers without GNN with hyperparameter support"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, 
                 dropout_rate=0.5, learning_rate=0.001, 
                 conv_layers=3, num_attention_heads=8):
        super(AttentionSumPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.conv_layers = conv_layers
        self.num_attention_heads = num_attention_heads
        
        # Direct projection to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Dynamic number of GAT layers based on conv_layers parameter
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_feats=hidden_dim, 
                out_feats=hidden_dim // num_attention_heads, 
                num_heads=num_attention_heads,
                residual=True,
                activation=F.relu,
                allow_zero_in_degree=True
            ) for _ in range(conv_layers)
        ])
        
        # Classifier with configurable dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, graph):
        h = self.input_proj(graph.ndata['feature'])
        
        for gat_layer in self.gat_layers:
            h = gat_layer(graph, h)
            h = h.view(-1, self.hidden_dim)
        
        graph.ndata['feature'] = h
        h_graph = dgl.sum_nodes(graph, 'feature')
        logits = self.classifier(h_graph)
        
        return logits


def train_and_evaluate(model, train_dataloader, valid_dataloader, device, 
                      label_network, branch, model_name, epochs=40):
    """Train and evaluate a model variant"""
    print(f"\n=== Training {model_name} ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_fscore = 0
    best_aupr = 0
    best_auc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for pids, graphs, labels in tqdm(train_dataloader, desc=f"Training {model_name}"):
            graphs = graphs.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels, dim=1)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(0)

            optimizer.zero_grad()
            logits = model(graphs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation every 20 epochs
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            model.eval()
            all_preds = []
            all_actuals = []
            
            with torch.no_grad():
                for pids, graphs, labels in valid_dataloader:
                    graphs = graphs.to(device)
                    labels = labels.to(device)
                    labels = torch.squeeze(labels, dim=1)
                    if len(labels.shape) == 1:
                        labels = labels.unsqueeze(0)
                    
                    logits = model(graphs)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    all_actuals.extend(labels.cpu().numpy())
            
            all_preds = np.array(all_preds)
            all_actuals = np.array(all_actuals)
            
            # Calculate metrics
            aupr = cacul_aupr(all_actuals.flatten(), all_preds.flatten())
            auc_score = roc_auc_score(all_actuals.flatten(), all_preds.flatten())
            
            # Find best F1 score and corresponding precision, recall
            best_f1 = 0
            best_precision = 0
            best_recall = 0
            for thresh in [x/100 for x in range(1,100)]:
                f_score, precision, recall = calculate_performance(
                    all_actuals, all_preds, label_network, threshold=thresh)
                if f_score > best_f1:
                    best_f1 = f_score
                    best_precision = precision
                    best_recall = recall
            
            if best_f1 > best_fscore:
                best_fscore = best_f1
                best_aupr = aupr
                best_auc = auc_score
            
            print(f"Epoch {epoch+1}: F1={best_f1:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}, AUC={auc_score:.4f}, AUPR={aupr:.4f}")
    
    return {
        'model_name': model_name,
        'best_fscore': best_fscore,
        'best_aupr': best_aupr,
        'best_auc': best_auc
    }

def run_ablation_study(args):
    """Run comprehensive ablation study"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    train_data_path = f'divided_data/{args.branch}_train_dataset'
    valid_data_path = f'divided_data/{args.branch}_valid_dataset'
    label_network_path = f'processed_data/label_{args.branch}_network'
    
    with open(train_data_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_dataset = pickle.load(f)
    with open(label_network_path, 'rb') as f:
        label_network = pickle.load(f)
    
    label_network = label_network.to(device)
    
    # Fix paths
    for dataset in [train_dataset, valid_dataset]:
        full_dataset = dataset.dataset
        full_dataset.file_paths = [
            p[3:] if p.startswith('../') else p for p in full_dataset.file_paths
        ]
    
    # Create dataloaders
    train_dataloader = GraphDataLoader(
        dataset=train_dataset, 
        batch_size=32, 
        drop_last=False, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_dataloader = GraphDataLoader(
        dataset=valid_dataset, 
        batch_size=32, 
        drop_last=False, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Define model variants
    model_variants = [
        ('AttentionSumPooling', AttentionSumPooling(1300, 512, args.labels_num))
    ]
    
    # Run experiments
    results = []
    for model_name, model in model_variants:
        model = model.to(device)
        result = train_and_evaluate(
            model, train_dataloader, valid_dataloader, device, 
            label_network, args.branch, model_name, epochs=40
        )
        results.append(result)
        
        # Save model
        torch.save(model.state_dict(), f'ablation_models/{model_name}_{args.branch}.pth')
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Model':<20} {'F1-Score':<10} {'AUC':<10} {'AUPR':<10}")
    print("-"*60)
    
    for result in results:
        print(f"{result['model_name']:<20} {result['best_fscore']:<10.4f} "
              f"{result['best_auc']:<10.4f} {result['best_aupr']:<10.4f}")
    
    # Save results
    import json
    with open(f'ablation_results_{args.branch}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ablation_results_{args.branch}.json")

def train_with_hyperparams(model, train_dataloader, valid_dataloader, device, 
                          label_network, hyperparams, epochs=40):
    """Train model with specific hyperparameters and return detailed metrics"""
    print(f"\n=== Training with hyperparams: {hyperparams} ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], 
                                weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_metrics = {'f1': 0, 'auc': 0, 'aupr': 0, 'precision_recall_curve': None}
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for pids, graphs, labels in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}"):
            graphs = graphs.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels, dim=1)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(0)

            optimizer.zero_grad()
            logits = model(graphs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            model.eval()
            all_preds = []
            all_actuals = []
            
            with torch.no_grad():
                for pids, graphs, labels in valid_dataloader:
                    graphs = graphs.to(device)
                    labels = labels.to(device)
                    labels = torch.squeeze(labels, dim=1)
                    if len(labels.shape) == 1:
                        labels = labels.unsqueeze(0)
                    
                    logits = model(graphs)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    all_actuals.extend(labels.cpu().numpy())
            
            all_preds = np.array(all_preds)
            all_actuals = np.array(all_actuals)
            
            # Calculate metrics
            aupr = cacul_aupr(all_actuals.flatten(), all_preds.flatten())
            auc_score = roc_auc_score(all_actuals.flatten(), all_preds.flatten())
            
            # Find best F1 score
            best_f1 = 0
            best_thresh = 0.5
            for thresh in [x/100 for x in range(1,100)]:
                f_score, precision, recall = calculate_performance(
                    all_actuals, all_preds, label_network, threshold=thresh)
                if f_score > best_f1:
                    best_f1 = f_score
                    best_thresh = thresh
            
            # Calculate precision-recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(
                all_actuals.flatten(), all_preds.flatten())
            
            if best_f1 > best_metrics['f1']:
                best_metrics.update({
                    'f1': best_f1,
                    'auc': auc_score,
                    'aupr': aupr,
                    'precision_recall_curve': (precision_curve, recall_curve),
                    'best_threshold': best_thresh
                })
            
            print(f"Epoch {epoch+1}: F1={best_f1:.4f}, AUC={auc_score:.4f}, AUPR={aupr:.4f}")
    
    return best_metrics

def hyperparameter_tuning(args):
    """Run hyperparameter tuning with grid search"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    train_data_path = f'divided_data/{args.branch}_train_dataset'
    valid_data_path = f'divided_data/{args.branch}_valid_dataset'
    label_network_path = f'processed_data/label_{args.branch}_network'
    
    with open(train_data_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_dataset = pickle.load(f)
    with open(label_network_path, 'rb') as f:
        label_network = pickle.load(f)
    
    label_network = label_network.to(device)
    
    # Fix paths
    for dataset in [train_dataset, valid_dataset]:
        full_dataset = dataset.dataset
        full_dataset.file_paths = [
            p[3:] if p.startswith('../') else p for p in full_dataset.file_paths
        ]
    
    # Create dataloaders
    train_dataloader = GraphDataLoader(
        dataset=train_dataset, 
        batch_size=32, 
        drop_last=False, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_dataloader = GraphDataLoader(
        dataset=valid_dataset, 
        batch_size=32, 
        drop_last=False, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Define hyperparameter search space
    hyperparams_space = {
        'dropout_rate': [0.3],
        'learning_rate': [ 0.0001],
        'conv_layers': [2],
        'num_attention_heads': [4]  # Additional hyperparameter
    }
    
    # Generate all combinations
    param_names = list(hyperparams_space.keys())
    param_values = list(hyperparams_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Total combinations to test: {len(all_combinations)}")
    
    results = []
    
    for i, combination in enumerate(all_combinations):
        hyperparams = dict(zip(param_names, combination))
        
        # Create model with current hyperparameters
        model = AttentionSumPooling(
            in_dim=1300, 
            hidden_dim=512, 
            out_dim=args.labels_num,
            **hyperparams
        ).to(device)
        
        # Train and evaluate
        metrics = train_with_hyperparams(
            model, train_dataloader, valid_dataloader, device, 
            label_network, hyperparams, epochs=40
        )
        
        # Store results
        result = {
            'hyperparams': hyperparams,
            'metrics': metrics,
            'combination_id': i
        }
        results.append(result)
        
        # Save model for this hyperparameter combination
        model_save_path = f'hyperparameter_models/model_{args.branch}_combo_{i:03d}.pth'
        os.makedirs('hyperparameter_models', exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
        print(f"Combination {i+1}/{len(all_combinations)} completed")
        print(f"Best F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}, AUPR: {metrics['aupr']:.4f}")
        print("-" * 50)
    
    # Find best hyperparameters
    best_result = max(results, key=lambda x: x['metrics']['f1'])
    print(f"\nBest hyperparameters: {best_result['hyperparams']}")
    print(f"Best F1: {best_result['metrics']['f1']:.4f}")
    print(f"Best AUC: {best_result['metrics']['auc']:.4f}")
    print(f"Best AUPR: {best_result['metrics']['aupr']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'hyperparameter_results_{args.branch}_{timestamp}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        if serializable_result['metrics']['precision_recall_curve'] is not None:
            precision, recall = serializable_result['metrics']['precision_recall_curve']
            serializable_result['metrics']['precision_recall_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        serializable_results.append(serializable_result)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-branch', '--branch', type=str, default='mf', 
                       help="ontology namespace: mf, bp, or cc")
    parser.add_argument('-labels_num', '--labels_num', type=int, default=324, 
                       help="number of labels for the branch")
    parser.add_argument('-mode', '--mode', type=str, default='hyperparameter', 
                       choices=['ablation', 'hyperparameter'],
                       help="mode: ablation study or hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Create directory for models and results
    os.makedirs('ablation_models', exist_ok=True)
    os.makedirs('hyperparameter_models', exist_ok=True)
    os.makedirs('hyperparameter_analysis', exist_ok=True)
    
    if args.mode == 'ablation':
        run_ablation_study(args)
    elif args.mode == 'hyperparameter':
        results = hyperparameter_tuning(args)