from torch import nn
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv,GATConv, AvgPooling, MaxPooling
from dgl.nn.pytorch import GraphConv, GATv2Conv

import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
###########################This model is not the best one at present.################################
class GNNTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8):
        super(GNNTransformer, self).__init__()
        self.hidden_dim = hidden_dim

        # 图卷积层，用于聚合局部邻域信息
        self.gnn = nn.ModuleList([
            GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True),
            GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True),
            GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        ])
        
        # GATv2层，用于在每个图中计算全局注意力
        # 它能正确处理DGL的批处理图，是TransformerEncoderLayer的理想替代品
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_feats=hidden_dim, 
                out_feats=hidden_dim // num_heads, 
                num_heads=num_heads,
                residual=True,      # 使用内置的残差连接
                activation=F.relu,  # 使用内置的激活函数
                allow_zero_in_degree=True
            ) for _ in range(3)  # 堆叠3层GAT
        ])
        
        # 将池化层和分类器合并为一个模块
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, graph):
        # 从图中获取初始节点特征
        h = graph.ndata['feature']
        
        # 通过GNN层
        for gnn_layer in self.gnn:
            h = gnn_layer(graph, h)
            h = F.relu(h)
        
        # 通过GATv2注意力层
        for gat_layer in self.gat_layers:
            # GATv2Conv的输出形状是 (节点数, 头数, 每头的特征维度)
            h = gat_layer(graph, h)
            # 将多头的输出合并为一个张量
            # 形状变为 (节点数, 头数 * 每头的特征维度) = (节点数, hidden_dim)
            h = h.view(-1, self.hidden_dim)
        
        # 更新图中的节点特征，用于后续的池化
        graph.ndata['feature'] = h
        
        # 对每个图的节点特征进行最大池化，得到图级别的表示
        h_graph = dgl.sum_nodes(graph, 'feature')
        
        # 通过分类器得到最终的logits
        logits = self.classifier(h_graph)
        
        return logits

