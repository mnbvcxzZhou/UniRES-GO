import pandas as pd
import pickle 
import collections
import numpy as np
import torch
import dgl
import os
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# 从PDB中读取seq数据（经过onehot处理）和结构数据，json中只用到了GO数据
# 这里的文件并不重要，你只需要导入将数据处理为一个字典，格式为 {"P01283":["GO:0002312","GO:0202312",...], }即可
# # --------------------------
# 加载 `filtered_protein_list.csv`
# --------------------------
df = pd.read_csv('../data/filtered_protein_list.csv', sep=" ", encoding="utf-8-sig", header=None)
protein_list = df[0].tolist()  # 获取蛋白质 ID 列表
print(f"Total proteins in filtered_protein_list: {len(protein_list)}")
with open("../data/uniprotkb_taxonomy_id_9606_AND_reviewed_2025_03_06.json","r") as f:
    human_dict = json.load(f)
labels = {}
for protein_id in human_dict:
    go_tags = human_dict[protein_id]
    if go_tags is not None and go_tags != []:  # 判断 go_tags 是否不为 None 且不为空列表
        labels[protein_id] = go_tags
print(f"Loaded GO labels for {len(labels)} proteins")

gos=[]
namespace=collections.defaultdict(str)
is_a=collections.defaultdict(list)
part=collections.defaultdict(list)
###根据规则来提取go term ，并依据其之间的依赖关系构建图谱
# 只取用is_a和part_of关系
# 边的方向：A is_a B  <=>  A -> B
# 同样，这里的文件也不重要，可以在Gene Ontology上下载
print("--------------1: go term processing")
with open('../data/go-basic.obo','r') as fin:
    for line in fin:
        if '[Typedef]' in line:
            break
        if line[:5]=='id: G':
            line=line.strip().split()
            gos.append(line[1])
        elif line[:4]=='is_a':
            line=line.strip().split()
            is_a[gos[-1]].append(line[1])
        elif line[:4]=='rela' and 'part' in line:
            line=line.strip().split()
            part[gos[-1]].append(line[2])
        elif line[:5]=='names':
            line=line.strip().split()
            namespace[gos[-1]]=line[1]
# 把所有关系并入到is_a中，方便处理
for i in part:
    is_a[i].extend(part[i])
##划分子空间，每个子空间是一个集合
bp,mf,cc=set(),set(),set()
for i in namespace:
    if namespace[i]=='biological_process':
        bp.add(i)
    elif namespace[i]=='molecular_function':
        mf.add(i)
    elif namespace[i]=='cellular_component':
        cc.add(i)


print("--------------2: propagate labels")
# 使用暴力求go_term_set的传递闭包（TODO 可优化）
def propagate(l):
    while True:
        length=len(l)
        temp=[]
        for i in l:
            temp.extend(is_a[i])
        l.update(temp)
        if len(l)==length:
            return l

# 只选取有GO标签的蛋白质进行数据处理
# 对于每个蛋白质，求他们功能标签的传递闭包
pro_with_go={}
for i in tqdm(labels):
    if len(labels[i])>0:
        pro_with_go[i]=propagate(set(labels[i]))
print("protein_num:",len(labels),"->",len(pro_with_go))


print("--------------3: split protein set")
# # TODO: 需要统一文件夹路径
# df=pd.read_csv("../data/filtered_protein_list.csv",sep=" ",encoding="utf-8-sig",header=None)
# tmp_list=df.values.tolist()
# protein_list=[]
# for i in tmp_list:
#     protein_list.append(i[0])
# print(len(protein_list))

label_bp=collections.defaultdict(list)
label_mf=collections.defaultdict(list)
label_cc=collections.defaultdict(list)

# 最终我们选取既有go又有struct_features的蛋白质
for i in pro_with_go:
    if i in protein_list:
        for j in pro_with_go[i]:
            if j in bp:
                label_bp[i].append(j)
            elif j in mf:
                label_mf[i].append(j)
            elif j in cc:
                label_cc[i].append(j)


print("--------------4: read all kinds of features")
# 这里不再预加载所有特征数据，以节省内存

def goterm2idx(term_set):
    term_dict={v:k for k,v in enumerate(term_set)}
    return term_dict

# 求出每个蛋白质筛选后的功能标签的multi-hot向量
def labels2onehot(protein2func_label,index):
    protein2func_onehot={}
    protein2func_label_filtered={}
    l=len(index)
    for i in protein2func_label:
        one_hot = [0]*l
        protein2func_label_filtered[i] = []
        for j in protein2func_label[i]:
            if j in index:
                one_hot[index[j]]=1
                protein2func_label_filtered[i].append(j)
        protein2func_onehot[i]=one_hot
    return protein2func_onehot,protein2func_label_filtered

# 根据bp、cc、mf三种GO标签进行分类
# label 表示需要处理的 {protein_id:[go_list]} 字典
# seq_feature_dic {protein_id:蛋白质序列的特征} 字典
# go_dependency: GO 词条的依赖关系
def label_process(label,ns_type,go_dependency):
    print("now processing: ",ns_type)
    # 第一步: 过滤出go_term中出现次数大于thresh的标签
    print("----step 1----")
    counter=collections.Counter()
    for i in label:
        counter.update(label[i])
    tong=dict(counter)
    final_go=set()
    for i in tong:
        if ns_type=='bp' and tong[i]>=250:
            final_go.add(i)
        if ns_type!='bp' and tong[i]>=100:
            final_go.add(i)
    print("total_process",ns_type,"final_go_term_size",len(final_go))
    
    # 第二步：对筛选出来的go_term进行编号
    print("----step 2----")
    term2idx=goterm2idx(final_go)
    with open('../processed_data/'+ns_type+'_term2idx.json','w') as f:
        json.dump(term2idx,f,indent=4)

    # 第三步：求出每个蛋白质的功能标签序列对应的multihot向量, 并且把label筛选一遍
    # 其实这里用onehot并不严谨，应该称为multi-hot
    print("----step 3----")
    pro2func_multi_hot,pro2func_filtered = labels2onehot(label,term2idx)
    final_protein_list=list(pro2func_filtered.keys())

    # 第四步：作图统计
    # 统计每个list的长度
    print("----step 4----")
    lengths = [len(value) for value in pro2func_filtered.values()]
    plt.gca().set_prop_cycle(None)
    # 绘制直方图
    n, bins, patches = plt.hist(lengths, bins=[0, 100, 200, 300, 400, 500], edgecolor='black', facecolor='blue')  # 这里的bins定义了区间，您可以根据需要调整
    # 在每个柱子上标注数字
    for i in range(len(n)):
        plt.text(bins[i] + 0.5, n[i] + 0.2, str(int(n[i])), ha='center', va='bottom')
    plt.xlabel('Protein Numbers')
    plt.ylabel('GO term Number')
    plt.title(ns_type+'-go')
    plt.legend(loc='upper right')  # 显示图例
    plt.savefig('../processed_data/histogram_'+ns_type+'.png', format='png')
    # plt.show()

    # 将字典转换为一对一的键值对
    pairs = [(key, val) for key, values in pro2func_filtered.items() for val in values]
    # 创建DataFrame
    df = pd.DataFrame(pairs, columns=['Protein', ns_type+'-go'])
    # 保存为CSV文件
    df.to_csv('../processed_data/gos_'+ns_type+'.csv', index=False)

    # 第五步：为每个蛋白质生成并保存独立的DGL图和标签
    print("----step 5----")
    
    # 为当前GO类别创建输出目录
    output_dir = f'../processed_data/{ns_type}_graph_data'
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    pt_folder = '../data/struct_data_v2/'
    
    for i in tqdm(final_protein_list, desc=f"Processing {ns_type}"):
        # 构建.pt文件路径
        pt_path = os.path.join(pt_folder, f"{i}.pt")
        
        # 检查数据文件是否存在
        if not os.path.exists(pt_path):
            print(f"File not found: {pt_path}")
            continue
            
        # 从 .pt 数据中即时读取图结构和节点特征
        try:
            data = torch.load(pt_path)
            if not hasattr(data, 'edge_index') or not hasattr(data, 'x'):
                continue
            edge_index = data.edge_index
            node_features = data.x
        except Exception:
            continue

        # 构建 DGL 图
        try:
            g = dgl.graph((edge_index[0], edge_index[1]))
            g = dgl.add_self_loop(g)
            if node_features.shape[0] != g.num_nodes():
                continue
            g.ndata['feature'] = node_features.float()
        except Exception:
            continue

        # 获取功能标签
        multihot_go = torch.tensor(pro2func_multi_hot[i], dtype=torch.float32)
        multihot_go = torch.unsqueeze(multihot_go, 0)

        # 将图和标签保存在一个字典中
        processed_data = {'graph': g, 'label': multihot_go}
        
        # 保存到独立的文件
        output_pkl_path = os.path.join(output_dir, f"{i}.pkl")
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(processed_data, f)
            
        processed_count += 1
        
    print(f"{ns_type} dataset size: {processed_count}")

    # 第六步：输出GO graph
    print("----step 6----")
    go_graph = dgl.DGLGraph()
    go_graph = dgl.add_self_loop(go_graph)
    go_graph.add_nodes(len(final_go))

    term_to_idx = {term: idx for idx, term in enumerate(final_go)}
    for child, parents in go_dependency.items():
        if child in term_to_idx:
            child_idx = term_to_idx[child]
            for parent in parents:
                if parent in term_to_idx:
                    parent_idx = term_to_idx[parent]
                    go_graph.add_edges(torch.tensor([child_idx]), torch.tensor([parent_idx]))

    with open('../processed_data/label_'+ns_type+'_network','wb')as f:
        pickle.dump(go_graph,f)  

print("--------------5: process all kinds of files")
label_process(label_bp,"bp",is_a)
label_process(label_cc,"cc",is_a)
label_process(label_mf,"mf",is_a)