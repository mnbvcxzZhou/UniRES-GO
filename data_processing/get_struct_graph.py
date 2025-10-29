from Bio.PDB import PDBParser, is_aa
import numpy as np
import torch
from torch_geometric.data import Data
import os
import json
from Bio import PDB
from transformers import AutoTokenizer, AutoModel
# Set PDB file directory
pdb_directory = "../data/filtered_pdb"  # Modify according to your file location

# Get all PDB files in the directory
pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith('.pdb')]
print(f"Found {len(pdb_files)} PDB files")

# Count processed files
save_path = '../data/struct_data_v2'
os.makedirs(save_path, exist_ok=True)
processed_files = [f.replace('.pt', '') for f in os.listdir(save_path) if f.endswith('.pt')]
print(f"Processed {len(processed_files)} files")

# ===== Configuration Section =====
distance_threshold = 10.0
# ===== Amino Acid Type One-hot Encoding Mapping =====
aa_list = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS',
          'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN',
          'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
aa_to_idx = {aa: idx for idx, aa in enumerate(aa_list)}
# -----------------------
# Initialize ESM2 Model
# -----------------------
esmmodel_checkpoint = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(esmmodel_checkpoint)
esmmodel = AutoModel.from_pretrained(esmmodel_checkpoint)
esmmodel.eval().cuda()  

feature_dim = esmmodel.config.hidden_size  # ESM2 output dimension: 1280
print(f"ESM2 embedding dim: {feature_dim}")

def aa_one_hot(resname):
    vec = [0] * 20 
    if resname in aa_to_idx:
        vec[aa_to_idx[resname]] = 1
  
    return vec

def extract_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file and generate ESM embedding"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Three-letter to single-letter mapping
    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                if resname in aa_3to1:
                    sequence += aa_3to1[resname]

    print(f"{pdb_path}:{sequence}")
    tokenized = tokenizer(sequence, return_tensors="pt")
    tokenized = {k: v.cuda() for k, v in tokenized.items()}

    with torch.no_grad():
        output = esmmodel(**tokenized)

    last_hidden_state = output.last_hidden_state.squeeze(0)  # [seq_len, 1280]
    return last_hidden_state,sequence

# Iterate through all PDB files
processed_count = 0
skipped_count = 0

for i, pdb_file in enumerate(pdb_files):
    # Check if already processed
    protein_id = pdb_file.replace("AF-", "").split('-')[0]
    save_path = '../data/struct_data_v2'
    os.makedirs(save_path, exist_ok=True)
    PyG_filename = os.path.join(save_path, f"{protein_id}.pt")
    
    if os.path.exists(PyG_filename):
        print(f"[{i+1}/{len(pdb_files)}] Skipping already processed protein: {protein_id}")
        skipped_count += 1
        continue
    
    processed_count += 1
    print(f"[{i+1}/{len(pdb_files)}] Processing protein: {protein_id} (new file #{processed_count})")
    
    # ===== Step 1: Extract residues and their Cα coordinates, sequence from PDB =====
    pdb_path = os.path.join(pdb_directory, pdb_file)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    sequence_feature,sequence = extract_sequence_from_pdb(pdb_path)
    
   

    residue_coords = []
    residue_features = []
    

    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue) and 'CA' in residue:
                    coord = residue['CA'].get_coord()
                    resname = residue.get_resname()
                    feature = aa_one_hot(resname)

                    residue_coords.append(coord)
                    residue_features.append(feature)
                

    residue_coords = np.array(residue_coords)
    residue_features = np.array(residue_features)
    
    print(f"Extracted {len(residue_features)} residues, sequence length: {len(sequence)}")
   
    # ===== Step 2: Concatenate feature matrices =====
    # Ensure ESM embedding length matches residue count
    if sequence_feature.shape[0] != len(residue_coords):
        min_length = min(sequence_feature.shape[0], len(residue_coords))
        sequence_feature = sequence_feature[:min_length]
        residue_coords = residue_coords[:min_length]
        residue_features = residue_features[:min_length]
        print(f"Adjusted to {min_length} residues")
    
    # Convert to tensor and concatenate features
    one_hot_tensor = torch.tensor(residue_features, dtype=torch.float).cuda()  # (num_residues, 20)
    esm_tensor = sequence_feature  # (num_residues, 1280)
    print(f"One-hot features shape: {one_hot_tensor.shape}")
    print(f"ESM features shape: {esm_tensor.shape}")
    
    # Concatenate feature matrices
    combined_features = torch.cat([one_hot_tensor, esm_tensor], dim=1)  # (num_residues, 20+1280)
    

    print(f"Combined features shape: {combined_features.shape}")

    # ===== Step 3: Build edges (edge_index) =====
    edge_index = []

    for i in range(len(residue_coords)):
        for j in range(i + 1, len(residue_coords)):
            dist = np.linalg.norm(residue_coords[i] - residue_coords[j])
            if dist < distance_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph bidirectional edges

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"Created {len(edge_index[0])} edges")

    # ===== Step 4: Build PyG Data object =====
    # Ensure data is saved on CPU
    combined_features_cpu = combined_features.cpu()
    edge_index_cpu = edge_index.cpu()
    
    data = Data(x=combined_features_cpu, edge_index=edge_index_cpu)
    
    # Add additional information
    data.residue_coords = torch.tensor(residue_coords, dtype=torch.float)
    data.sequence = sequence  # Save original sequence string
    
    # ===== Step 5: Save =====
    PyG_filename = os.path.join(save_path, f"{protein_id}.pt")

    torch.save(data, PyG_filename)

    print(f"Saved struct data for {protein_id} as {PyG_filename}")

# Display processing summary
print(f"\n=== Processing Complete ===")
print(f"Total files: {len(pdb_files)}")
print(f"Skipped files: {skipped_count}")
print(f"Newly processed files: {processed_count}")
print(f"Current total processed: {len(processed_files) + processed_count}")
