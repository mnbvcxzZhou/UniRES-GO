import os
import glob
import pickle
import torch
from torch.utils.data import Dataset

class ProteinGraphDataset(Dataset):
    """
    A PyTorch Dataset that loads protein graph data from individual .pkl files.
    This approach is memory-efficient as it only loads one file at a time.
    """
    def __init__(self, data_dir):
        super().__init__()
        # Get a list of all .pkl file paths in the specified directory
        self.file_paths = glob.glob(os.path.join(data_dir, '*.pkl'))
        if not self.file_paths:
            raise ValueError(f"No .pkl files were found in directory: {data_dir}")

    def __getitem__(self, index):
        """
        Loads and returns a single data sample.
        """
        # Get the file path for the requested index
        file_path = self.file_paths[index]
        
        # Load the data from the .pkl file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        graph = data['graph']
        label = data['label']
        
        # The protein ID is derived from the filename
        protein_id = os.path.basename(file_path).split('.')[0]

        # Note: Sequence features are now part of the graph's node features (g.ndata['feature'])
        # and are not returned separately.
        return protein_id, graph, label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_paths)

if __name__ == "__main__":
    # Define the Gene Ontology namespaces to process
    ns_type_list = ['bp', 'mf', 'cc']
    
    for ns_type in ns_type_list:
        print(f"--- Dividing {ns_type.upper()} dataset ---")
        
        # This path must match the output directory from 'labels_load.py'
        data_dir = f'../processed_data/{ns_type}_graph_data'
        
        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}. Skipping.")
            continue
            
        try:
            # Create the dataset instance
            dataset = ProteinGraphDataset(data_dir=data_dir)
            print(f"Found {len(dataset)} total samples for {ns_type}.")
        except ValueError as e:
            print(e)
            continue

        # Ensure there's enough data to create non-empty splits
        if len(dataset) < 3:
            print(f"Not enough data to split (found {len(dataset)} samples). Skipping.")
            continue

        # Define split sizes (e.g., 70% train, 15% validation, 15% test)
        train_size = int(len(dataset) * 0.7)
        valid_size = int(len(dataset) * 0.15)
        test_size = len(dataset) - train_size - valid_size
        
        # Perform the random split
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(42)  # for reproducible splits
        )

        # Create the output directory for the split data if it doesn't exist
        output_dir = '../divided_data/'
        os.makedirs(output_dir, exist_ok=True)

        # Save the dataset subsets (these are lightweight 'Subset' objects)
        with open(os.path.join(output_dir, f'{ns_type}_train_dataset'), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(output_dir, f'{ns_type}_valid_dataset'), 'wb') as f:
            pickle.dump(valid_dataset, f)
        with open(os.path.join(output_dir, f'{ns_type}_test_dataset'), 'wb') as f:
            pickle.dump(test_dataset, f)
            
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}\n")

    print("--- All datasets divided successfully! ---")