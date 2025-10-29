import pandas as pd
import os
import shutil

# Read filtered_protein_list.csv and get the list of protein IDs
filtered_df = pd.read_csv('../data/filtered_protein_list.csv', encoding="utf-8-sig", header=None)
protein_ids = filtered_df.iloc[:, 0].tolist()

# Set folder paths
input_folder = '../data/pdb'  # Path to the source folder
output_folder = '../data/filtered_pdb'  # Path to save the filtered files

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get all files in the folder
all_files = os.listdir(input_folder)

# Track which protein IDs are found
found_protein_ids = set()

# Iterate over files and filter those that meet the criteria
for file in all_files:
    if file.endswith('.pdb'):
        # Extract protein ID and F value
        parts = file.split('-')
        protein_id = parts[1]  # Keep the second part only, e.g., "Q7Z407"
        f_value = parts[2]  # e.g., "F1"

        # If the protein ID is in the CSV and the F value is F1
        if protein_id in protein_ids and f_value == 'F1':
            # Copy the matching file to the output folder
            shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, file))
            found_protein_ids.add(protein_id)
# Output protein IDs that were not found
missing_proteins = set(protein_ids) - found_protein_ids
if missing_proteins:
    print("The following protein IDs did not have corresponding files:")
    for protein in missing_proteins:
        print(protein)
else:
    print("All protein files have been found and processed.")

print("Filtering completed!")
