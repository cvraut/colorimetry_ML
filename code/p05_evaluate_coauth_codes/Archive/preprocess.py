import os
import pandas as pd
import numpy as np
import re

class Preprocessor(object):
    def __init__(self):
        pass

    def mine_components(self, output_file):
        """ones begining with f are related to the foler called full results ones beginning with t to TI results"""
        self.FTERAW = '../data/raw spectrum data/FULL/TE/TXT'
        self.FTMRAW = '../data/raw spectrum data/FULL/TM/TXT'
        self.TTERAW = '../data/raw spectrum data/TI/TE/TXT'
        self.TTMRAW = '../data/raw spectrum data/TI/TM/TXT'
        # Define directory paths and their specific table names
        dir_configs = [
            (self.FTERAW, 'FTER'),
            (self.FTMRAW, 'FTMR'),
            (self.TTERAW, 'TTER'),
            (self.TTMRAW, 'TTMR')
        ]

        # Dictionary to hold arrays with specific table names
        data_tables = {}

        # Process each directory
        for dir_path, table_name in dir_configs:
            indexsweep = []
            for fpath in os.listdir(dir_path):
                instance = []
                # Extract number from filename (e.g., 1300 from "index=1300.txt")
                number = fpath[-8:-4]  # Assumes format like before
                instance.append(float(number) / 1000)  # Convert to float and divide
                with open(os.path.join(dir_path, fpath), 'r') as f:
                    for line in f:
                        instance.append(float(line.strip()))  # Convert lines to float
                indexsweep.append(instance)
            # Store with specific table name
            array = np.array(indexsweep)
            data_tables[table_name] = array[array[:, 0].argsort()]
        # Save all tables to one .npz file
        np.savez_compressed(output_file, **data_tables)

    def mine_denoised(self, output_file):
        """ones begining with f are related to the foler called full results ones beginning with t to TI results"""
        self.DTERAW = '../data/raw spectrum data/D/TE'
        self.DTMRAW = '../data/raw spectrum data/D/TM'

        # Define directory paths and their specific table names
        dir_configs = [
            (self.DTERAW, 'DTER'),
            (self.DTMRAW, 'DTMR')
        ]
        # Dictionary to hold arrays with specific table names
        data_tables = {}

        # Process each directory
        for dir_path, table_name in dir_configs:
            indexsweep = []
            for fpath in os.listdir(dir_path):
                instance = []
                # Extract number from filename (e.g., 1300 from "index=1300.txt")
                number = fpath[-8:-4]  # Assumes format like before
                instance.append(float(number) / 1000)  # Convert to float and divide
                with open(os.path.join(dir_path, fpath), 'r') as f:
                    next(f)
                    for line in f:
                        columns = line.strip().split('\t')
                        if len(columns) >= 3:
                            third_column = float(columns[2])
                            instance.append(third_column)
                        else:
                            print(f"Warning: Line in {fpath} does not have enough columns: {line.strip()}")
                    indexsweep.append(instance)
            # Store with specific table name
            array = np.array(indexsweep)
            data_tables[table_name] = array[array[:, 0].argsort()]
        # Save all tables to one .npz file
        np.savez_compressed(output_file, **data_tables)


    def compare_tables(self,a,b):
        if a.shape != b.shape:
            print("The tables have different shapes and cannot be compared directly.")
            return

            # Step 2: Compute the difference array
        diff = a != b

        # Step 3: Check if there are any differences
        if not np.any(diff):
            print("The tables are identical.")
        else:
            print("The tables are not identical.")
            # Step 4: Find indices where they differ
            diff_indices = np.where(diff)
            # Step 5: Print each differing index and the corresponding values
            for idx in zip(*diff_indices):
                print(f"Element at {idx}: a{idx} = {a[idx]}, b{idx} = {b[idx]}")


pp = Preprocessor()
output_file = '../data/processed_data/noisy.npz'
pp.mine_components(output_file)

# Verify: Check first 5 rows of each table
loaded = np.load(output_file)
table_names = ['FTER','FTMR','TTER','TTMR']
for table_name in table_names:
    print(f"{table_name} first 5 rows:\n", loaded[table_name][:5])
    print(f"{table_name} last 5 rows:\n", loaded[table_name][-5:])

denoised_output = '../data/processed_data/denoised.npz'
pp.mine_denoised(denoised_output)

loaded_denoised = np.load(denoised_output)
table_names = ['DTER','DTMR']
for table_name in table_names:
    print(f"{table_name} first 5 rows:\n", loaded_denoised[table_name][:5])
    print(f"{table_name} last 5 rows:\n", loaded_denoised[table_name][-5:])

#merging noisy and denoised data into one single dataset
combined_data = {**loaded_denoised, **loaded}
np.savez_compressed('../data/processed_data/all.npz', **combined_data)

