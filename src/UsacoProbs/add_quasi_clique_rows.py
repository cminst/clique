#!/usr/bin/env python3
import csv
import glob
import os

def add_quasi_clique_rows():
    # Reference CSV that has QuasiClique rows
    reference_csv = "Table1-nTotal=2500,eps=1.0e+06.csv"
    
    # Get all Table1 CSV files except the reference
    csv_files = glob.glob("Table1*.csv")
    target_files = [f for f in csv_files if f != reference_csv]
    
    # Extract QuasiClique rows from reference
    quasi_clique_rows = []
    with open(reference_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "QuasiClique":
                quasi_clique_rows.append(row)
    
    print(f"Found {len(quasi_clique_rows)} QuasiClique rows in reference file")
    
    # Process each target CSV
    for csv_file in target_files:
        print(f"Processing {csv_file}...")
        
        # Read all rows from the target file
        all_rows = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_rows.append(row)
        
        # Skip if it already has QuasiClique rows
        has_quasi_clique = any(row[0] == "QuasiClique" for row in all_rows[1:])  # Skip header
        if has_quasi_clique:
            print(f"  Skipping {csv_file} - already has QuasiClique rows")
            continue
        
        # Find insertion points for each QuasiClique row
        header = all_rows[0]
        data_rows = all_rows[1:]
        
        # Create new rows with QuasiClique data inserted in correct positions
        new_rows = [header]
        
        for quasi_row in quasi_clique_rows:
            # Find the position to insert (after the Densest row with same parameters)
            cluster_size = quasi_row[1]
            internal_density = quasi_row[2]
            external_density = quasi_row[3]
            
            # Find the Densest row with matching parameters
            densest_position = None
            for i, row in enumerate(data_rows):
                if (row[0] == "Densest" and 
                    row[1] == cluster_size and 
                    row[2] == internal_density and 
                    row[3] == external_density):
                    densest_position = i
                    break
            
            if densest_position is not None:
                # Insert QuasiClique row after Densest row
                data_rows.insert(densest_position + 1, quasi_row)
        
        new_rows.extend(data_rows)
        
        # Write back to file
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
        
        print(f"  Added {len(quasi_clique_rows)} QuasiClique rows to {csv_file}")

if __name__ == "__main__":
    add_quasi_clique_rows()