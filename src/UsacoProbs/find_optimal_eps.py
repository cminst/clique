import csv
import os
import re
from pathlib import Path

def extract_eps_from_filename(filename):
    """Extract eps value from filename using regex."""
    match = re.search(r'eps=([0-9.]+(?:e[+-]?\d+)?)', filename)
    if match:
        return float(match.group(1))
    return None

def calculate_average_f1(csv_file):
    """Calculate average F1 score for L-RMC method in a CSV file."""
    f1_scores = []
    
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Method'] == 'L-RMC':
                    f1_scores.append(float(row['F1']))
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None
    
    if not f1_scores:
        return None
    
    return sum(f1_scores) / len(f1_scores)

def analyze_csv_files():
    """Analyze all CSV files and find optimal eps value."""
    csv_files = []
    
    # Find all CSV files matching the pattern
    for file in Path('.').glob('L-RMC-Versus-Other-Graph_Algs-nTotal=2000,eps=*.csv'):
        csv_files.append(file)
    
    if not csv_files:
        print("No matching CSV files found!")
        return
    
    results = []
    
    for csv_file in sorted(csv_files):
        eps = extract_eps_from_filename(csv_file.name)
        if eps is None:
            continue
            
        avg_f1 = calculate_average_f1(csv_file)
        if avg_f1 is not None:
            results.append((eps, avg_f1))
            print(f"eps={eps:.1e}: Average F1 = {avg_f1:.3f}")
    
    if not results:
        print("No valid F1 scores found!")
        return
    
    # Find optimal eps (highest F1 score)
    optimal_eps, optimal_f1 = max(results, key=lambda x: x[1])
    
    print(f"\nOptimal eps value: {optimal_eps:.1e}")
    print(f"Optimal average F1 score: {optimal_f1:.3f}")

if __name__ == "__main__":
    analyze_csv_files()