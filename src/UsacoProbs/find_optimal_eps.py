import csv
import re
from pathlib import Path

def extract_eps_from_filename(filename):
    """Extract eps value from filename using regex."""
    match = re.search(r'eps=([0-9.]+(?:e[+-]?\d+)?)', filename)
    if match:
        return float(match.group(1))
    return None

def calculate_average_f1(csv_file, method_name):
    """Calculate average F1 score for a given method in a CSV file."""
    f1_scores = []

    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Method'] == method_name:
                    f1_scores.append(float(row['F1']))
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

    if not f1_scores:
        return None

    return sum(f1_scores) / len(f1_scores)

def analyze_csv_files():
    """Analyze all CSV files and find optimal eps value for L-RMC, then compare other methods at that eps."""
    csv_files = []

    # Find all CSV files matching the pattern
    for file in Path('.').glob('Hard-nTotal=2500,eps=*.csv'):
        csv_files.append(file)

    if not csv_files:
        print("No matching CSV files found!")
        return

    lrmc_results = []

    for csv_file in sorted(csv_files):
        eps = extract_eps_from_filename(csv_file.name)
        if eps is None:
            continue

        avg_f1 = calculate_average_f1(csv_file, 'L-RMC')
        if avg_f1 is not None:
            lrmc_results.append((eps, avg_f1))
            print(f"eps={eps:.1e}: Average F1 = {avg_f1:.3f}")

    if not lrmc_results:
        print("No valid L-RMC F1 scores found!")
        return

    # Find optimal eps (highest F1 score for L-RMC)
    optimal_eps, optimal_f1 = max(lrmc_results, key=lambda x: x[1])

    print(f"\nOptimal eps value: {optimal_eps:.1e}")
    print(f"Optimal average F1 score (L-RMC): {optimal_f1:.3f}")

    # Now find the file corresponding to optimal_eps
    optimal_file = None
    for file in csv_files:
        if abs(extract_eps_from_filename(file.name) - optimal_eps) < 1e-10:  # floating point comparison
            optimal_file = file
            break

    if optimal_file is None:
        print("Could not locate file for optimal eps!")
        return

    # Extract all methods present in the optimal file
    methods = set()
    try:
        with open(optimal_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                methods.add(row['Method'])
    except Exception as e:
        print(f"Error reading {optimal_file} to detect methods: {e}")
        return

    # Calculate average F1 for all methods in the optimal file
    print(f"(eps={optimal_eps}) F1 scores for all methods:")
    all_method_results = []
    for method in sorted(methods):
        avg_f1 = calculate_average_f1(optimal_file, method)
        if avg_f1 is not None:
            all_method_results.append((method, avg_f1))
            print(f"  {method}: Average F1 = {avg_f1:.3f}")

if __name__ == "__main__":
    analyze_csv_files()
