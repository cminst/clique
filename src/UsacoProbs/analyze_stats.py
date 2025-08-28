#!/usr/bin/env python3
import csv
import sys
import argparse
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

def calculate_numeric_stats(values):
    """Calculate comprehensive statistics for numeric values."""
    if not values:
        return {
            'count': 0, 'mean': 'N/A', 'median': 'N/A', 'std': 'N/A',
            'min': 'N/A', 'max': 'N/A', 'sum': 'N/A'
        }

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        'count': n,
        'mean': f"{mean(values):.3f}",
        'median': f"{sorted_values[n//2] if n % 2 == 1 else (sorted_values[n//2-1] + sorted_values[n//2])/2:.3f}",
        'std': f"{stdev(values) if n > 1 else 0:.3f}",
        'min': f"{min(values):.3f}",
        'max': f"{max(values):.3f}",
        'sum': f"{sum(values):.3f}"
    }

def analyze_csv_file(csv_file):
    """Analyze a CSV file and return comprehensive statistics."""
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                print(f"Error: No columns found in {csv_file}")
                return None

            data = list(reader)
            if not data:
                print(f"Error: No data found in {csv_file}")
                return None

            # Analyze each column
            column_stats = {}
            method_stats = defaultdict(list)

            for column in reader.fieldnames:
                numeric_values = []
                text_values = []

                for row in data:
                    value = row[column]
                    if value.strip():
                        try:
                            numeric_values.append(float(value))
                        except ValueError:
                            text_values.append(value.strip())

                if numeric_values:
                    column_stats[column] = {
                        'type': 'numeric',
                        'stats': calculate_numeric_stats(numeric_values),
                        'unique_values': len(set(numeric_values))
                    }
                else:
                    column_stats[column] = {
                        'type': 'categorical',
                        'unique_values': len(set(text_values)),
                        'value_counts': defaultdict(int)
                    }
                    for val in text_values:
                        column_stats[column]['value_counts'][val] += 1

            # Group by method if Method column exists
            if 'Method' in reader.fieldnames:
                for row in data:
                    method = row['Method']
                    for column in reader.fieldnames:
                        if column != 'Method':
                            try:
                                value = float(row[column])
                                method_stats[(method, column)].append(value)
                            except ValueError:
                                pass

            return {
                'file_path': csv_file,
                'total_rows': len(data),
                'columns': reader.fieldnames,
                'column_stats': column_stats,
                'method_stats': method_stats
            }

    except Exception as e:
        print(f"Error analyzing {csv_file}: {e}")
        return None

def display_summary_table(console, analysis_result):
    """Display a summary table with basic statistics."""
    table = Table(title=f"Summary Statistics for {Path(analysis_result['file_path']).name}")

    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Mean", style="yellow", justify="right")
    table.add_column("Std Dev", style="yellow", justify="right")
    table.add_column("Min", style="blue", justify="right")
    table.add_column("Max", style="blue", justify="right")
    table.add_column("Unique", style="red", justify="right")

    for column, stats in analysis_result['column_stats'].items():
        if stats['type'] == 'numeric':
            s = stats['stats']
            table.add_row(
                column,
                "Numeric",
                str(s['count']),
                s['mean'],
                s['std'],
                s['min'],
                s['max'],
                str(stats['unique_values'])
            )
        else:
            table.add_row(
                column,
                "Categorical",
                str(stats['unique_values']),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                str(stats['unique_values'])
            )

    console.print(table)

def display_detailed_table(console, analysis_result):
    """Display detailed statistics for numeric columns."""
    numeric_columns = [(col, stats) for col, stats in analysis_result['column_stats'].items()
                      if stats['type'] == 'numeric']

    if not numeric_columns:
        return

    table = Table(title="Detailed Statistics for Numeric Columns")

    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Count", style="green", justify="right")
    table.add_column("Mean", style="yellow", justify="right")
    table.add_column("Median", style="yellow", justify="right")
    table.add_column("Std Dev", style="bright_yellow", justify="right")
    table.add_column("Min", style="blue", justify="right")
    table.add_column("Max", style="blue", justify="right")
    table.add_column("Sum", style="purple", justify="right")

    for column, stats in numeric_columns:
        s = stats['stats']
        table.add_row(
            column,
            str(s['count']),
            s['mean'],
            s['median'],
            s['std'],
            s['min'],
            s['max'],
            s['sum']
        )

    console.print(table)

def display_method_analysis(console, analysis_result):
    """Display analysis grouped by method."""
    if not analysis_result['method_stats']:
        return

    # Get all unique methods and columns
    methods = set()
    columns = set()
    for (method, column) in analysis_result['method_stats'].keys():
        methods.add(method)
        columns.add(column)

    excluded_columns = []

    for column in sorted(columns):
        # Check if all methods have the same average for this column
        column_means = []
        for method in sorted(methods):
            values = analysis_result['method_stats'].get((method, column), [])
            if values:
                stats = calculate_numeric_stats(values)
                column_means.append(float(stats['mean']))

        # Skip if all means are identical (within floating point precision)
        if len(set(round(mean, 10) for mean in column_means)) <= 1:
            excluded_columns.append(column)
            continue

        table = Table(title=f"{column} by Method")

        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Count", style="green", justify="right")
        table.add_column("Mean", style="yellow", justify="right")
        table.add_column("Std Dev", style="bright_yellow", justify="right")
        table.add_column("Min", style="blue", justify="right")
        table.add_column("Max", style="blue", justify="right")

        for method in sorted(methods):
            values = analysis_result['method_stats'].get((method, column), [])
            if values:
                stats = calculate_numeric_stats(values)
                table.add_row(
                    method,
                    str(stats['count']),
                    stats['mean'],
                    stats['std'],
                    stats['min'],
                    stats['max']
                )

        console.print(table)

    # Show excluded columns note
    if excluded_columns:
        console.print(f"\n[dim]Excluded columns (identical averages across all methods): {', '.join(excluded_columns)}[/dim]\n")

def display_categorical_analysis(console, analysis_result):
    """Display analysis for categorical columns."""
    categorical_columns = [(col, stats) for col, stats in analysis_result['column_stats'].items()
                          if stats['type'] == 'categorical']

    for column, stats in categorical_columns:
        if stats['value_counts']:
            table = Table(title=f"Distribution of {column}")

            table.add_column("Value", style="cyan", no_wrap=True)
            table.add_column("Count", style="green", justify="right")
            table.add_column("Percentage", style="yellow", justify="right")

            total = sum(stats['value_counts'].values())
            sorted_values = sorted(stats['value_counts'].items(), key=lambda x: x[1], reverse=True)

            for value, count in sorted_values[:10]:  # Show top 10
                percentage = (count / total) * 100
                table.add_row(value, str(count), f"{percentage:.1f}%")

            if len(sorted_values) > 10:
                table.add_row("...", f"+{len(sorted_values) - 10} more", "")

            console.print(table)

def main():
    parser = argparse.ArgumentParser(description='Analyze CSV file with comprehensive statistics')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary statistics')

    args = parser.parse_args()

    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        print(f"Error: File {csv_file} does not exist!")
        sys.exit(1)

    console = Console()

    # Print file info panel
    console.print(Panel(
        f"[bold blue]Analyzing:[/] {csv_file.name}\n[bold blue]Full path:[/] {csv_file.absolute()}",
        title="Synthetic Graph CSV Analysis",
        border_style="blue"
    ))

    # Analyze the file
    analysis_result = analyze_csv_file(csv_file)
    if not analysis_result:
        sys.exit(1)

    # Display results
    console.print(f"\n[bold green]Total Rows:[/] {analysis_result['total_rows']}")
    console.print(f"[bold green]Total Columns:[/] {len(analysis_result['columns'])}")

    display_summary_table(console, analysis_result)

    if not args.summary_only:
        display_detailed_table(console, analysis_result)
        display_method_analysis(console, analysis_result)
        display_categorical_analysis(console, analysis_result)

if __name__ == "__main__":
    main()
