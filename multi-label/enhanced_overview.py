"""
Enhanced overview script for multi-label classification data.
"""

import pandas as pd
import argparse
from multi_label_utils import (
    get_multi_label_stats, 
    get_label_cooccurrence_matrix,
    filter_by_labels,
    create_binary_matrix
)


def print_detailed_overview(df: pd.DataFrame, output_file: str = None):
    """Print detailed overview of multi-label dataset."""
    
    # Basic info
    print("=" * 80)
    print("MULTI-LABEL DATASET OVERVIEW")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Identify label columns (all except description)
    label_columns = [col for col in df.columns if col not in ['description']]
    print(f"Label columns: {label_columns}")
    
    # Get comprehensive stats
    stats = get_multi_label_stats(df, label_columns)
    
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS BY DIMENSION")
    print("=" * 80)
    
    for col, col_stats in stats.items():
        print(f"\n--- {col.upper()} ---")
        print(f"Records with labels: {col_stats['records_with_labels']}/{col_stats['total_records']} "
              f"({col_stats['records_with_labels']/col_stats['total_records']*100:.1f}%)")
        print(f"Records with multiple labels: {col_stats['records_with_multiple_labels']} "
              f"({col_stats['records_with_multiple_labels']/col_stats['total_records']*100:.1f}%)")
        print(f"Unique labels: {len(col_stats['unique_labels'])}")
        print(f"Avg labels per record: {col_stats['avg_labels_per_record']:.2f}")
        print(f"Max labels per record: {col_stats['max_labels_per_record']}")
        print(f"Min labels per record: {col_stats['min_labels_per_record']}")
        
        print(f"\nLabel frequency distribution:")
        total_records = col_stats['total_records']
        for label, count in col_stats['label_counts'].most_common():
            percentage = (count / total_records * 100)
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("CO-OCCURRENCE ANALYSIS")
    print("=" * 80)
    
    for col in label_columns:
        print(f"\n--- {col.upper()} LABEL CO-OCCURRENCE ---")
        try:
            cooccurrence_df = get_label_cooccurrence_matrix(df, col)
            
            # Show top co-occurrences (off-diagonal elements)
            print("Top label co-occurrences:")
            cooccurrences = []
            for i in range(len(cooccurrence_df)):
                for j in range(i+1, len(cooccurrence_df)):
                    label1 = cooccurrence_df.index[i]
                    label2 = cooccurrence_df.columns[j]
                    count = cooccurrence_df.iloc[i, j]
                    if count > 0:
                        cooccurrences.append((label1, label2, count))
            
            cooccurrences.sort(key=lambda x: x[2], reverse=True)
            for label1, label2, count in cooccurrences[:10]:  # Top 10
                percentage = (count / len(df) * 100)
                print(f"  {label1} + {label2}: {count} ({percentage:.1f}%)")
            
            if not cooccurrences:
                print("  No co-occurrences found (all records have single labels)")
                
        except Exception as e:
            print(f"  Error calculating co-occurrence: {e}")
    
    # Save to file if requested
    if output_file:
        print(f"\n" + "=" * 80)
        print("SAVING DETAILED REPORT")
        print("=" * 80)
        
        with open(output_file, 'w') as f:
            f.write("MULTI-LABEL DATASET REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Label columns: {label_columns}\n\n")
            
            for col, col_stats in stats.items():
                f.write(f"\n{col.upper()} STATISTICS:\n")
                f.write(f"- Records with labels: {col_stats['records_with_labels']}\n")
                f.write(f"- Records with multiple labels: {col_stats['records_with_multiple_labels']}\n")
                f.write(f"- Unique labels: {len(col_stats['unique_labels'])}\n")
                f.write(f"- Average labels per record: {col_stats['avg_labels_per_record']:.2f}\n")
                f.write(f"- Label counts:\n")
                for label, count in col_stats['label_counts'].most_common():
                    f.write(f"  {label}: {count}\n")
        
        print(f"✅ Detailed report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-label dataset overview")
    parser.add_argument(
        "--file", 
        type=str, 
        default="temp_output.parquet",
        help="Path to the parquet file to analyze"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Optional: save detailed report to this file"
    )
    parser.add_argument(
        "--filter-labels",
        type=str,
        nargs="+",
        help="Filter data to show only records with these labels"
    )
    parser.add_argument(
        "--filter-column",
        type=str,
        default="purpose",
        help="Column to apply label filter to"
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["any", "all"],
        default="any",
        help="Filter mode: 'any' (has any of the labels) or 'all' (has all labels)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.file}...")
        df = pd.read_parquet(args.file)
        
        # Apply filter if requested
        if args.filter_labels:
            print(f"Filtering by {args.filter_column} labels: {args.filter_labels} (mode: {args.filter_mode})")
            original_size = len(df)
            df = filter_by_labels(df, args.filter_column, args.filter_labels, args.filter_mode)
            print(f"Filtered dataset: {len(df)}/{original_size} records")
        
        # Generate overview
        print_detailed_overview(df, args.output_report)
        
    except FileNotFoundError:
        print(f"❌ Error: File {args.file} not found!")
        print("Make sure you have generated a labelled dataset first.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
