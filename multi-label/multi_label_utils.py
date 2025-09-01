"""
Utility functions for working with multi-label classification data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer


def parse_multi_labels(label_string: str) -> List[str]:
    """Parse pipe-separated multi-labels into a list."""
    if pd.isna(label_string) or label_string == '':
        return []
    return [label.strip() for label in str(label_string).split('|') if label.strip()]


def labels_to_string(labels: List[str]) -> str:
    """Convert list of labels to pipe-separated string."""
    return '|'.join(labels) if labels else ''


def get_all_unique_labels(df: pd.DataFrame, label_column: str) -> Set[str]:
    """Get all unique labels from a multi-label column."""
    all_labels = set()
    for item in df[label_column]:
        labels = parse_multi_labels(item)
        all_labels.update(labels)
    return all_labels


def create_binary_matrix(df: pd.DataFrame, label_column: str) -> tuple[np.ndarray, List[str]]:
    """
    Create a binary matrix representation of multi-labels.
    
    Returns:
        binary_matrix: numpy array where each row is a sample and each column is a label
        label_names: list of label names corresponding to columns
    """
    # Parse all labels
    label_lists = [parse_multi_labels(item) for item in df[label_column]]
    
    # Create binary matrix
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(label_lists)
    
    return binary_matrix, list(mlb.classes_)


def get_label_cooccurrence_matrix(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Create a co-occurrence matrix showing how often labels appear together.
    """
    binary_matrix, label_names = create_binary_matrix(df, label_column)
    
    # Calculate co-occurrence
    cooccurrence = np.dot(binary_matrix.T, binary_matrix)
    
    # Create DataFrame
    cooccurrence_df = pd.DataFrame(
        cooccurrence, 
        index=label_names, 
        columns=label_names
    )
    
    return cooccurrence_df


def get_multi_label_stats(df: pd.DataFrame, label_columns: List[str]) -> Dict:
    """
    Get comprehensive statistics for multi-label data.
    """
    stats = {}
    
    for col in label_columns:
        col_stats = {
            'total_records': len(df),
            'records_with_labels': 0,
            'records_with_multiple_labels': 0,
            'unique_labels': set(),
            'label_counts': Counter(),
            'avg_labels_per_record': 0,
            'max_labels_per_record': 0,
            'min_labels_per_record': float('inf')
        }
        
        label_counts_per_record = []
        
        for item in df[col]:
            labels = parse_multi_labels(item)
            num_labels = len(labels)
            
            if num_labels > 0:
                col_stats['records_with_labels'] += 1
                col_stats['unique_labels'].update(labels)
                col_stats['label_counts'].update(labels)
                
                if num_labels > 1:
                    col_stats['records_with_multiple_labels'] += 1
                
                label_counts_per_record.append(num_labels)
                col_stats['max_labels_per_record'] = max(col_stats['max_labels_per_record'], num_labels)
                col_stats['min_labels_per_record'] = min(col_stats['min_labels_per_record'], num_labels)
        
        if label_counts_per_record:
            col_stats['avg_labels_per_record'] = np.mean(label_counts_per_record)
        else:
            col_stats['min_labels_per_record'] = 0
        
        col_stats['unique_labels'] = list(col_stats['unique_labels'])
        stats[col] = col_stats
    
    return stats


def filter_by_labels(df: pd.DataFrame, label_column: str, required_labels: List[str], 
                    mode: str = 'any') -> pd.DataFrame:
    """
    Filter dataframe by labels.
    
    Args:
        df: DataFrame to filter
        label_column: Column containing multi-labels
        required_labels: Labels to filter by
        mode: 'any' (has any of the labels) or 'all' (has all of the labels)
    """
    def has_labels(label_string):
        labels = set(parse_multi_labels(label_string))
        required_set = set(required_labels)
        
        if mode == 'any':
            return bool(labels.intersection(required_set))
        elif mode == 'all':
            return required_set.issubset(labels)
        else:
            raise ValueError("mode must be 'any' or 'all'")
    
    return df[df[label_column].apply(has_labels)]


def convert_to_sklearn_format(df: pd.DataFrame, text_column: str, label_columns: List[str]) -> tuple:
    """
    Convert multi-label dataframe to sklearn-compatible format.
    
    Returns:
        X: text data (list or array)
        y_dict: dictionary with label_column -> binary_matrix mappings
        label_names_dict: dictionary with label_column -> label_names mappings
    """
    X = df[text_column].tolist()
    y_dict = {}
    label_names_dict = {}
    
    for col in label_columns:
        binary_matrix, label_names = create_binary_matrix(df, col)
        y_dict[col] = binary_matrix
        label_names_dict[col] = label_names
    
    return X, y_dict, label_names_dict


if __name__ == "__main__":
    # Example usage
    print("Multi-label utilities loaded successfully!")
    print("Available functions:")
    print("- parse_multi_labels()")
    print("- labels_to_string()")
    print("- get_all_unique_labels()")
    print("- create_binary_matrix()")
    print("- get_label_cooccurrence_matrix()")
    print("- get_multi_label_stats()")
    print("- filter_by_labels()")
    print("- convert_to_sklearn_format()")
