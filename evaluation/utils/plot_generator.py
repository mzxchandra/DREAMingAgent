import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Any

def plot_confusion_matrix(
    tp: int, fp: int, tn: int, fn: int, 
    output_path: Path,
    title: str = "Confusion Matrix"
):
    """Plot a confusion matrix heat map."""
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [['TN', 'FP'], ['FN', 'TP']]
    
    # Format text for annotations
    annot = np.array([[f"{l}\n{v}" for l, v in zip(row_l, row_v)] 
                      for row_l, row_v in zip(labels, matrix)])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=annot, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Predicted False', 'Predicted True'],
                yticklabels=['Actual False', 'Actual True'])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_bar_chart(
    categories: List[str], 
    values: List[float], 
    output_path: Path,
    ylabel: str = "Values", 
    title: str = "Bar Chart",
    ylim: Optional[tuple] = None
):
    """Plot a generic bar chart."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color='skyblue', edgecolor='navy')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_histogram(
    data: List[float], 
    output_path: Path,
    xlabel: str = "Value", 
    title: str = "Histogram",
    bins: int = 20
):
    """Plot a histogram."""
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scatter(
    x: List[float],
    y: List[float],
    output_path: Path,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Scatter Plot",
    labels: Optional[List[str]] = None
):
    """Plot a scatter plot, optionally with labels."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, edgecolors='w', s=80)
    
    if labels:
        # Avoid clutter if too many labels
        if len(labels) < 30:
            for i, txt in enumerate(labels):
                plt.annotate(txt, (x[i], y[i]), fontsize=8)
                
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
