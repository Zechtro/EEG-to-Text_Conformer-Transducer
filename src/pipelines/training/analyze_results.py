#!/usr/bin/env python3
"""
Script untuk visualisasi training history dari all0.py
Dapat dijalankan setelah training selesai untuk analisis lebih detail
"""

import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Setup
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
HISTORY_FILE = os.path.join(PROJECT_ROOT, 'training_history.json')
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'test_results.csv')

def plot_detailed_history(history_data, output_dir):
    """Create detailed visualization of training history"""
    
    # Extract data
    epochs = range(1, len(history_data['train_loss']) + 1)
    train_loss = history_data['train_loss']
    val_loss = history_data['val_loss']
    train_cer = history_data.get('train_cer', [0] * len(train_loss))  # backward compatibility
    val_cer = history_data['val_cer']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ===== Plot 1: Train vs Val Loss =====
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, train_loss, 'b-', linewidth=2.5, marker='o', markersize=4, label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', linewidth=2.5, marker='s', markersize=4, label='Val Loss')
    ax1.fill_between(epochs, train_loss, val_loss, alpha=0.2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training vs Validation Loss over Epochs', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.4, linestyle='--')
    
    # ===== Plot 2: Train vs Val CER =====
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, train_cer, 'g-', linewidth=2, marker='^', markersize=5, label='Train CER')
    ax2.plot(epochs, val_cer, 'm-', linewidth=2, marker='v', markersize=5, label='Val CER')
    ax2.fill_between(epochs, train_cer, val_cer, alpha=0.2, color='cyan')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('CER', fontsize=11, fontweight='bold')
    ax2.set_title('Train vs Val CER', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    
    # ===== Plot 3: Loss Convergence (log scale) =====
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(epochs, train_loss, 'b-', linewidth=2, marker='o', label='Train Loss')
    ax3.semilogy(epochs, val_loss, 'r-', linewidth=2, marker='s', label='Val Loss')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss (log scale)', fontsize=11, fontweight='bold')
    ax3.set_title('Loss Convergence (Log Scale)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.4, which='both', linestyle='--')
    
    # ===== Plot 4: CER on Log Scale =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(epochs, train_cer, 'g-', linewidth=2, marker='^', label='Train CER')
    ax4.semilogy(epochs, val_cer, 'm-', linewidth=2, marker='v', label='Val CER')
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('CER (log scale)', fontsize=11, fontweight='bold')
    ax4.set_title('CER Convergence (Log Scale)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.4, which='both', linestyle='--')
    
    # ===== Plot 5: Loss Improvement Stats =====
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate improvements
    train_loss_improv = 100 * (train_loss[0] - train_loss[-1]) / (train_loss[0] + 1e-9)
    val_loss_improv = 100 * (val_loss[0] - val_loss[-1]) / (val_loss[0] + 1e-9)
    train_cer_improv = 100 * (train_cer[0] - train_cer[-1]) / (train_cer[0] + 1e-9) if train_cer[0] > 0 else 0
    val_cer_improv = 100 * (val_cer[0] - val_cer[-1]) / (val_cer[0] + 1e-9)
    
    metrics_names = ['Train Loss', 'Val Loss', 'Train CER', 'Val CER']
    metrics_values = [train_loss_improv, val_loss_improv, train_cer_improv, val_cer_improv]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax5.bar(range(len(metrics_names)), metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_xticks(range(len(metrics_names)))
    ax5.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Training Improvements', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ===== Plot 6: Training Statistics =====
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Calculate stats
    best_train_cer_epoch = train_cer.index(min(train_cer)) + 1 if train_cer else 1
    best_val_loss_epoch = val_loss.index(min(val_loss)) + 1
    best_val_cer_epoch = val_cer.index(min(val_cer)) + 1
    best_train_cer = min(train_cer) if train_cer else 0
    best_val_loss = min(val_loss)
    best_val_cer = min(val_cer)
    final_train_loss = train_loss[-1]
    final_train_cer = train_cer[-1] if train_cer else 0
    final_val_loss = val_loss[-1]
    final_val_cer = val_cer[-1]
    
    stats_text = f"""
DETAILED TRAINING STATISTICS
{'─' * 100}

📊 LOSS METRICS:
   Initial Train Loss:        {train_loss[0]:.6f}    →    Final Train Loss:        {final_train_loss:.6f}    (↓ {train_loss_improv:.1f}%)
   Initial Val Loss:          {val_loss[0]:.6f}    →    Final Val Loss:          {final_val_loss:.6f}    (↓ {val_loss_improv:.1f}%)
   Best Val Loss:             {best_val_loss:.6f}  (Epoch {best_val_loss_epoch})

📊 CER METRICS:
   Initial Train CER:         {train_cer[0]:.4f}       →    Final Train CER:         {final_train_cer:.4f}       (↓ {train_cer_improv:.1f}%)
   Initial Val CER:           {val_cer[0]:.4f}       →    Final Val CER:           {final_val_cer:.4f}       (↓ {val_cer_improv:.1f}%)
   Best Train CER:            {best_train_cer:.4f}  (Epoch {best_train_cer_epoch})    |    Best Val CER:             {best_val_cer:.4f}  (Epoch {best_val_cer_epoch})

📊 GENERAL:
   Total Epochs:              {len(epochs)}
   Gap (Final):               Train Loss: {abs(final_train_loss - final_val_loss):.6f}    |    CER: {abs(final_train_cer - final_val_cer):.4f}
    """
    
    ax6.text(0.02, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    # Save figure
    plot_path = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed training analysis saved to: {plot_path}")
    plt.close()

def plot_test_results_stats(results_file, output_dir):
    """Visualize test results statistics"""
    
    if not os.path.exists(results_file):
        print(f"⚠ Test results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: CER Distribution
    axes[0, 0].hist(df['cer'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['cer'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["cer"].mean():.3f}')
    axes[0, 0].axvline(df['cer'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["cer"].median():.3f}')
    axes[0, 0].set_xlabel('Character Error Rate', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('CER Distribution on Test Set', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: CER by Gender
    if 'gender' in df.columns:
        df.boxplot(column='cer', by='gender', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Gender', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('CER', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('CER by Gender', fontsize=12, fontweight='bold')
        axes[0, 1].get_figure().suptitle('')  # Remove default title
    
    # Plot 3: CER by Subject (top 10)
    if 'subject' in df.columns:
        subject_stats = df.groupby('subject')['cer'].agg(['mean', 'count']).sort_values('mean').head(10)
        axes[1, 0].barh(range(len(subject_stats)), subject_stats['mean'], color='coral', edgecolor='black')
        axes[1, 0].set_yticks(range(len(subject_stats)))
        axes[1, 0].set_yticklabels(subject_stats.index)
        axes[1, 0].set_xlabel('Average CER', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Top 10 Subjects by Average CER', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Test Results Statistics
    axes[1, 1].axis('off')
    
    stats_text = f"""
TEST RESULTS SUMMARY
{'─' * 45}

📊 Total Samples: {len(df)}

📊 CER Statistics:
   Mean:       {df['cer'].mean():.4f}
   Median:     {df['cer'].median():.4f}
   Std Dev:    {df['cer'].std():.4f}
   Min:        {df['cer'].min():.4f}
   Max:        {df['cer'].max():.4f}

📊 CER Categories:
   Perfect (0.00):    {(df['cer'] == 0).sum()} ({100*(df['cer'] == 0).sum()/len(df):.1f}%)
   Excellent (<0.1):  {(df['cer'] < 0.1).sum()} ({100*(df['cer'] < 0.1).sum()/len(df):.1f}%)
   Good (< 0.2):      {(df['cer'] < 0.2).sum()} ({100*(df['cer'] < 0.2).sum()/len(df):.1f}%)
   Fair (< 0.4):      {(df['cer'] < 0.4).sum()} ({100*(df['cer'] < 0.4).sum()/len(df):.1f}%)

📊 Unique Subjects: {df['subject'].nunique() if 'subject' in df.columns else 'N/A'}
📊 Gender Split:
   {df['gender'].value_counts().to_string() if 'gender' in df.columns else 'N/A'}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=9.5, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'test_results_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Test results analysis saved to: {plot_path}")
    plt.close()

def main():
    print("=" * 70)
    print("Training History & Test Results Visualization")
    print("=" * 70)
    
    # Load training history
    if os.path.exists(HISTORY_FILE):
        print(f"\n📂 Loading training history from: {HISTORY_FILE}")
        with open(HISTORY_FILE, 'r') as f:
            history_data = json.load(f)
        
        print(f"   Epochs: {len(history_data['train_loss'])}")
        print(f"   Best Val Loss: {min(history_data['val_loss']):.6f}")
        print(f"   Best Val CER: {min(history_data['val_cer']):.4f}")
        
        # Create plots
        plot_detailed_history(history_data, PROJECT_ROOT)
    else:
        print(f"✗ Training history file not found: {HISTORY_FILE}")
    
    # Load test results
    if os.path.exists(RESULTS_FILE):
        print(f"\n📂 Loading test results from: {RESULTS_FILE}")
        plot_test_results_stats(RESULTS_FILE, PROJECT_ROOT)
    else:
        print(f"✗ Test results file not found: {RESULTS_FILE}")
    
    print("\n" + "=" * 70)
    print("✓ Visualization complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
