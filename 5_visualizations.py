import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_fairness_improvements():
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Fully populated Multi-Dimensional Metrics from Log Outputs
    records = [
        # --- Accuracy ---
        {'Dimension': 'Geography (Rural)', 'Model': 'Baseline', 'Metric': 'Accuracy', 'Value': 0.7235},
        {'Dimension': 'Geography (Rural)', 'Model': 'Adversarial', 'Metric': 'Accuracy', 'Value': 0.7300},
        {'Dimension': 'Religion (Minority)', 'Model': 'Baseline', 'Metric': 'Accuracy', 'Value': 0.7235},
        {'Dimension': 'Religion (Minority)', 'Model': 'Adversarial', 'Metric': 'Accuracy', 'Value': 0.7344},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Baseline', 'Metric': 'Accuracy', 'Value': 0.7235},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Adversarial', 'Metric': 'Accuracy', 'Value': 0.7298},
        
        # --- Disparate Impact ---
        {'Dimension': 'Geography (Rural)', 'Model': 'Baseline', 'Metric': 'Disparate Impact', 'Value': 1.0024},
        {'Dimension': 'Geography (Rural)', 'Model': 'Adversarial', 'Metric': 'Disparate Impact', 'Value': 1.0288},
        {'Dimension': 'Religion (Minority)', 'Model': 'Baseline', 'Metric': 'Disparate Impact', 'Value': 0.9403},
        {'Dimension': 'Religion (Minority)', 'Model': 'Adversarial', 'Metric': 'Disparate Impact', 'Value': 0.9331},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Baseline', 'Metric': 'Disparate Impact', 'Value': 0.9351},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Adversarial', 'Metric': 'Disparate Impact', 'Value': 0.8133},

        # --- Equal Opportunity Difference ---
        {'Dimension': 'Geography (Rural)', 'Model': 'Baseline', 'Metric': 'Eq. Opp. Diff', 'Value': 0.0000},
        {'Dimension': 'Geography (Rural)', 'Model': 'Adversarial', 'Metric': 'Eq. Opp. Diff', 'Value': 0.0149},
        {'Dimension': 'Religion (Minority)', 'Model': 'Baseline', 'Metric': 'Eq. Opp. Diff', 'Value': -0.0387},
        {'Dimension': 'Religion (Minority)', 'Model': 'Adversarial', 'Metric': 'Eq. Opp. Diff', 'Value': -0.0438},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Baseline', 'Metric': 'Eq. Opp. Diff', 'Value': -0.0371},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Adversarial', 'Metric': 'Eq. Opp. Diff', 'Value': -0.1314},

        # --- Statistical Parity Difference ---
        {'Dimension': 'Geography (Rural)', 'Model': 'Baseline', 'Metric': 'Stat. Parity Diff', 'Value': 0.0022},
        {'Dimension': 'Geography (Rural)', 'Model': 'Adversarial', 'Metric': 'Stat. Parity Diff', 'Value': 0.0253},
        {'Dimension': 'Religion (Minority)', 'Model': 'Baseline', 'Metric': 'Stat. Parity Diff', 'Value': -0.0553},
        {'Dimension': 'Religion (Minority)', 'Model': 'Adversarial', 'Metric': 'Stat. Parity Diff', 'Value': -0.0601},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Baseline', 'Metric': 'Stat. Parity Diff', 'Value': -0.0627},
        {'Dimension': 'Caste (Marginalized)', 'Model': 'Adversarial', 'Metric': 'Stat. Parity Diff', 'Value': -0.1847},
    ]
    df_metrics = pd.DataFrame(records)
    
    # Create the visualization (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Multi-Dimensional Fairness Audit: Adversarial Debiasing Across Geog, Religion, & Caste', fontsize=18, y=1.02)
    
    palette = ['#3498DB', '#E74C3C'] # Blue for Baseline, Red for Adversarial

    metrics_list = [
        ('Accuracy', 'Predictive Accuracy', [0, 1.0], None),
        ('Disparate Impact', 'Disparate Impact (Ratio)', [0, 1.2], 1.0),
        ('Eq. Opp. Diff', 'Equal Opportunity Difference', [-0.15, 0.15], 0.0),
        ('Stat. Parity Diff', 'Statistical Parity Difference', [-0.15, 0.15], 0.0)
    ]
    
    for idx, (metric_name, title, ylims, ideal_line) in enumerate(metrics_list):
        ax = axes[idx]
        sns.barplot(x='Dimension', y='Value', hue='Model', 
                    data=df_metrics[df_metrics['Metric'] == metric_name], 
                    ax=ax, palette=palette)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_ylabel('Score / Ratio' if idx < 2 else 'Difference')
        ax.set_xlabel('')
        
        if ideal_line is not None:
            ax.axhline(y=ideal_line, color='k', linestyle='--', label=f'Ideal ({ideal_line})', alpha=0.6)
            
        if idx == 0:
            ax.legend(loc='lower right')
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
                
        # Rotate x labels for clean look
        ax.tick_params(axis='x', rotation=15)
        
        
    plt.tight_layout()
    plt.savefig('/Users/aakashsangani/Desktop/CreditAudit/fairness_results_plot.png', dpi=300, bbox_inches='tight')
    
    print("Optimization graph saved as /Users/aakashsangani/Desktop/CreditAudit/fairness_results_plot.png")

if __name__ == "__main__":
    visualize_fairness_improvements()

