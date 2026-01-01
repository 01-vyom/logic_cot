import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from collections import Counter
import re
from datetime import datetime

warnings.filterwarnings("ignore")

# --- Configuration ---
RESULTS_DIR = "/home/ec2-user/code/personal/logic_cot/experiment_results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "irac_analysis_summary.csv")
DETAILED_CSV = os.path.join(RESULTS_DIR, "irac_detailed_probe_analysis.csv")

# Plot styling configuration
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "Set2"
FONT_SIZE = 12
TITLE_SIZE = 14

def setup_plotting_style():
    """Configure matplotlib and seaborn styling for publication-ready plots."""
    plt.style.use('default')
    sns.set_palette(COLOR_PALETTE)
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.titlesize': TITLE_SIZE + 2,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def load_data():
    """Load the analysis results from CSV files."""
    print(f"Loading data from {RESULTS_DIR}...")
    
    # Load summary data
    if os.path.exists(SUMMARY_CSV):
        df_summary = pd.read_csv(SUMMARY_CSV)
        print(f"Loaded summary data: {len(df_summary)} questions")
    else:
        print(f"Warning: Summary CSV not found at {SUMMARY_CSV}")
        df_summary = pd.DataFrame()
    
    # Load detailed probe data
    if os.path.exists(DETAILED_CSV):
        df_detailed = pd.read_csv(DETAILED_CSV)
        print(f"Loaded detailed data: {len(df_detailed)} probes")
    else:
        print(f"Warning: Detailed CSV not found at {DETAILED_CSV}")
        df_detailed = pd.DataFrame()
    
    return df_summary, df_detailed

def create_suir_detection_overview(df_summary, save_path):
    """Create plots showing SUIR detection overview."""
    if df_summary.empty:
        print("Skipping SUIR detection overview - no summary data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SUIR Detection Overview', fontsize=TITLE_SIZE + 4, y=0.98)
    
    # 1. Distribution of SUIR detection rates
    axes[0, 0].hist(df_summary['suir_detection_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('SUIR Detection Rate')
    axes[0, 0].set_ylabel('Number of Questions')
    axes[0, 0].set_title('Distribution of SUIR Detection Rates')
    axes[0, 0].axvline(df_summary['suir_detection_rate'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_summary["suir_detection_rate"].mean():.3f}')
    axes[0, 0].legend()
    
    # 2. Box plot of detection rates
    box_data = df_summary['suir_detection_rate'].dropna()
    if len(box_data) > 0:
        axes[0, 1].boxplot(box_data)
        axes[0, 1].set_ylabel('SUIR Detection Rate')
        axes[0, 1].set_title('SUIR Detection Rate Distribution')
        axes[0, 1].set_xticklabels(['All Questions'])
    
    # 3. Questions with any SUIR detection
    suir_detected = (df_summary['suir_detection_rate'] > 0).sum()
    no_suir = (df_summary['suir_detection_rate'] == 0).sum()
    
    labels = ['SUIR Detected', 'No SUIR']
    sizes = [suir_detected, no_suir]
    colors = ['lightcoral', 'lightgray']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Questions with SUIR Detection')
    
    # 4. Hypothesis count vs detection rate
    if 'num_hypotheses' in df_summary.columns:
        axes[1, 1].scatter(df_summary['num_hypotheses'], df_summary['suir_detection_rate'], 
                          alpha=0.6, color='green')
        axes[1, 1].set_xlabel('Number of Hypotheses Tested')
        axes[1, 1].set_ylabel('SUIR Detection Rate')
        axes[1, 1].set_title('Hypotheses vs SUIR Detection')
        
        # Add correlation if enough data points
        valid_data = df_summary[['num_hypotheses', 'suir_detection_rate']].dropna()
        if len(valid_data) > 2:
            correlation = valid_data.corr().iloc[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Corr: {correlation:.3f}', 
                           transform=axes[1, 1].transAxes, verticalalignment='top')
    
    # Adjust layout to prevent title overlap and improve spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved SUIR detection overview to {save_path}")

def create_answer_flip_analysis(df_summary, save_path):
    """Create plots showing answer flip analysis."""
    if df_summary.empty:
        print("Skipping answer flip analysis - no summary data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Answer Flip Analysis', fontsize=TITLE_SIZE + 4, y=0.98)
    
    # 1. Distribution of answer flip rates
    flip_data = df_summary['answer_flip_rate_when_ignoring'].dropna()
    if len(flip_data) > 0:
        axes[0, 0].hist(flip_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 0].set_xlabel('Answer Flip Rate When Ignoring')
        axes[0, 0].set_ylabel('Number of Questions')
        axes[0, 0].set_title('Distribution of Answer Flip Rates')
        axes[0, 0].axvline(flip_data.mean(), color='red', linestyle='--', 
                          label=f'Mean: {flip_data.mean():.3f}')
        axes[0, 0].legend()
    
    # 2. SUIR detection vs Answer flip correlation
    if len(df_summary) > 1:
        valid_data = df_summary[['suir_detection_rate', 'answer_flip_rate_when_ignoring']].dropna()
        if len(valid_data) > 0:
            axes[0, 1].scatter(valid_data['suir_detection_rate'], 
                              valid_data['answer_flip_rate_when_ignoring'], alpha=0.6)
            axes[0, 1].set_xlabel('SUIR Detection Rate')
            axes[0, 1].set_ylabel('Answer Flip Rate')
            axes[0, 1].set_title('SUIR Detection vs Answer Flip Rate')
            
            if len(valid_data) > 2:
                correlation = valid_data.corr().iloc[0, 1]
                axes[0, 1].text(0.05, 0.95, f'Corr: {correlation:.3f}', 
                               transform=axes[0, 1].transAxes, verticalalignment='top')
    
    # 3. High flip rate questions
    high_flip = (df_summary['answer_flip_rate_when_ignoring'] > 0.5).sum()
    some_flip = ((df_summary['answer_flip_rate_when_ignoring'] > 0) & 
                 (df_summary['answer_flip_rate_when_ignoring'] <= 0.5)).sum()
    no_flip = (df_summary['answer_flip_rate_when_ignoring'] == 0).sum()
    
    categories = ['High Flip (>50%)', 'Some Flip (1-50%)', 'No Flip (0%)']
    counts = [high_flip, some_flip, no_flip]
    colors = ['red', 'yellow', 'lightgreen']
    
    bars = axes[1, 0].bar(categories, counts, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Number of Questions')
    axes[1, 0].set_title('Answer Flip Categories')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')
    
    # 4. Box plot comparison
    flip_categories = []
    suir_rates = []
    
    for idx, row in df_summary.iterrows():
        flip_rate = row['answer_flip_rate_when_ignoring']
        suir_rate = row['suir_detection_rate']
        
        if pd.notna(flip_rate) and pd.notna(suir_rate):
            if flip_rate == 0:
                category = 'No Flip'
            elif flip_rate <= 0.5:
                category = 'Some Flip'
            else:
                category = 'High Flip'
            
            flip_categories.append(category)
            suir_rates.append(suir_rate)
    
    if flip_categories:
        flip_df = pd.DataFrame({'Category': flip_categories, 'SUIR_Rate': suir_rates})
        sns.boxplot(data=flip_df, x='Category', y='SUIR_Rate', ax=axes[1, 1])
        axes[1, 1].set_title('SUIR Detection by Flip Category')
        axes[1, 1].set_ylabel('SUIR Detection Rate')
    
    # Adjust layout to prevent title overlap and improve spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved answer flip analysis to {save_path}")

def create_semantic_similarity_analysis(df_summary, save_path):
    """Create plots showing semantic similarity analysis."""
    if df_summary.empty:
        print("Skipping semantic similarity analysis - no summary data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Semantic Similarity Analysis', fontsize=TITLE_SIZE + 4, y=0.98)
    
    # Get similarity data
    sim_ignore = df_summary['avg_cot_sim_base_vs_ignore'].dropna()
    sim_use = df_summary['avg_cot_sim_base_vs_use'].dropna()
    
    # 1. Dual histogram of similarities
    if len(sim_ignore) > 0 or len(sim_use) > 0:
        axes[0, 0].hist(sim_ignore, bins=15, alpha=0.6, label='Base vs Ignore', color='blue')
        axes[0, 0].hist(sim_use, bins=15, alpha=0.6, label='Base vs Use', color='red')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('CoT Similarity Distributions')
        axes[0, 0].legend()
    
    # 2. Scatter plot comparing similarities
    valid_both = df_summary[['avg_cot_sim_base_vs_ignore', 'avg_cot_sim_base_vs_use']].dropna()
    if len(valid_both) > 0:
        axes[0, 1].scatter(valid_both['avg_cot_sim_base_vs_ignore'], 
                          valid_both['avg_cot_sim_base_vs_use'], alpha=0.6)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='y=x')
        axes[0, 1].set_xlabel('Base vs Ignore Similarity')
        axes[0, 1].set_ylabel('Base vs Use Similarity')
        axes[0, 1].set_title('Similarity Comparison')
        axes[0, 1].legend()
        
        # Add correlation
        if len(valid_both) > 2:
            correlation = valid_both.corr().iloc[0, 1]
            axes[0, 1].text(0.05, 0.95, f'Corr: {correlation:.3f}', 
                           transform=axes[0, 1].transAxes, verticalalignment='top')
    
    # 3. Box plots by SUIR detection status
    suir_categories = []
    ignore_similarities = []
    
    for idx, row in df_summary.iterrows():
        suir_rate = row['suir_detection_rate']
        sim_val = row['avg_cot_sim_base_vs_ignore']
        
        if pd.notna(suir_rate) and pd.notna(sim_val):
            if suir_rate > 0:
                category = 'SUIR Detected'
            else:
                category = 'No SUIR'
            
            suir_categories.append(category)
            ignore_similarities.append(sim_val)
    
    if suir_categories:
        sim_df = pd.DataFrame({'SUIR_Status': suir_categories, 'Similarity': ignore_similarities})
        sns.boxplot(data=sim_df, x='SUIR_Status', y='Similarity', ax=axes[1, 0])
        axes[1, 0].set_title('Base-Ignore Similarity by SUIR Status')
        axes[1, 0].set_ylabel('Cosine Similarity')
    
    # 4. Similarity difference analysis
    if len(valid_both) > 0:
        similarity_diff = valid_both['avg_cot_sim_base_vs_use'] - valid_both['avg_cot_sim_base_vs_ignore']
        axes[1, 1].hist(similarity_diff, bins=15, alpha=0.7, color='purple')
        axes[1, 1].axvline(0, color='red', linestyle='--', label='No difference')
        axes[1, 1].set_xlabel('Similarity Difference (Use - Ignore)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Similarity Difference Distribution')
        axes[1, 1].legend()
        
        mean_diff = similarity_diff.mean()
        axes[1, 1].axvline(mean_diff, color='orange', linestyle=':', 
                          label=f'Mean: {mean_diff:.3f}')
        axes[1, 1].legend()
    
    # Adjust layout to prevent title overlap and improve spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved semantic similarity analysis to {save_path}")

def create_llm_judge_analysis(df_detailed, save_path):
    """Create plots showing LLM judge evaluation results."""
    if df_detailed.empty or 'llm_judge_relevance' not in df_detailed.columns:
        print("Skipping LLM judge analysis - no detailed data with judge results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM Judge Evaluation Results', fontsize=TITLE_SIZE + 4, y=0.98)
    
    # 1. Relevance judgments pie chart (make it larger)
    relevance_counts = df_detailed['llm_judge_relevance'].value_counts()
    if len(relevance_counts) > 0:
        colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightgray']
        # Create pie chart with larger radius and better label positioning
        wedges, texts, autotexts = axes[0, 0].pie(relevance_counts.values, labels=relevance_counts.index, 
                                                   colors=colors[:len(relevance_counts)], 
                                                   autopct='%1.1f%%', startangle=90,
                                                   textprops={'fontsize': 11})
        axes[0, 0].set_title('Implicit Factor Relevance Judgments', pad=15)
    
    # 2. Faithfulness change distribution
    faithfulness_counts = df_detailed['llm_judge_faithfulness'].value_counts()
    if len(faithfulness_counts) > 0:
        axes[0, 1].bar(range(len(faithfulness_counts)), faithfulness_counts.values, 
                       color='skyblue', alpha=0.7)
        axes[0, 1].set_xticks(range(len(faithfulness_counts)))
        axes[0, 1].set_xticklabels(faithfulness_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Faithfulness Change Categories')
        
        # Add count labels on bars
        for i, count in enumerate(faithfulness_counts.values):
            axes[0, 1].text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    # 3. Cross-tabulation heatmap
    if 'llm_judge_relevance' in df_detailed.columns and 'llm_judge_faithfulness' in df_detailed.columns:
        cross_tab = pd.crosstab(df_detailed['llm_judge_relevance'], 
                               df_detailed['llm_judge_faithfulness'], margins=False)
        if not cross_tab.empty:
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title('Relevance vs Faithfulness Cross-tabulation')
            axes[1, 0].set_xlabel('Faithfulness Change')
            axes[1, 0].set_ylabel('Relevance Judgment')
    
    # 4. SUIR detection success by judge evaluations
    if 'suir_detected_this_probe' in df_detailed.columns:
        # Group by relevance and calculate SUIR detection rates
        judge_suir = df_detailed.groupby('llm_judge_relevance')['suir_detected_this_probe'].agg(['count', 'sum', 'mean']).reset_index()
        judge_suir['detection_rate'] = judge_suir['mean']
        
        bars = axes[1, 1].bar(judge_suir['llm_judge_relevance'], judge_suir['detection_rate'], 
                             color='lightcoral', alpha=0.7)
        axes[1, 1].set_ylabel('SUIR Detection Rate')
        axes[1, 1].set_xlabel('Judge Relevance Assessment')
        axes[1, 1].set_title('SUIR Detection by Judge Relevance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add rate labels on bars
        for bar, rate in zip(bars, judge_suir['detection_rate']):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.2f}', ha='center', va='bottom')
    
    # Adjust layout to prevent title overlap and improve spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved LLM judge analysis to {save_path}")

def create_hypothesis_effectiveness_analysis(df_detailed, save_path):
    """Create plots showing hypothesis effectiveness analysis."""
    if df_detailed.empty:
        print("Skipping hypothesis effectiveness analysis - no detailed data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hypothesis Effectiveness Analysis', fontsize=TITLE_SIZE + 4, y=0.98)
    
    # 1. Hypothesis length vs effectiveness
    if 'hypothesis' in df_detailed.columns and 'suir_detected_this_probe' in df_detailed.columns:
        df_detailed['hypothesis_length'] = df_detailed['hypothesis'].str.len()
        length_suir = df_detailed.groupby(pd.cut(df_detailed['hypothesis_length'], bins=5))['suir_detected_this_probe'].mean()
        
        axes[0, 0].bar(range(len(length_suir)), length_suir.values, alpha=0.7, color='green')
        axes[0, 0].set_xticks(range(len(length_suir)))
        axes[0, 0].set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' for interval in length_suir.index], 
                                  rotation=45)
        axes[0, 0].set_xlabel('Hypothesis Length (characters)')
        axes[0, 0].set_ylabel('SUIR Detection Rate')
        axes[0, 0].set_title('Hypothesis Length vs SUIR Detection')
    
    # 2. Answer matching patterns
    if all(col in df_detailed.columns for col in ['baseline_answer_matches_use_answer', 
                                                   'baseline_answer_differs_from_ignore_answer']):
        match_use = df_detailed['baseline_answer_matches_use_answer'].sum()
        no_match_use = len(df_detailed) - match_use
        
        diff_ignore = df_detailed['baseline_answer_differs_from_ignore_answer'].sum()
        same_ignore = len(df_detailed) - diff_ignore
        
        categories = ['Match Use', 'No Match Use', 'Diff Ignore', 'Same Ignore']
        counts = [match_use, no_match_use, diff_ignore, same_ignore]
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        
        bars = axes[0, 1].bar(categories, counts, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Answer Matching Patterns')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           str(count), ha='center', va='bottom')
    
    # 3. Similarity scores by SUIR detection
    if all(col in df_detailed.columns for col in ['suir_detected_this_probe', 
                                                   'similarity_base_cot_vs_ignore_cot']):
        suir_detected = df_detailed[df_detailed['suir_detected_this_probe'] == True]
        suir_not_detected = df_detailed[df_detailed['suir_detected_this_probe'] == False]
        
        sim_detected = suir_detected['similarity_base_cot_vs_ignore_cot'].dropna()
        sim_not_detected = suir_not_detected['similarity_base_cot_vs_ignore_cot'].dropna()
        
        if len(sim_detected) > 0 and len(sim_not_detected) > 0:
            axes[1, 0].hist(sim_detected, bins=10, alpha=0.6, label='SUIR Detected', color='red')
            axes[1, 0].hist(sim_not_detected, bins=10, alpha=0.6, label='No SUIR', color='blue')
            axes[1, 0].set_xlabel('Base-Ignore CoT Similarity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('CoT Similarity by SUIR Detection')
            axes[1, 0].legend()
    
    # 4. Top hypothesis patterns (word frequency analysis)
    if 'hypothesis' in df_detailed.columns:
        # Simple word frequency analysis
        all_hypotheses = ' '.join(df_detailed['hypothesis'].dropna().astype(str))
        # Extract common words (longer than 3 characters, not common stop words)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_hypotheses.lower())
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'will', 'been', 
                     'were', 'said', 'each', 'which', 'their', 'time', 'would'}
        words = [word for word in words if word not in stop_words]
        
        word_freq = Counter(words).most_common(10)
        if word_freq:
            words, counts = zip(*word_freq)
            axes[1, 1].barh(range(len(words)), counts, color='purple', alpha=0.7)
            axes[1, 1].set_yticks(range(len(words)))
            axes[1, 1].set_yticklabels(words)
            axes[1, 1].set_xlabel('Frequency')
            axes[1, 1].set_title('Most Common Words in Hypotheses')
    
    # Adjust layout to prevent title overlap and improve spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved hypothesis effectiveness analysis to {save_path}")

def create_question_performance_overview(df_summary, save_path):
    """Create plots showing overall question performance patterns."""
    if df_summary.empty:
        print("Skipping question performance overview - no summary data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Question Performance Overview', fontsize=TITLE_SIZE + 4, y=0.98)
    
    # 1. Correlation matrix of key metrics
    metrics = ['suir_detection_rate', 'answer_flip_rate_when_ignoring', 
              'avg_cot_sim_base_vs_ignore', 'avg_cot_sim_base_vs_use']
    available_metrics = [m for m in metrics if m in df_summary.columns]
    
    if len(available_metrics) > 1:
        corr_matrix = df_summary[available_metrics].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 0])
        axes[0, 0].set_title('Correlation Matrix of Key Metrics')
    
    # 2. Top and bottom performing questions
    if 'suir_detection_rate' in df_summary.columns:
        top_questions = df_summary.nlargest(5, 'suir_detection_rate')
        bottom_questions = df_summary.nsmallest(5, 'suir_detection_rate')
        
        # Show top performers
        if len(top_questions) > 0:
            axes[0, 1].barh(range(len(top_questions)), top_questions['suir_detection_rate'], 
                           color='lightgreen', alpha=0.7)
            axes[0, 1].set_yticks(range(len(top_questions)))
            axes[0, 1].set_yticklabels([f"Q{i[:10]}..." for i in top_questions['question_id']], 
                                      fontsize=10)
            axes[0, 1].set_xlabel('SUIR Detection Rate')
            axes[0, 1].set_title('Top 5 Questions by SUIR Detection')
    
    # 3. Summary statistics table (as text)
    stats_text = []
    for metric in available_metrics:
        values = df_summary[metric].dropna()
        if len(values) > 0:
            stats_text.append(f"{metric}:")
            stats_text.append(f"  Mean: {values.mean():.3f}")
            stats_text.append(f"  Std:  {values.std():.3f}")
            stats_text.append(f"  Min:  {values.min():.3f}")
            stats_text.append(f"  Max:  {values.max():.3f}")
            stats_text.append("")
    
    axes[1, 0].text(0.1, 0.9, '\n'.join(stats_text), transform=axes[1, 0].transAxes, 
                   fontfamily='monospace', fontsize=10, verticalalignment='top')
    axes[1, 0].set_title('Summary Statistics')
    axes[1, 0].axis('off')
    
    # 4. Overall experiment summary
    total_questions = len(df_summary)
    questions_with_suir = (df_summary['suir_detection_rate'] > 0).sum()
    avg_hypotheses = df_summary.get('num_hypotheses', pd.Series([0])).mean()
    
    summary_text = [
        f"Total Questions Analyzed: {total_questions}",
        f"Questions with SUIR Detected: {questions_with_suir}",
        f"SUIR Detection Rate: {questions_with_suir/total_questions*100:.1f}%",
        f"Average Hypotheses per Question: {avg_hypotheses:.1f}",
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    
    axes[1, 1].text(0.1, 0.9, '\n'.join(summary_text), transform=axes[1, 1].transAxes, 
                   fontfamily='monospace', fontsize=12, verticalalignment='top')
    axes[1, 1].set_title('Experiment Summary')
    axes[1, 1].axis('off')
    
    # Adjust layout to prevent title overlap and improve spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved question performance overview to {save_path}")

def create_combined_pdf_report(df_summary, df_detailed):
    """Create a combined PDF report with all visualizations."""
    pdf_path = os.path.join(PLOTS_DIR, "irac_analysis_report.pdf")
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: SUIR Detection Overview
        create_suir_detection_overview(df_summary, "temp_suir.png")
        if os.path.exists("temp_suir.png"):
            img = plt.imread("temp_suir.png")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            os.remove("temp_suir.png")
        
        # Page 2: Answer Flip Analysis
        create_answer_flip_analysis(df_summary, "temp_flip.png")
        if os.path.exists("temp_flip.png"):
            img = plt.imread("temp_flip.png")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            os.remove("temp_flip.png")
        
        # Page 3: Semantic Similarity Analysis
        create_semantic_similarity_analysis(df_summary, "temp_sim.png")
        if os.path.exists("temp_sim.png"):
            img = plt.imread("temp_sim.png")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            os.remove("temp_sim.png")
        
        # Page 4: LLM Judge Analysis (if available)
        create_llm_judge_analysis(df_detailed, "temp_judge.png")
        if os.path.exists("temp_judge.png"):
            img = plt.imread("temp_judge.png")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            os.remove("temp_judge.png")
        
        # Page 5: Hypothesis Effectiveness
        create_hypothesis_effectiveness_analysis(df_detailed, "temp_hypo.png")
        if os.path.exists("temp_hypo.png"):
            img = plt.imread("temp_hypo.png")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            os.remove("temp_hypo.png")
        
        # Page 6: Question Performance Overview
        create_question_performance_overview(df_summary, "temp_perf.png")
        if os.path.exists("temp_perf.png"):
            img = plt.imread("temp_perf.png")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            os.remove("temp_perf.png")
    
    print(f"Combined PDF report saved to {pdf_path}")

def main():
    """Main function to generate all plots and reports."""
    print("IRAC Analysis Plotting Script")
    print("=" * 40)
    
    # Setup
    setup_plotting_style()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created plots directory: {PLOTS_DIR}")
    
    # Load data
    df_summary, df_detailed = load_data()
    
    if df_summary.empty and df_detailed.empty:
        print("No data found to plot. Please run analyze_results.py first.")
        return
    
    print(f"\nGenerating plots in {PLOTS_DIR}...")
    
    # Generate individual plots
    plots_created = []
    
    if not df_summary.empty:
        # SUIR Detection Overview
        suir_path = os.path.join(PLOTS_DIR, "01_suir_detection_overview.png")
        create_suir_detection_overview(df_summary, suir_path)
        plots_created.append(suir_path)
        
        # Answer Flip Analysis
        flip_path = os.path.join(PLOTS_DIR, "02_answer_flip_analysis.png")
        create_answer_flip_analysis(df_summary, flip_path)
        plots_created.append(flip_path)
        
        # Semantic Similarity Analysis
        sim_path = os.path.join(PLOTS_DIR, "03_semantic_similarity_analysis.png")
        create_semantic_similarity_analysis(df_summary, sim_path)
        plots_created.append(sim_path)
        
        # Question Performance Overview
        perf_path = os.path.join(PLOTS_DIR, "06_question_performance_overview.png")
        create_question_performance_overview(df_summary, perf_path)
        plots_created.append(perf_path)
    
    if not df_detailed.empty:
        # LLM Judge Analysis
        judge_path = os.path.join(PLOTS_DIR, "04_llm_judge_analysis.png")
        create_llm_judge_analysis(df_detailed, judge_path)
        plots_created.append(judge_path)
        
        # Hypothesis Effectiveness Analysis
        hypo_path = os.path.join(PLOTS_DIR, "05_hypothesis_effectiveness_analysis.png")
        create_hypothesis_effectiveness_analysis(df_detailed, hypo_path)
        plots_created.append(hypo_path)
    
    # Create combined PDF report
    if plots_created:
        create_combined_pdf_report(df_summary, df_detailed)
    
    print(f"\nPlotting complete! Generated {len(plots_created)} individual plots.")
    print(f"All files saved to: {PLOTS_DIR}")
    
    # Print summary of what was created
    print("\nFiles created:")
    for plot_path in plots_created:
        print(f"  - {os.path.basename(plot_path)}")
    print(f"  - irac_analysis_report.pdf")

if __name__ == "__main__":
    main()
