import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# --- Configuration ---
RESULTS_DIR = "/home/ec2-user/code/personal/logic_cot/experiment_results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "irac_analysis_summary.csv")
DETAILED_CSV = os.path.join(RESULTS_DIR, "irac_detailed_probe_analysis.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "irac_analysis_summary.json")
REPORT_OUTPUT = os.path.join(RESULTS_DIR, "IRAC_ANALYSIS_REPORT.md")

def load_data():
    """Load all analysis results."""
    print("Loading analysis data...")
    
    df_summary = pd.DataFrame()
    df_detailed = pd.DataFrame()
    json_data = None
    
    if os.path.exists(SUMMARY_CSV):
        df_summary = pd.read_csv(SUMMARY_CSV)
        print(f"✓ Loaded summary CSV: {len(df_summary)} questions")
    else:
        print(f"✗ Warning: Summary CSV not found at {SUMMARY_CSV}")
    
    if os.path.exists(DETAILED_CSV):
        df_detailed = pd.read_csv(DETAILED_CSV)
        print(f"✓ Loaded detailed CSV: {len(df_detailed)} probes")
    else:
        print(f"✗ Warning: Detailed CSV not found at {DETAILED_CSV}")
    
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON, 'r') as f:
            json_data = json.load(f)
        print(f"✓ Loaded JSON data: {len(json_data)} question analyses")
    else:
        print(f"✗ Warning: JSON data not found at {SUMMARY_JSON}")
    
    return df_summary, df_detailed, json_data

def format_percentage(value):
    """Format value as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"

def format_number(value, decimals=3):
    """Format number with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def generate_executive_summary(df_summary, df_detailed):
    """Generate executive summary section."""
    md = ["# IRAC Analysis Report: Structural Unfaithfulness and Implicit Reliance in LLM Reasoning\n"]
    md.append(f"**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
    md.append("---\n")
    
    md.append("## Executive Summary\n")
    
    if df_summary.empty:
        md.append("*No data available for analysis.*\n")
        return "\n".join(md)
    
    total_questions = len(df_summary)
    questions_with_suir = (df_summary['suir_detection_rate'] > 0).sum()
    suir_percentage = (questions_with_suir / total_questions * 100) if total_questions > 0 else 0
    
    avg_suir_rate = df_summary['suir_detection_rate'].mean()
    avg_flip_rate = df_summary['answer_flip_rate_when_ignoring'].mean()
    avg_hypotheses = df_summary['num_hypotheses'].mean()
    
    md.append(f"This report presents findings from the Implicit Reliance Articulation & Contrast (IRAC) method, ")
    md.append(f"designed to detect Structural Unfaithfulness and Implicit Reliance (SUIR) in Large Language Model ")
    md.append(f"reasoning processes.\n\n")
    
    md.append("### Key Findings\n\n")
    md.append(f"- **Total Questions Analyzed:** {total_questions}\n")
    md.append(f"- **Questions with SUIR Detected:** {questions_with_suir} ({suir_percentage:.1f}%)\n")
    md.append(f"- **Average SUIR Detection Rate:** {format_percentage(avg_suir_rate)}\n")
    md.append(f"- **Average Answer Flip Rate:** {format_percentage(avg_flip_rate)}\n")
    md.append(f"- **Average Hypotheses Tested per Question:** {avg_hypotheses:.1f}\n")
    
    if not df_detailed.empty and 'llm_judge_relevance' in df_detailed.columns:
        total_probes = len(df_detailed)
        relevance_yes = (df_detailed['llm_judge_relevance'] == 'Yes').sum()
        md.append(f"- **Total Probes Conducted:** {total_probes}\n")
        md.append(f"- **LLM Judge Relevance Confirmations:** {relevance_yes} ({relevance_yes/total_probes*100:.1f}%)\n")
    
    md.append("\n")
    return "\n".join(md)

def generate_methodology_section():
    """Generate methodology section."""
    md = ["## Methodology Overview\n"]
    md.append("### The IRAC Method\n\n")
    md.append("The Implicit Reliance Articulation & Contrast (IRAC) method involves three stages:\n\n")
    md.append("**Stage 1: Baseline CoT Generation**\n")
    md.append("- Obtain the LLM's standard Chain-of-Thought (CoT) and answer for each question\n")
    md.append("- Establish baseline reasoning patterns\n\n")
    
    md.append("**Stage 2: Hypothesis Generation & Probing**\n")
    md.append("- Generate hypotheses about potential implicit factors (structural cues, unstated assumptions, shortcuts)\n")
    md.append("- Probe the model with two conditions for each hypothesis:\n")
    md.append("  - **Articulate & Use:** Explicitly acknowledge and use the implicit factor\n")
    md.append("  - **Articulate & Ignore:** Explicitly acknowledge but avoid relying on the factor\n\n")
    
    md.append("**Stage 3: Contrastive Analysis**\n")
    md.append("- Compare baseline answers and CoTs with probed responses\n")
    md.append("- Calculate SUIR scores based on:\n")
    md.append("  - Answer consistency (baseline == use, baseline ≠ ignore)\n")
    md.append("  - Semantic similarity between CoTs\n")
    md.append("  - LLM judge evaluations of relevance and faithfulness\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_suir_detection_section(df_summary):
    """Generate SUIR detection analysis section."""
    md = ["## SUIR Detection Analysis\n\n"]
    
    if df_summary.empty:
        md.append("*No summary data available.*\n\n")
        return "\n".join(md)
    
    # Add visualization first
    plot_path = os.path.join(PLOTS_DIR, "01_suir_detection_overview.png")
    if os.path.exists(plot_path):
        md.append("### Visualization\n\n")
        md.append("![SUIR Detection Overview](plots/01_suir_detection_overview.png)\n\n")
        md.append("*Figure 1: SUIR Detection Overview - Distribution, box plot, pie chart, and correlation analysis*\n\n")
    
    md.append("### Detection Rate Distribution\n\n")
    
    # Statistics
    suir_rates = df_summary['suir_detection_rate'].dropna()
    if not suir_rates.empty:
        md.append("| Statistic | Value |\n")
        md.append("| --------- | ----- |\n")
        md.append(f"| Mean | {format_percentage(suir_rates.mean())} |\n")
        md.append(f"| Median | {format_percentage(suir_rates.median())} |\n")
        md.append(f"| Std Dev | {format_percentage(suir_rates.std())} |\n")
        md.append(f"| Min | {format_percentage(suir_rates.min())} |\n")
        md.append(f"| Max | {format_percentage(suir_rates.max())} |\n")
        md.append("\n")
    
    # Categories
    high_suir = (df_summary['suir_detection_rate'] > 0.5).sum()
    moderate_suir = ((df_summary['suir_detection_rate'] > 0.2) & 
                     (df_summary['suir_detection_rate'] <= 0.5)).sum()
    low_suir = ((df_summary['suir_detection_rate'] > 0) & 
                (df_summary['suir_detection_rate'] <= 0.2)).sum()
    no_suir = (df_summary['suir_detection_rate'] == 0).sum()
    
    md.append("### SUIR Detection Categories\n\n")
    md.append("| Category | Count | Percentage |\n")
    md.append("|----------|-------|------------|\n")
    md.append(f"| High SUIR (>50%) | {high_suir} | {high_suir/len(df_summary)*100:.1f}% |\n")
    md.append(f"| Moderate SUIR (20-50%) | {moderate_suir} | {moderate_suir/len(df_summary)*100:.1f}% |\n")
    md.append(f"| Low SUIR (1-20%) | {low_suir} | {low_suir/len(df_summary)*100:.1f}% |\n")
    md.append(f"| No SUIR (0%) | {no_suir} | {no_suir/len(df_summary)*100:.1f}% |\n")
    md.append("\n")
    
    # Top questions
    if 'question_id' in df_summary.columns:
        top_questions = df_summary.nlargest(5, 'suir_detection_rate')
        if not top_questions.empty:
            md.append("### Top 5 Questions by SUIR Detection Rate\n\n")
            md.append("| Question ID | SUIR Rate | # Hypotheses |\n")
            md.append("|-------------|-----------|-------------|\n")
            for _, row in top_questions.iterrows():
                qid = str(row['question_id'])[:30] + "..." if len(str(row['question_id'])) > 30 else str(row['question_id'])
                md.append(f"| {qid} | {format_percentage(row['suir_detection_rate'])} | {row['num_hypotheses']:.0f} |\n")
            md.append("\n")
    
    # Visual reference
    md.append("### Visualizations\n\n")
    md.append("See **Figure 1: SUIR Detection Overview** in `plots/01_suir_detection_overview.png` for:\n")
    md.append("- Distribution histogram of SUIR detection rates\n")
    md.append("- Box plot showing quartiles and outliers\n")
    md.append("- Pie chart of questions with/without SUIR detection\n")
    md.append("- Correlation between number of hypotheses and detection rate\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_answer_flip_section(df_summary):
    """Generate answer flip analysis section."""
    md = ["## Answer Flip Analysis\n\n"]
    
    if df_summary.empty:
        md.append("*No summary data available.*\n\n")
        return "\n".join(md)
    
    # Add visualization first
    plot_path = os.path.join(PLOTS_DIR, "02_answer_flip_analysis.png")
    if os.path.exists(plot_path):
        md.append("### Visualization\n\n")
        md.append("![Answer Flip Analysis](plots/02_answer_flip_analysis.png)\n\n")
        md.append("*Figure 2: Answer Flip Analysis - Distribution, correlation with SUIR, categories, and box plots*\n\n")
    
    md.append("### Overview\n\n")
    md.append("Answer flip rate measures how often the model's answer changes when explicitly asked to ")
    md.append("ignore an implicit factor, indicating reliance on unstated assumptions.\n\n")
    
    # Statistics
    flip_rates = df_summary['answer_flip_rate_when_ignoring'].dropna()
    if not flip_rates.empty:
        md.append("| Statistic | Value |\n")
        md.append("| --------- | ----- |\n")
        md.append(f"| Mean Flip Rate | {format_percentage(flip_rates.mean())} |\n")
        md.append(f"| Median Flip Rate | {format_percentage(flip_rates.median())} |\n")
        md.append(f"| Std Dev | {format_percentage(flip_rates.std())} |\n")
        md.append(f"| Max Flip Rate | {format_percentage(flip_rates.max())} |\n")
        md.append("\n")
    
    # Correlation with SUIR
    if len(df_summary) > 2:
        valid_data = df_summary[['suir_detection_rate', 'answer_flip_rate_when_ignoring']].dropna()
        if len(valid_data) > 2:
            correlation = valid_data.corr().iloc[0, 1]
            md.append("### Correlation with SUIR Detection\n\n")
            md.append(f"**Pearson Correlation Coefficient:** {format_number(correlation, 3)}\n\n")
            
            if correlation > 0.5:
                md.append("Strong positive correlation suggests answer flips are a reliable indicator of SUIR.\n\n")
            elif correlation > 0.3:
                md.append("Moderate positive correlation indicates answer flips partially predict SUIR.\n\n")
            else:
                md.append("Weak correlation suggests answer flips alone may not fully capture SUIR.\n\n")
    
    # Categories
    high_flip = (df_summary['answer_flip_rate_when_ignoring'] > 0.5).sum()
    some_flip = ((df_summary['answer_flip_rate_when_ignoring'] > 0) & 
                 (df_summary['answer_flip_rate_when_ignoring'] <= 0.5)).sum()
    no_flip = (df_summary['answer_flip_rate_when_ignoring'] == 0).sum()
    
    md.append("### Flip Rate Categories\n\n")
    md.append("| Category | Count | Percentage |\n")
    md.append("|----------|-------|------------|\n")
    md.append(f"| High Flip (>50%) | {high_flip} | {high_flip/len(df_summary)*100:.1f}% |\n")
    md.append(f"| Some Flip (1-50%) | {some_flip} | {some_flip/len(df_summary)*100:.1f}% |\n")
    md.append(f"| No Flip (0%) | {no_flip} | {no_flip/len(df_summary)*100:.1f}% |\n")
    md.append("\n")
    
    md.append("### Visualizations\n\n")
    md.append("See **Figure 2: Answer Flip Analysis** in `plots/02_answer_flip_analysis.png`\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_similarity_section(df_summary):
    """Generate semantic similarity analysis section."""
    md = ["## Semantic Similarity Analysis\n\n"]
    
    if df_summary.empty:
        md.append("*No summary data available.*\n\n")
        return "\n".join(md)
    
    # Add visualization first
    plot_path = os.path.join(PLOTS_DIR, "03_semantic_similarity_analysis.png")
    if os.path.exists(plot_path):
        md.append("### Visualization\n\n")
        md.append("![Semantic Similarity Analysis](plots/03_semantic_similarity_analysis.png)\n\n")
        md.append("*Figure 3: Semantic Similarity Analysis - Distributions, comparisons, SUIR status, and difference analysis*\n\n")
    
    md.append("### CoT Semantic Divergence\n\n")
    md.append("Semantic similarity measures how much the Chain-of-Thought changes when implicit factors ")
    md.append("are explicitly considered. Lower similarity indicates greater divergence.\n\n")
    
    # Base vs Ignore statistics
    sim_ignore = df_summary['avg_cot_sim_base_vs_ignore'].dropna()
    sim_use = df_summary['avg_cot_sim_base_vs_use'].dropna()
    
    if not sim_ignore.empty or not sim_use.empty:
        md.append("| Comparison | Mean | Median | Std Dev |\n")
        md.append("| ---------- | ---- | ------ | ------- |\n")
        
        if not sim_ignore.empty:
            md.append(f"| Baseline vs Ignore | {format_number(sim_ignore.mean())} | ")
            md.append(f"{format_number(sim_ignore.median())} | {format_number(sim_ignore.std())} |\n")
        
        if not sim_use.empty:
            md.append(f"| Baseline vs Use | {format_number(sim_use.mean())} | ")
            md.append(f"{format_number(sim_use.median())} | {format_number(sim_use.std())} |\n")
        
        md.append("\n")
    
    # Interpretation
    md.append("### Interpretation\n\n")
    
    if not sim_ignore.empty:
        mean_ignore = sim_ignore.mean()
        if mean_ignore > 0.8:
            md.append(f"**High similarity ({format_number(mean_ignore)})** between baseline and ignore CoTs ")
            md.append("suggests the model maintains similar reasoning even when asked to avoid implicit factors.\n\n")
        elif mean_ignore > 0.6:
            md.append(f"**Moderate similarity ({format_number(mean_ignore)})** indicates some CoT changes ")
            md.append("when implicit factors are explicitly ignored.\n\n")
        else:
            md.append(f"**Low similarity ({format_number(mean_ignore)})** shows significant CoT divergence ")
            md.append("when avoiding implicit factors, suggesting strong implicit reliance.\n\n")
    
    # Difference analysis
    valid_both = df_summary[['avg_cot_sim_base_vs_ignore', 'avg_cot_sim_base_vs_use']].dropna()
    if len(valid_both) > 0:
        similarity_diff = valid_both['avg_cot_sim_base_vs_use'] - valid_both['avg_cot_sim_base_vs_ignore']
        mean_diff = similarity_diff.mean()
        
        md.append("### Similarity Difference (Use - Ignore)\n\n")
        md.append(f"**Mean Difference:** {format_number(mean_diff)}\n\n")
        
        if abs(mean_diff) < 0.05:
            md.append("Minimal difference suggests balanced reasoning changes in both conditions.\n\n")
        elif mean_diff > 0:
            md.append("Positive difference indicates baseline CoT is more similar to 'use' condition, ")
            md.append("supporting implicit reliance hypothesis.\n\n")
        else:
            md.append("Negative difference suggests unexpected pattern requiring further investigation.\n\n")
    
    md.append("### Visualizations\n\n")
    md.append("See **Figure 3: Semantic Similarity Analysis** in `plots/03_semantic_similarity_analysis.png`\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_llm_judge_section(df_detailed):
    """Generate LLM judge analysis section."""
    md = ["## LLM Judge Evaluation Results\n\n"]
    
    if df_detailed.empty or 'llm_judge_relevance' not in df_detailed.columns:
        md.append("*LLM judge evaluations not available in the dataset.*\n\n")
        return "\n".join(md)
    
    # Add visualization first
    plot_path = os.path.join(PLOTS_DIR, "04_llm_judge_analysis.png")
    if os.path.exists(plot_path):
        md.append("### Visualization\n\n")
        md.append("![LLM Judge Analysis](plots/04_llm_judge_analysis.png)\n\n")
        md.append("*Figure 4: LLM Judge Evaluation Results - Relevance pie chart, faithfulness distribution, cross-tabulation, and SUIR detection by relevance*\n\n")
    
    md.append("### Relevance Assessments\n\n")
    md.append("The LLM judge evaluated whether each identified implicit factor was plausibly ")
    md.append("relevant to the model's original answer.\n\n")
    
    # Relevance counts
    relevance_counts = df_detailed['llm_judge_relevance'].value_counts()
    total = len(df_detailed)
    
    md.append("| Assessment | Count | Percentage |\n")
    md.append("|------------|-------|------------|\n")
    for assessment, count in relevance_counts.items():
        md.append(f"| {assessment} | {count} | {count/total*100:.1f}% |\n")
    md.append("\n")
    
    # Faithfulness assessments
    if 'llm_judge_faithfulness' in df_detailed.columns:
        md.append("### Faithfulness Change Assessments\n\n")
        md.append("The LLM judge evaluated how the original CoT's perceived faithfulness changed ")
        md.append("after seeing the model's reasoning when asked to ignore implicit factors.\n\n")
        
        faithfulness_counts = df_detailed['llm_judge_faithfulness'].value_counts()
        
        md.append("| Assessment | Count | Percentage |\n")
        md.append("|------------|-------|------------|\n")
        for assessment, count in faithfulness_counts.items():
            md.append(f"| {assessment} | {count} | {count/total*100:.1f}% |\n")
        md.append("\n")
    
    # Cross analysis
    if 'suir_detected_this_probe' in df_detailed.columns:
        md.append("### SUIR Detection by Judge Relevance\n\n")
        
        judge_suir = df_detailed.groupby('llm_judge_relevance')['suir_detected_this_probe'].agg(['count', 'sum', 'mean'])
        
        md.append("| Judge Assessment | Total Probes | SUIR Detected | Detection Rate |\n")
        md.append("|------------------|--------------|---------------|----------------|\n")
        for assessment, row in judge_suir.iterrows():
            md.append(f"| {assessment} | {int(row['count'])} | {int(row['sum'])} | ")
            md.append(f"{format_percentage(row['mean'])} |\n")
        md.append("\n")
    
    md.append("### Visualizations\n\n")
    md.append("See **Figure 4: LLM Judge Analysis** in `plots/04_llm_judge_analysis.png`\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_hypothesis_section(df_detailed):
    """Generate hypothesis effectiveness section."""
    md = ["## Hypothesis Effectiveness Analysis\n\n"]
    
    if df_detailed.empty:
        md.append("*No detailed probe data available.*\n\n")
        return "\n".join(md)
    
    # Add visualization first
    plot_path = os.path.join(PLOTS_DIR, "05_hypothesis_effectiveness_analysis.png")
    if os.path.exists(plot_path):
        md.append("### Visualization\n\n")
        md.append("![Hypothesis Effectiveness Analysis](plots/05_hypothesis_effectiveness_analysis.png)\n\n")
        md.append("*Figure 5: Hypothesis Effectiveness - Length vs detection, answer matching patterns, similarity by SUIR status, and word frequency*\n\n")
    
    md.append("### Answer Matching Patterns\n\n")
    
    if 'baseline_answer_matches_use_answer' in df_detailed.columns:
        match_use = df_detailed['baseline_answer_matches_use_answer'].sum()
        total = len(df_detailed)
        
        md.append(f"- **Baseline matches 'Use' answer:** {match_use} / {total} ({match_use/total*100:.1f}%)\n")
    
    if 'baseline_answer_differs_from_ignore_answer' in df_detailed.columns:
        diff_ignore = df_detailed['baseline_answer_differs_from_ignore_answer'].sum()
        
        md.append(f"- **Baseline differs from 'Ignore' answer:** {diff_ignore} / {total} ({diff_ignore/total*100:.1f}%)\n")
    
    md.append("\n")
    
    # SUIR detection rate
    if 'suir_detected_this_probe' in df_detailed.columns:
        suir_detected = df_detailed['suir_detected_this_probe'].sum()
        md.append(f"**Overall SUIR Detection Rate (probe-level):** {suir_detected} / {total} ({suir_detected/total*100:.1f}%)\n\n")
    
    # Hypothesis characteristics
    if 'hypothesis' in df_detailed.columns:
        md.append("### Hypothesis Characteristics\n\n")
        
        df_detailed_copy = df_detailed.copy()
        df_detailed_copy['hypothesis_length'] = df_detailed_copy['hypothesis'].str.len()
        
        md.append(f"- **Average hypothesis length:** {df_detailed_copy['hypothesis_length'].mean():.0f} characters\n")
        md.append(f"- **Median hypothesis length:** {df_detailed_copy['hypothesis_length'].median():.0f} characters\n")
        md.append(f"- **Total unique hypotheses:** {df_detailed['hypothesis'].nunique()}\n\n")
    
    md.append("### Visualizations\n\n")
    md.append("See **Figure 5: Hypothesis Effectiveness Analysis** in `plots/05_hypothesis_effectiveness_analysis.png`\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_conclusions_section(df_summary, df_detailed):
    """Generate conclusions and implications section."""
    md = ["## Conclusions and Implications\n"]
    
    if df_summary.empty:
        md.append("*Insufficient data for conclusions.*\n\n")
        return "\n".join(md)
    
    md.append("### Key Takeaways\n\n")
    
    # Calculate success criteria
    suir_detection_percentage = (df_summary['suir_detection_rate'] > 0).sum() / len(df_summary) * 100
    
    md.append(f"1. **SUIR Detection Success:** SUIR was detected in {suir_detection_percentage:.1f}% of analyzed questions, ")
    
    if suir_detection_percentage >= 15:
        md.append("meeting the success criterion of 10-15% detection rate specified in the research proposal.\n\n")
    elif suir_detection_percentage >= 10:
        md.append("approaching the success criterion of 10-15% detection rate.\n\n")
    else:
        md.append("below the target 10-15% detection rate, suggesting potential refinements needed.\n\n")
    
    # Answer flip significance
    avg_flip_rate = df_summary['answer_flip_rate_when_ignoring'].mean()
    md.append(f"2. **Answer Changes:** On average, {format_percentage(avg_flip_rate)} of answers changed ")
    md.append("when models were asked to ignore implicit factors, demonstrating the IRAC method's ")
    md.append("effectiveness in revealing hidden dependencies.\n\n")
    
    # Semantic similarity insights
    sim_ignore = df_summary['avg_cot_sim_base_vs_ignore'].dropna()
    if not sim_ignore.empty:
        mean_sim = sim_ignore.mean()
        md.append(f"3. **CoT Divergence:** Average similarity of {format_number(mean_sim)} between baseline ")
        md.append("and 'ignore' conditions indicates ")
        if mean_sim > 0.8:
            md.append("relatively minor CoT changes, suggesting implicit factors may not substantially ")
            md.append("alter reasoning paths even when their influence on answers is evident.\n\n")
        else:
            md.append("substantial CoT changes, confirming that implicit factors significantly shape ")
            md.append("the reasoning process.\n\n")
    
    # LLM judge validation
    if not df_detailed.empty and 'llm_judge_relevance' in df_detailed.columns:
        relevance_yes = (df_detailed['llm_judge_relevance'] == 'Yes').sum()
        total_probes = len(df_detailed)
        md.append(f"4. **Expert Validation:** LLM judge confirmed relevance in {relevance_yes/total_probes*100:.1f}% ")
        md.append("of cases, providing external validation of the identified implicit factors.\n\n")
    
    md.append("### Implications for LLM Trustworthiness\n\n")
    md.append("These findings have important implications:\n\n")
    md.append("- **Transparency Limitations:** Even when CoTs appear logically sound, they may not fully ")
    md.append("capture the model's decision-making process.\n\n")
    md.append("- **High-Stakes Applications:** For critical applications, additional verification beyond ")
    md.append("CoT analysis is warranted.\n\n")
    md.append("- **Prompt Engineering:** Understanding implicit reliance patterns can inform better ")
    md.append("prompt design to elicit more faithful reasoning.\n\n")
    md.append("- **Model Development:** IRAC can serve as a diagnostic tool during model training ")
    md.append("to reduce implicit biases.\n\n")
    
    md.append("### Future Directions\n\n")
    md.append("1. Expand analysis to additional datasets (StrategyQA, TruthfulQA, BBH)\n")
    md.append("2. Compare SUIR patterns across different model families and sizes\n")
    md.append("3. Develop automated hypothesis generation improvements\n")
    md.append("4. Investigate intervention strategies to reduce implicit reliance\n")
    md.append("5. Create fine-tuning approaches that enhance CoT faithfulness\n\n")
    
    md.append("---\n\n")
    return "\n".join(md)

def generate_appendix_section(df_summary, df_detailed):
    """Generate appendix with additional details."""
    md = ["## Appendix: Additional Details\n"]
    
    md.append("### Data Files\n\n")
    md.append("All analysis results are available in the following files:\n\n")
    md.append(f"- **Summary CSV:** `{os.path.basename(SUMMARY_CSV)}`\n")
    md.append(f"- **Detailed CSV:** `{os.path.basename(DETAILED_CSV)}`\n")
    md.append(f"- **JSON Analysis:** `{os.path.basename(SUMMARY_JSON)}`\n")
    md.append(f"- **Plots Directory:** `plots/`\n\n")
    
    md.append("### Available Visualizations\n\n")
    plot_files = [
        ("01_suir_detection_overview.png", "SUIR Detection Overview"),
        ("02_answer_flip_analysis.png", "Answer Flip Analysis"),
        ("03_semantic_similarity_analysis.png", "Semantic Similarity Analysis"),
        ("04_llm_judge_analysis.png", "LLM Judge Evaluation Results"),
        ("05_hypothesis_effectiveness_analysis.png", "Hypothesis Effectiveness"),
        ("06_question_performance_overview.png", "Question Performance Overview"),
        ("irac_analysis_report.pdf", "Combined PDF Report")
    ]
    
    for filename, description in plot_files:
        plot_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(plot_path):
            md.append(f"- ✓ **{description}:** `plots/{filename}`\n")
        else:
            md.append(f"- ✗ *{description}:* `plots/{filename}` (not found)\n")
    
    md.append("\n")
    
    if not df_summary.empty:
        md.append("### Summary Statistics Table\n\n")
        md.append("| Metric | Mean | Median | Std Dev | Min | Max |\n")
        md.append("|--------|------|--------|---------|-----|-----|\n")
        
        metrics = [
            ('suir_detection_rate', 'SUIR Detection Rate'),
            ('answer_flip_rate_when_ignoring', 'Answer Flip Rate'),
            ('avg_cot_sim_base_vs_ignore', 'CoT Sim (Base vs Ignore)'),
            ('avg_cot_sim_base_vs_use', 'CoT Sim (Base vs Use)'),
            ('num_hypotheses', 'Number of Hypotheses')
        ]
        
        for metric_col, metric_name in metrics:
            if metric_col in df_summary.columns:
                values = df_summary[metric_col].dropna()
                if len(values) > 0:
                    md.append(f"| {metric_name} | {format_number(values.mean())} | ")
                    md.append(f"{format_number(values.median())} | {format_number(values.std())} | ")
                    md.append(f"{format_number(values.min())} | {format_number(values.max())} |\n")
        
        md.append("\n")
    
    md.append("---\n\n")
    md.append("*End of Report*\n")
    return "\n".join(md)

def generate_full_report(df_summary, df_detailed, json_data):
    """Generate the complete markdown report."""
    report_sections = []
    
    # Generate all sections
    report_sections.append(generate_executive_summary(df_summary, df_detailed))
    report_sections.append(generate_methodology_section())
    report_sections.append(generate_suir_detection_section(df_summary))
    report_sections.append(generate_answer_flip_section(df_summary))
    report_sections.append(generate_similarity_section(df_summary))
    report_sections.append(generate_llm_judge_section(df_detailed))
    report_sections.append(generate_hypothesis_section(df_detailed))
    report_sections.append(generate_conclusions_section(df_summary, df_detailed))
    report_sections.append(generate_appendix_section(df_summary, df_detailed))
    
    return "\n".join(report_sections)

def main():
    """Main function to generate the report."""
    print("=" * 60)
    print("IRAC Analysis Report Generator")
    print("=" * 60)
    print()
    
    # Load data
    df_summary, df_detailed, json_data = load_data()
    
    if df_summary.empty and df_detailed.empty:
        print("\n✗ Error: No data available to generate report.")
        print("  Please run analyze_results.py first to generate analysis data.")
        return
    
    print("\n" + "=" * 60)
    print("Generating markdown report...")
    print("=" * 60)
    
    # Generate report
    report_content = generate_full_report(df_summary, df_detailed, json_data)
    
    # Save report
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✓ Report successfully generated!")
    print(f"  Location: {REPORT_OUTPUT}")
    print(f"  Size: {len(report_content)} characters")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Report Contents:")
    print("=" * 60)
    print("✓ Executive Summary")
    print("✓ Methodology Overview")
    print("✓ SUIR Detection Analysis")
    print("✓ Answer Flip Analysis")
    print("✓ Semantic Similarity Analysis")
    
    if not df_detailed.empty and 'llm_judge_relevance' in df_detailed.columns:
        print("✓ LLM Judge Evaluation Results")
    else:
        print("○ LLM Judge Evaluation Results (data not available)")
    
    print("✓ Hypothesis Effectiveness Analysis")
    print("✓ Conclusions and Implications")
    print("✓ Appendix: Additional Details")
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
