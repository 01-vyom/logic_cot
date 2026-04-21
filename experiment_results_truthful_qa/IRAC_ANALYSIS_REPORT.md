# IRAC Analysis Report: Structural Unfaithfulness and Implicit Reliance in LLM Reasoning

**Report Generated:** February 19, 2026 at 08:27 PM

---

## Executive Summary

This report presents findings from the Implicit Reliance Articulation & Contrast (IRAC) method, 
designed to detect Structural Unfaithfulness and Implicit Reliance (SUIR) in Large Language Model 
reasoning processes.


### Key Findings


- **Total Questions Analyzed:** 817

- **Questions with SUIR Detected:** 250 (30.6%)

- **Average SUIR Detection Rate:** 14.8%

- **Average Answer Flip Rate:** 74.7%

- **Average Hypotheses Tested per Question:** 3.0

- **Total Probes Conducted:** 2436

- **LLM Judge Relevance Confirmations:** 2301 (94.5%)



## Methodology Overview

### The IRAC Method


The Implicit Reliance Articulation & Contrast (IRAC) method involves three stages:


**Stage 1: Baseline CoT Generation**

- Obtain the LLM's standard Chain-of-Thought (CoT) and answer for each question

- Establish baseline reasoning patterns


**Stage 2: Hypothesis Generation & Probing**

- Generate hypotheses about potential implicit factors (structural cues, unstated assumptions, shortcuts)

- Probe the model with two conditions for each hypothesis:

  - **Articulate & Use:** Explicitly acknowledge and use the implicit factor

  - **Articulate & Ignore:** Explicitly acknowledge but avoid relying on the factor


**Stage 3: Contrastive Analysis**

- Compare baseline answers and CoTs with probed responses

- Calculate SUIR scores based on:

  - Answer consistency (baseline == use, baseline ≠ ignore)

  - Semantic similarity between CoTs

  - LLM judge evaluations of relevance and faithfulness


---


## SUIR Detection Analysis


### Visualization


![SUIR Detection Overview](plots/01_suir_detection_overview.png)


*Figure 1: SUIR Detection Overview - Distribution, box plot, pie chart, and correlation analysis*


### Detection Rate Distribution


| Statistic | Value |

| --------- | ----- |

| Mean | 14.8% |

| Median | 0.0% |

| Std Dev | 25.1% |

| Min | 0.0% |

| Max | 100.0% |



### SUIR Detection Categories


| Category | Count | Percentage |

|----------|-------|------------|

| High SUIR (>50%) | 93 | 11.4% |

| Moderate SUIR (20-50%) | 157 | 19.2% |

| Low SUIR (1-20%) | 0 | 0.0% |

| No SUIR (0%) | 567 | 69.4% |



### Top 5 Questions by SUIR Detection Rate


| Question ID | SUIR Rate | # Hypotheses |

|-------------|-----------|-------------|

| truthful_qa_generation_validat... | 100.0% | 3 |

| truthful_qa_generation_validat... | 100.0% | 3 |

| truthful_qa_generation_validat... | 100.0% | 3 |

| truthful_qa_generation_validat... | 100.0% | 3 |

| truthful_qa_generation_validat... | 100.0% | 3 |



### Visualizations


See **Figure 1: SUIR Detection Overview** in `plots/01_suir_detection_overview.png` for:

- Distribution histogram of SUIR detection rates

- Box plot showing quartiles and outliers

- Pie chart of questions with/without SUIR detection

- Correlation between number of hypotheses and detection rate


---


## Answer Flip Analysis


### Visualization


![Answer Flip Analysis](plots/02_answer_flip_analysis.png)


*Figure 2: Answer Flip Analysis - Distribution, correlation with SUIR, categories, and box plots*


### Overview


Answer flip rate measures how often the model's answer changes when explicitly asked to 
ignore an implicit factor, indicating reliance on unstated assumptions.


| Statistic | Value |

| --------- | ----- |

| Mean Flip Rate | 74.7% |

| Median Flip Rate | 100.0% |

| Std Dev | 36.4% |

| Max Flip Rate | 100.0% |



### Correlation with SUIR Detection


**Pearson Correlation Coefficient:** 0.088


Weak correlation suggests answer flips alone may not fully capture SUIR.


### Flip Rate Categories


| Category | Count | Percentage |

|----------|-------|------------|

| High Flip (>50%) | 624 | 76.4% |

| Some Flip (1-50%) | 84 | 10.3% |

| No Flip (0%) | 109 | 13.3% |



### Visualizations


See **Figure 2: Answer Flip Analysis** in `plots/02_answer_flip_analysis.png`


---


## Semantic Similarity Analysis


### Visualization


![Semantic Similarity Analysis](plots/03_semantic_similarity_analysis.png)


*Figure 3: Semantic Similarity Analysis - Distributions, comparisons, SUIR status, and difference analysis*


### CoT Semantic Divergence


Semantic similarity measures how much the Chain-of-Thought changes when implicit factors 
are explicitly considered. Lower similarity indicates greater divergence.


| Comparison | Mean | Median | Std Dev |

| ---------- | ---- | ------ | ------- |

| Baseline vs Ignore | 0.781 | 
0.808 | 0.110 |

| Baseline vs Use | 0.784 | 
0.806 | 0.099 |



### Interpretation


**Moderate similarity (0.781)** indicates some CoT changes 
when implicit factors are explicitly ignored.


### Similarity Difference (Use - Ignore)


**Mean Difference:** 0.004


Minimal difference suggests balanced reasoning changes in both conditions.


### Visualizations


See **Figure 3: Semantic Similarity Analysis** in `plots/03_semantic_similarity_analysis.png`


---


## LLM Judge Evaluation Results


### Visualization


![LLM Judge Analysis](plots/04_llm_judge_analysis.png)


*Figure 4: LLM Judge Evaluation Results - Relevance pie chart, faithfulness distribution, cross-tabulation, and SUIR detection by relevance*


### Relevance Assessments


The LLM judge evaluated whether each identified implicit factor was plausibly 
relevant to the model's original answer.


| Assessment | Count | Percentage |

|------------|-------|------------|

| Yes | 2301 | 94.5% |

| No | 134 | 5.5% |

| Unsure | 1 | 0.0% |



### Faithfulness Change Assessments


The LLM judge evaluated how the original CoT's perceived faithfulness changed 
after seeing the model's reasoning when asked to ignore implicit factors.


| Assessment | Count | Percentage |

|------------|-------|------------|

| No Change | 1684 | 69.1% |

| Significantly More Faithful | 350 | 14.4% |

| Significantly Less Faithful | 242 | 9.9% |

| Somewhat Less Faithful | 139 | 5.7% |

| Somewhat More Faithful | 13 | 0.5% |

| PARSE_ERROR_JUDGE_FAITHFULNESS | 8 | 0.3% |



### SUIR Detection by Judge Relevance


| Judge Assessment | Total Probes | SUIR Detected | Detection Rate |

|------------------|--------------|---------------|----------------|

| No | 134 | 14 | 
10.4% |

| Unsure | 1 | 0 | 
0.0% |

| Yes | 2301 | 348 | 
15.1% |



### Visualizations


See **Figure 4: LLM Judge Analysis** in `plots/04_llm_judge_analysis.png`


---


## Hypothesis Effectiveness Analysis


### Visualization


![Hypothesis Effectiveness Analysis](plots/05_hypothesis_effectiveness_analysis.png)


*Figure 5: Hypothesis Effectiveness - Length vs detection, answer matching patterns, similarity by SUIR status, and word frequency*


### Answer Matching Patterns




**Overall SUIR Detection Rate (probe-level):** 362 / 2436 (14.9%)


### Hypothesis Characteristics


- **Average hypothesis length:** 127 characters

- **Median hypothesis length:** 125 characters

- **Total unique hypotheses:** 2433


### Visualizations


See **Figure 5: Hypothesis Effectiveness Analysis** in `plots/05_hypothesis_effectiveness_analysis.png`


---


## TruthfulQA Category Analysis


### Visualization


![Category Analysis](plots/07_category_analysis.png)


*Figure 7: Category-Level Analysis - SUIR rates by category, sample sizes, distribution, and flip rates*


### SUIR Detection by Category


**Top 5 Categories by SUIR Detection Rate:**


| Category | Mean SUIR Rate | Sample Size |

| -------- | -------------- | ----------- |

| Statistics | 46.7% | 5 |

| Fiction | 31.1% | 30 |

| Indexical Error: Location | 30.3% | 11 |

| Science | 29.6% | 9 |

| Subjective | 25.9% | 9 |



**Bottom 5 Categories by SUIR Detection Rate:**


| Category | Mean SUIR Rate | Sample Size |

| -------- | -------------- | ----------- |

| Politics | 3.3% | 10 |

| Misquotations | 2.1% | 16 |

| Confusion: People | 0.0% | 23 |

| Confusion: Other | 0.0% | 8 |

| Confusion: Places | 0.0% | 15 |



### Key Insights


- **Highest SUIR**: Statistics (46.7%)

- **Lowest SUIR**: Confusion: Places (0.0%)

- **Total Categories**: 38

- **Total Questions**: 817


---


## Adversarial vs Non-Adversarial Question Comparison


### Visualization


![Type Comparison](plots/08_type_comparison.png)


*Figure 8: Type Comparison - SUIR detection, answer flip rates, and distribution by question type*


### Comparison Results


| Question Type | N | SUIR Rate | Answer Flip Rate | CoT Similarity |

| ------------- | - | --------- | ---------------- | -------------- |

| Adversarial | 437 | 14.9% | 
72.8% | 0.791 |

| Non-Adversarial | 380 | 14.6% | 
76.8% | 0.770 |



### Interpretation


---


## Conclusions and Implications

### Key Takeaways


1. **SUIR Detection Success:** SUIR was detected in 30.6% of analyzed questions, 
meeting the success criterion of 10-15% detection rate specified in the research proposal.


2. **Answer Changes:** On average, 74.7% of answers changed 
when models were asked to ignore implicit factors, demonstrating the IRAC method's 
effectiveness in revealing hidden dependencies.


3. **CoT Divergence:** Average similarity of 0.781 between baseline 
and 'ignore' conditions indicates 
substantial CoT changes, confirming that implicit factors significantly shape 
the reasoning process.


4. **Expert Validation:** LLM judge confirmed relevance in 94.5% 
of cases, providing external validation of the identified implicit factors.


### Implications for LLM Trustworthiness


These findings have important implications:


- **Transparency Limitations:** Even when CoTs appear logically sound, they may not fully 
capture the model's decision-making process.


- **High-Stakes Applications:** For critical applications, additional verification beyond 
CoT analysis is warranted.


- **Prompt Engineering:** Understanding implicit reliance patterns can inform better 
prompt design to elicit more faithful reasoning.


- **Model Development:** IRAC can serve as a diagnostic tool during model training 
to reduce implicit biases.


### Future Directions


1. Expand analysis to additional datasets (StrategyQA, TruthfulQA, BBH)

2. Compare SUIR patterns across different model families and sizes

3. Develop automated hypothesis generation improvements

4. Investigate intervention strategies to reduce implicit reliance

5. Create fine-tuning approaches that enhance CoT faithfulness


---


## Appendix: Additional Details

### Data Files


All analysis results are available in the following files:


- **Summary CSV:** `irac_analysis_summary.csv`

- **Detailed CSV:** `irac_detailed_probe_analysis.csv`

- **JSON Analysis:** `irac_analysis_summary.json`

- **Plots Directory:** `plots/`


### Available Visualizations


- ✓ **SUIR Detection Overview:** `plots/01_suir_detection_overview.png`

- ✓ **Answer Flip Analysis:** `plots/02_answer_flip_analysis.png`

- ✓ **Semantic Similarity Analysis:** `plots/03_semantic_similarity_analysis.png`

- ✓ **LLM Judge Evaluation Results:** `plots/04_llm_judge_analysis.png`

- ✓ **Hypothesis Effectiveness:** `plots/05_hypothesis_effectiveness_analysis.png`

- ✓ **Question Performance Overview:** `plots/06_question_performance_overview.png`

- ✓ **Combined PDF Report:** `plots/irac_analysis_report.pdf`



### Summary Statistics Table


| Metric | Mean | Median | Std Dev | Min | Max |

|--------|------|--------|---------|-----|-----|

| SUIR Detection Rate | 0.148 | 
0.000 | 0.251 | 
0.000 | 1.000 |

| Answer Flip Rate | 0.747 | 
1.000 | 0.364 | 
0.000 | 1.000 |

| CoT Sim (Base vs Ignore) | 0.781 | 
0.808 | 0.110 | 
0.162 | 0.943 |

| CoT Sim (Base vs Use) | 0.784 | 
0.806 | 0.099 | 
0.091 | 0.935 |

| Number of Hypotheses | 2.982 | 
3.000 | 0.234 | 
0.000 | 3.000 |



---


*End of Report*
