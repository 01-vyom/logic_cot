# Multi-Dataset IRAC Analysis Guide

This guide explains how to run IRAC experiments on different datasets (GSM8K, TruthfulQA, and StrategyQA) and analyze the results.

## Quick Start

### Running Experiments

**For GSM8K (math reasoning):**
```bash
# In run_experiment.py, set:
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
MAX_QUESTIONS_TO_PROCESS = 10  # or None for all

python run_experiment.py
```

**For TruthfulQA (truthfulness / hallucination):**
```bash
# In run_experiment.py, set:
DATASET_NAME = "truthful_qa"
DATASET_CONFIG = "generation"
DATASET_SPLIT = "validation"  # TruthfulQA's test set (817 questions)
MAX_QUESTIONS_TO_PROCESS = 10  # Start with subset, then use None for all

python run_experiment.py
```

**For StrategyQA (multi-hop reasoning):**
```bash
# In run_experiment.py, set:
DATASET_NAME = "strategy_qa"
DATASET_CONFIG = None  # Not needed for StrategyQA
DATASET_SPLIT = "test"  # 687 questions
MAX_QUESTIONS_TO_PROCESS = None  # or set to integer for subset

python run_experiment.py
```

### Using --subset for Quick Testing

All datasets support the `--subset` flag to test on a small number of questions first:

```bash
# Test on 20 questions only
python run_experiment.py --subset 20

# Test with a custom output directory suffix
python run_experiment.py --subset 20 --output-suffix "_test"
```

### Analyzing Results

**For GSM8K (default):**
```bash
python analyze_results.py
python create_plots.py
python generate_report.py
```

**For TruthfulQA:**
```bash
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_truthful_qa
python create_plots.py /home/ec2-user/code/personal/logic_cot/experiment_results_truthful_qa
python generate_report.py /home/ec2-user/code/personal/logic_cot/experiment_results_truthful_qa
```

**For StrategyQA:**
```bash
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
python create_plots.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
python generate_report.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
```

## Dataset Specifications

### GSM8K
- **Type:** Grade school math word problems
- **What it tests:** Mathematical calculation and reasoning
- **Answer Format:** Numerical values
- **Fields Used:** `question`, `answer`
- **Answer Comparison:** Exact string matching
- **Hypothesis Context:** Mathematical reasoning principles and common student errors
- **Output Directory:** `experiment_results_gsm8k/`

### TruthfulQA
- **Type:** Questions designed to elicit truthful vs imitative falsehood responses
- **What it tests:** What the model knows (factual accuracy)
- **Answer Format:** Short text (1-2 sentences)
- **Fields Used:** `question`, `best_answer`, `correct_answers`, `incorrect_answers`, `category`, `type`
- **Answer Comparison:** LLM-based semantic equivalence
- **Hypothesis Prompt:** Avoids meta-language, focuses on factual premises
- **Output Directory:** `experiment_results_truthful_qa/`
- **Categories:** Health, Law, Finance, Politics, and 34 other categories

### StrategyQA
- **Type:** Multi-hop yes/no reasoning questions requiring fact chaining
- **What it tests:** How the model reasons (multi-hop logic)
- **Answer Format:** Boolean (True/False)
- **Fields Used:** `qid`, `question`, `answer` (bool), `facts`, `term`, `description`
- **Answer Comparison:** Boolean normalization (True/Yes→"true", False/No→"false")
- **Hypothesis Prompt:** Focuses on reasoning chain dependencies and bridging knowledge
- **Output Directory:** `experiment_results_strategy_qa/`
- **Source:** [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA)
- **Paper:** [Geva et al. 2021](https://arxiv.org/abs/2101.02235)
- **Test Split:** 687 questions

## Configuration Parameters

### In `run_experiment.py`

```python
# Core Configuration
MODEL_NAME = DEFAULT_MODEL_HF
TEMPERATURE_BASELINE = 0.0
TEMPERATURE_HYPOTHESIS_GEN = 0.0
TEMPERATURE_PROBE = 0.0
MAX_NEW_TOKENS = 10_000
NUM_HYPOTHESES_TO_PROBE_PER_QUESTION = 3

# Dataset Configuration
DATASET_NAME = "gsm8k" or "truthful_qa" or "strategy_qa"
DATASET_CONFIG = "main" or "generation" or None
DATASET_SPLIT = "test" or "validation"
MAX_QUESTIONS_TO_PROCESS = None  # None for all, or integer for subset
```

### Command Line Arguments

```bash
python run_experiment.py --subset 50          # Test on first 50 questions
python run_experiment.py --output-suffix _v2  # Output to experiment_results_{dataset}_v2
python run_experiment.py --subset 20 --output-suffix _test  # Both
```

## Output Structure

### Directory Organization
```
experiment_results_{dataset_name}/
├── all_experiment_results.csv
├── _all_experiment_logs.json
├── irac_analysis_summary.csv
├── irac_analysis_summary.json
├── irac_detailed_probe_analysis.csv
├── IRAC_ANALYSIS_REPORT.md
└── plots/
    ├── 01_suir_detection_overview.png
    ├── 02_answer_flip_analysis.png
    ├── 03_semantic_similarity_analysis.png
    ├── 04_llm_judge_analysis.png
    ├── 05_hypothesis_effectiveness_analysis.png
    ├── 06_question_performance_overview.png
    ├── 07_category_analysis.png          # TruthfulQA/StrategyQA only
    ├── 08_type_comparison.png            # TruthfulQA only
    └── irac_analysis_report.pdf
```

## Key Differences Between Datasets

### Answer Parsing

**GSM8K:**
- Numerical answers extracted from `\boxed{}`, `$$...$$`, or `$...$` patterns
- Focus on numerical value extraction
- Example: "The final answer is $42$" → `42`

**TruthfulQA:**
- Text-based answers (Yes/No, short text)
- Early detection of Yes/No/True/False patterns
- Falls back to extracting short text (≤100 chars, ≤10 words)
- Example: "The final answer is No" → `No`

**StrategyQA:**
- Boolean answers (True/False, Yes/No)
- Parser detects Yes/No/True/False at start of answer
- Normalized during analysis: True/Yes → "true", False/No → "false"
- Example: "The final answer is True" → `True` (normalized to `true` in analysis)

### Answer Comparison

| Dataset | Method | Speed | Why |
|---------|--------|-------|-----|
| GSM8K | Exact string match | Fast | Numbers are deterministic |
| TruthfulQA | LLM semantic equivalence | Slow | Open-ended text answers |
| StrategyQA | Boolean normalization | Fast | Binary answers (True/False) |

### Hypothesis Generation

**GSM8K:**
```
"Consider general mathematical reasoning principles and common student errors."
```

**TruthfulQA:**
```
"Consider common misconceptions about {category}, false beliefs, imitative 
falsehoods, and factors that lead people to answer incorrectly."
```
- Avoids meta-language ("the model might...")
- Focuses on factual premises

**StrategyQA:**
```
Focuses on multi-hop reasoning:
- Missing intermediate reasoning steps
- Implicit fact connections between entities
- Bridging knowledge required to chain facts
- Temporal/causal relationships not explicitly stated
```
- Avoids meta-language
- Focuses on reasoning chain dependencies

## Analysis Pipeline

### Step 1: Run Experiment
```bash
python run_experiment.py
```
**Output:** `all_experiment_results.csv`, `_all_experiment_logs.json`

### Step 2: Analyze Results
```bash
python analyze_results.py [results_dir]
```
**Output:** `irac_analysis_summary.csv`, `irac_analysis_summary.json`, `irac_detailed_probe_analysis.csv`

### Step 3: Create Visualizations
```bash
python create_plots.py [results_dir]
```
**Output:** PNG files in `plots/` directory, `irac_analysis_report.pdf`

### Step 4: Generate Report
```bash
python generate_report.py [results_dir]
```
**Output:** `IRAC_ANALYSIS_REPORT.md`

## Full Example Workflows

### Running StrategyQA (Recommended First Run)

```bash
# 1. Edit run_experiment.py:
# DATASET_NAME = "strategy_qa"
# DATASET_CONFIG = None
# DATASET_SPLIT = "test"

# 2. Test on 20 questions first
python run_experiment.py --subset 20 --output-suffix "_test"

# 3. Analyze the test results
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa_test
python create_plots.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa_test
python generate_report.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa_test

# 4. If results look good, run full experiment (687 questions)
python run_experiment.py

# 5. Analyze full results
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
python create_plots.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
python generate_report.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
```

### Running All Three Datasets Sequentially

```bash
# 1. Run GSM8K experiment
# Set: DATASET_NAME = "gsm8k", DATASET_CONFIG = "main", DATASET_SPLIT = "test"
python run_experiment.py

# 2. Analyze GSM8K
python analyze_results.py
python create_plots.py
python generate_report.py

# 3. Run TruthfulQA experiment
# Set: DATASET_NAME = "truthful_qa", DATASET_CONFIG = "generation", DATASET_SPLIT = "validation"
python run_experiment.py

# 4. Analyze TruthfulQA
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_truthful_qa
python create_plots.py /home/ec2-user/code/personal/logic_cot/experiment_results_truthful_qa
python generate_report.py /home/ec2-user/code/personal/logic_cot/experiment_results_truthful_qa

# 5. Run StrategyQA experiment
# Set: DATASET_NAME = "strategy_qa", DATASET_CONFIG = None, DATASET_SPLIT = "test"
python run_experiment.py

# 6. Analyze StrategyQA
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
python create_plots.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
python generate_report.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
```

## Backward Compatibility

All analysis scripts maintain backward compatibility:

- **Without arguments:** Use default GSM8K directory (`experiment_results`)
- **With argument:** Use specified dataset directory

Existing GSM8K experiments continue to work unchanged.

## Cross-Dataset Comparison

| Feature | GSM8K | TruthfulQA | StrategyQA |
|---------|-------|------------|------------|
| Domain | Math / Logic | Common Sense / Hallucination | Multi-hop Logic |
| Answer Type | Numerical | Text | Boolean (True/False) |
| Questions | 1,319 (test) | 817 (validation) | 687 (test) |
| Hypothesis Focus | Calculation shortcuts | Factual premises | Reasoning chain gaps |
| Comparison Method | Exact match | LLM semantic | Boolean normalization |
| Expected SUIR Rate | ~21% | Variable | TBD |

## Troubleshooting

### Issue: Parse errors in StrategyQA answers
**Solution:** The parser handles Yes/No/True/False at the start of answers. If the model outputs longer text, the parser extracts the core boolean via the same patterns used for TruthfulQA.

### Issue: High flip rate on StrategyQA
**Note:** Boolean answers should have less sensitivity than TruthfulQA (only two possible values), so flip rates in the 20-60% range would indicate genuine reasoning sensitivity.

### Issue: Wrong output directory
**Solution:** Verify `DATASET_NAME` in `run_experiment.py` matches your intended dataset.

### Issue: Resume capability
All datasets support automatic resume. If an experiment is interrupted, simply run the same command again and it will pick up where it left off.

### Issue: Analysis script can't find CSV
**Solution:** Pass the correct results directory as command-line argument:
```bash
python analyze_results.py /home/ec2-user/code/personal/logic_cot/experiment_results_strategy_qa
```

## StrategyQA-Specific Notes

### Multi-Hop Reasoning Structure

StrategyQA questions require connecting 2+ facts to reach a boolean answer:

```
Question: "Was ship that recovered Apollo 13 named after a World War II battle?"
Fact 1: Apollo 13 was recovered by the USS Iwo Jima
Fact 2: Battle of Iwo Jima was a WW2 battle
Answer: True (connecting both facts)
```

### What IRAC Reveals for StrategyQA

The IRAC method is particularly effective for StrategyQA because:
1. **Multi-hop shortcuts:** Model might skip intermediate reasoning steps
2. **Entity association:** Model might rely on keyword associations rather than logical chains
3. **Reasoning faithfulness:** CoT might show full reasoning, but model may internally shortcut

### Ground Truth Facts

StrategyQA provides ground truth `facts` for each question. These are stored in `ground_truth_answer_details` but **not used during hypothesis generation** (to test pure model reasoning). They can be used for post-hoc evaluation.