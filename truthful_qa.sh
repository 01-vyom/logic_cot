DATASET_NAME = "truthful_qa"
DATASET_CONFIG = "generation"
DATASET_SPLIT = "train"  # This is TruthfulQA's test set (817 questions)
MAX_QUESTIONS_TO_PROCESS = 10  # Start with subset, then use None for all

python run_experiment.py


python3 run_experiment.py &> /home/ec2-user/code/personal/logic_cot_truthful_qa.out 
