#!/bin/bash
#SBATCH --job-name=qwen3-32B-logic-cot-gsm8k-1000
#SBATCH --output=qwen3-32B-logic-cot-gsm8k-1000.out
#SBATCH --error=qwen3-32B-logic-cot-gsm8k-1000.err
#SBATCH --account=daisyw
#SBATCH --qos=daisyw
#SBATCH --mail-type=ALL
#SBATCH --mail-user=angerstick6@gmail.com
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4          
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100gb
#SBATCH --time=100:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module purge
ml mamba

mamba activate logic_cot_env

export TOKENIZERS_PARALLELISM=false

T1=$(date +%s)
python3 run_experiment.py
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"