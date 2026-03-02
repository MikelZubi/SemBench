#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=WiC_LLM
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --time=12:00:00
#SBATCH --output=./SLURM/LOG/irteeraWSD70B.log
#SBATCH --error=./SLURM/ERRORE/erroreWSD70B.err

module load Python

source ~/inguruneak/GRAL/bin/activate

huggingface-cli login --token $(cat tokenHF.txt)
#models=("Llama2" "Llama3_8B" "Llama3_70B" "Gemma3_4B" "Gemma3_12B" "Gemma3_27B" "Qwen3_8B" "Qwen3_14B" "Qwen3_32B" "Latxa")
models=("Llama3_70B" "Latxa")


for model in "${models[@]}"; do
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 5 --language EU
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 5 --language ES
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 5 --language EN
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 0 --language EN
    python WSD/LLMsWiCVLLM.py --modelname $model --k 5 --language EN
    python WSD/LLMsWiCVLLM.py --modelname $model --k 0 --language EN

done

 


