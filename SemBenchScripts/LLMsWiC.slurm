#!/bin/bash
#SBATCH --qos=long
#SBATCH --job-name=WiC_LLM
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-sxm4
#SBATCH --time=2-00:00:00
#SBATCH --output=./SLURM/LOG/irteeraWSD.log
#SBATCH --error=./SLURM/ERRORE/erroreWSD.err

module load Python

source ~/inguruneak/GRAL/bin/activate

huggingface-cli login --token $(cat tokenHF.txt)
models=("Llama2" "Llama3_8B" "Llama3_70B" "Gemma3_4B" "Gemma3_12B" "Gemma3_27B" "Qwen3_8B" "Qwen3_14B" "Qwen3_32B" "Latxa_70B" "Latxa_8B" "Qwen3_4B")

for model in "${models[@]}"; do
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 0 --language ES
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 0 --language EU
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 0 --language EN
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 5 --language ES
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 5 --language EU
    python WSD/LLMsWiCExampleVLLM.py --modelname $model --k 5 --language EN

    python WSD/LLMsWiCVLLM.py --modelname $model --k 5 --language EN
    python WSD/LLMsWiCVLLM.py --modelname $model --k 0 --language EN
done




