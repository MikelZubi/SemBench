#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=WSD_LLM
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --time=08:00:00
#SBATCH --output=./SLURM/LOG/irteeraWSD.log
#SBATCH --error=./SLURM/ERRORE/erroreWSD.err

module load Python


source ~/inguruneak/GRAL/bin/activate

huggingface-cli login --token $(cat tokenHF.txt)



echo "Mistral"
#srun python WSD/LLMsWiC.py Mistral 0 Oxford
#srun python WSD/LLMsWiC.py Mistral 1 Oxford
#srun python WSD/LLMsWiC.py Mistral 2 Oxford
#srun python WSD/LLMsWiC.py Mistral 3 Oxford
#srun python WSD/LLMsWiC.py Mistral 4 Oxford
#srun python WSD/LLMsWiC.py Mistral 5 Oxford

echo "Zephyr"
#srun python WSD/LLMsWiC.py Zephyr 0 Oxford
#srun python WSD/LLMsWiC.py Zephyr 1 Oxford
#srun python WSD/LLMsWiC.py Zephyr 2 Oxford
#srun python WSD/LLMsWiC.py Zephyr 3 Oxford
#srun python WSD/LLMsWiC.py Zephyr 4 Oxford
#srun python WSD/LLMsWiC.py Zephyr 5 Oxford

echo "Starling"
#srun python WSD/LLMsWiC.py Starling 0 Oxford
#srun python WSD/LLMsWiC.py Starling 1 Oxford
#srun python WSD/LLMsWiC.py Starling 2 Oxford
#srun python WSD/LLMsWiC.py Starling 3 Oxford
#srun python WSD/LLMsWiC.py Starling 4 Oxford
#srun python WSD/LLMsWiC.py Starling 5 Oxford

echo "Llama3"
#srun python WSD/LLMsWiC.py Llama3 0 Oxford
#srun python WSD/LLMsWiC.py Llama3 1 Oxford
#srun python WSD/LLMsWiC.py Llama3 2 Oxford
#srun python WSD/LLMsWiC.py Llama3 3 Oxford
#srun python WSD/LLMsWiC.py Llama3 4 Oxford
#srun python WSD/LLMsWiC.py Llama3 5 Oxford

echo "OpenChat"
#srun python WSD/LLMsWiC.py OpenChat 0 Oxford
#srun python WSD/LLMsWiC.py OpenChat 1 Oxford
#srun python WSD/LLMsWiC.py OpenChat 2 Oxford
#srun python WSD/LLMsWiC.py OpenChat 3 Oxford
#srun python WSD/LLMsWiC.py OpenChat 4 Oxford
#srun python WSD/LLMsWiC.py OpenChat 5 Oxford

echo "MistralDPO"
#srun python WSD/LLMsWiC.py MistralDPO 0 Oxford
#srun python WSD/LLMsWiC.py MistralDPO 1 Oxford
#srun python WSD/LLMsWiC.py MistralDPO 2 Oxford
#srun python WSD/LLMsWiC.py MistralDPO 3 Oxford
#srun python WSD/LLMsWiC.py MistralDPO 4 Oxford
#srun python WSD/LLMsWiC.py MistralDPO 5 Oxford

echo "Llama3_70B"
#python WSD/LLMsWiCVLLM.py Llama3_70B 0 Oxford
#srun python WSD/LLMsWiC.py Llama3_70B 1 Oxford
#srun python WSD/LLMsWiC.py Llama3_70B 2 Oxford
#srun python WSD/LLMsWiC.py Llama3_70B 3 Oxford
#srun python WSD/LLMsWiC.py Llama3_70B 4 Oxford
python WSD/LLMsWiCVLLM.py Llama3_70B 5 Oxford

echo "Llama2"
#srun python WSD/LLMsWiC.py Llama2 0 Oxford
#srun python WSD/LLMsWiC.py Llama2 1 Oxford
#srun python WSD/LLMsWiC.py Llama2 2 Oxford
#srun python WSD/LLMsWiC.py Llama2 3 Oxford
#srun python WSD/LLMsWiC.py Llama2 4 Oxford
#srun python WSD/LLMsWiC.py Llama2 5 Oxford

echo "Gemma2 9B"
#srun python WSD/LLMsWiC.py Gemma2_9B 0 Oxford
#srun python WSD/LLMsWiC.py Gemma2_9B 1 Oxford
#srun python WSD/LLMsWiC.py Gemma2_9B 2 Oxford
#srun python WSD/LLMsWiC.py Gemma2_9B 3 Oxford
#srun python WSD/LLMsWiC.py Gemma2_9B 4 Oxford
#srun python WSD/LLMsWiC.py Gemma2_9B 5 Oxford

echo "Gemma2 27B"
#srun python WSD/LLMsWiC.py Gemma2_27B 0 Oxford
#srun python WSD/LLMsWiC.py Gemma2_27B 1 Oxford
#srun python WSD/LLMsWiC.py Gemma2_27B 2 Oxford
#srun python WSD/LLMsWiC.py Gemma2_27B 3 Oxford
#srun python WSD/LLMsWiC.py Gemma2_27B 4 Oxford
#srun python WSD/LLMsWiC.py Gemma2_27B 5 Oxford

python WSD/generateTable.py
 


