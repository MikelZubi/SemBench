#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=WSDE_LLM
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=4-00:00:00
#SBATCH --output=./SLURM/LOG/irteeraWSDE.log
#SBATCH --error=./SLURM/ERRORE/erroreWSDE.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus

module load Python


source ~/inguruneak/GRAL/bin/activate

huggingface-cli login --token $(cat tokenHF.txt)



echo "Mistral"
#python WSD/LLMsWiCExample.py Mistral 0 Oxford
#srun python WSD/LLMsWiCExample.py Mistral 1 Oxford
#srun python WSD/LLMsWiCExample.py Mistral 2 Oxford
#srun python WSD/LLMsWiCExample.py Mistral 3 Oxford
#srun python WSD/LLMsWiCExample.py Mistral 4 Oxford
#python WSD/LLMsWiCExample.py Mistral 5 Oxford
python WSD/LLMsWiCExample.py --modelname Mistral --k 5 --language ES
python WSD/LLMsWiCExample.py --modelname Mistral --k 5 --language EN
python WSD/LLMsWiCExample.py --modelname Mistral --k 0 --language EN

echo "Zephyr"
#python WSD/LLMsWiCExample.py Zephyr 0 Oxford
#srun python WSD/LLMsWiCExample.py Zephyr 1 Oxford
#srun python WSD/LLMsWiCExample.py Zephyr 2 Oxford
#srun python WSD/LLMsWiCExample.py Zephyr 3 Oxford
#srun python WSD/LLMsWiCExample.py Zephyr 4 Oxford
python WSD/LLMsWiCExample.py --modelname Zephyr --k 5 --language ES
python WSD/LLMsWiCExample.py --modelname Zephyr --k 5 --language EN
python WSD/LLMsWiCExample.py --modelname Zephyr --k 0 --language EN

echo "Starling"
#python WSD/LLMsWiCExample.py Starling 0 Oxford
#srun python WSD/LLMsWiCExample.py Starling 1 Oxford
#srun python WSD/LLMsWiCExample.py Starling 2 Oxford
#srun python WSD/LLMsWiCExample.py Starling 3 Oxford
#srun python WSD/LLMsWiCExample.py Starling 4 Oxford
python WSD/LLMsWiCExample.py --modelname Starling --k 5 --language ES
python WSD/LLMsWiCExample.py --modelname Starling --k 5 --language EN
python WSD/LLMsWiCExample.py --modelname Starling --k 0 --language EN

echo "Llama3"
#python WSD/LLMsWiCExampleVLLM.py Llama3 0 Oxford
#srun python WSD/LLMsWiCExample.py Llama3 1 Oxford
#srun python WSD/LLMsWiCExample.py Llama3 2 Oxford
#srun python WSD/LLMsWiCExample.py Llama3 3 Oxford
#srun python WSD/LLMsWiCExample.py Llama3 4 Oxford
python WSD/LLMsWiCExampleVLLM.py --modelname Llama3 --k 5 --language ES
python WSD/LLMsWiCExampleVLLM.py --modelname Llama3 --k 5 --language EN
python WSD/LLMsWiCExampleVLLM.py --modelname Llama3 --k 0 --language EN

#echo "OpenChat"
##srun python WSD/LLMsWiCExample.py OpenChat 0 Oxford
#srun python WSD/LLMsWiCExample.py OpenChat 1 Oxford
#srun python WSD/LLMsWiCExample.py OpenChat 2 Oxford
#srun python WSD/LLMsWiCExample.py OpenChat 3 Oxford
#srun python WSD/LLMsWiCExample.py OpenChat 4 Oxford
#srun python WSD/LLMsWiCExample.py OpenChat 5 Oxford

echo "MistralDPO"
#python WSD/LLMsWiCExample.py MistralDPO 0 Oxford
#srun python WSD/LLMsWiCExample.py MistralDPO 1 Oxford
#srun python WSD/LLMsWiCExample.py MistralDPO 2 Oxford
#srun python WSD/LLMsWiCExample.py MistralDPO 3 Oxford
#srun python WSD/LLMsWiCExample.py MistralDPO 4 Oxford
python WSD/LLMsWiCExample.py --modelname MistralDPO --k 5 --language ES
python WSD/LLMsWiCExample.py --modelname MistralDPO --k 5 --language EN
python WSD/LLMsWiCExample.py --modelname MistralDPO --k 0 --language EN

echo "Llama2"
#python WSD/LLMsWiCExampleVLLM.py Llama2 0 Oxford
#srun python WSD/LLMsWiCExample.py Llama2 1 Oxford
#srun python WSD/LLMsWiCExample.py Llama2 2 Oxford
#srun python WSD/LLMsWiCExample.py Llama2 3 Oxford
#srun python WSD/LLMsWiCExample.py Llama2 4 Oxford
python WSD/LLMsWiCExampleVLLM.py --modelname Llama2 --k 5 --language ES
python WSD/LLMsWiCExampleVLLM.py --modelname Llama2 --k 5 --language EN
python WSD/LLMsWiCExampleVLLM.py --modelname Llama2 --k 0 --language EN

echo "Gemma2 9B"
#python WSD/LLMsWiCExampleVLLM.py Gemma2_9B 0 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_9B 1 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_9B 2 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_9B 3 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_9B 4 Oxford
python WSD/LLMsWiCExampleVLLM.py --modelname Gemma2_9B --k 5 --language ES
python WSD/LLMsWiCExampleVLLM.py --modelname Gemma2_9B --k 5 --language EN
python WSD/LLMsWiCExampleVLLM.py --modelname Gemma2_9B --k 0 --language EN

echo "Gemma2 27B"
#python WSD/LLMsWiCExampleVLLM.py Gemma2_27B 0 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_27B 1 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_27B 2 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_27B 3 Oxford
#srun python WSD/LLMsWiCExample.py Gemma2_27B 4 Oxford
python WSD/LLMsWiCExampleVLLM.py --modelname Gemma2_27B --k 5 --language ES
python WSD/LLMsWiCExampleVLLM.py --modelname Gemma2_27B --k 5 --language EN
python WSD/LLMsWiCExampleVLLM.py --modelname Gemma2_27B --k 0 --language EN

echo "Llama3_70B"
#python WSD/LLMsWiCExampleVLLM.py Llama3_70B 0 Oxford
#srun python WSD/LLMsWiCExample.py Llama3_70B 1 Oxford
#srun python WSD/LLMsWiCExample.py Llama3_70B 2 Oxford
#srun python WSD/LLMsWiCExample.py Llama3_70B 3 Oxford
#srun python WSD/LLMsWiCExample.py Llama3_70B 4 Oxford
#srun python WSD/LLMsWiCExample.py Llama3_70B 5 Oxford

 


