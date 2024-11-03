#!/bin/bash
#SBATCH --nodes=1 #Número de Nós
#SBATCH --ntasks-per-node=24 #Número de tarefas por Nó
#SBATCH --ntasks=24 #Número total de tarefas MPI
#SBATCH --cpus-per-task=1 #Número de threads por tarefas
#SBATCH -p gpu #Fila (partition) a ser utilizada
#SBATCH -J text_features #Nome job
#SBATCH --exclusive #Utilização exclusiva dos nós durante a execução do job
#SBATCH --chdir=/u/fibz/capincho #path a partir do qual será executado o job
#SBATCH --account=tornado #Conta do projeto
#SBATCH --output=/u/fibz/capincho/logs/saida.out #arquivo onde será escrita a saída stdout da execução job
#SBATCH --error=/u/fibz/capincho/logs/saida.err #arquivo onde será escrito os erros de execução job tderr

# Access positional arguments
PATH=$1
MODEL=$2
OUTPUT=$3

# Load any necessary modules (Python module, if needed)
module load python/3.8.0
module load cuda/11.6

# Install dependencies one by one to check specific err logs
pip3 install git+https://github.com/openai/CLIP.git
pip3 install --upgrade pip
pip3 install huggingface-hub==0.25.1
pip3 install matplotlib==3.7.5
pip3 install numpy==1.24.3
pip3 install open_clip_torch==2.29.0
pip3 install pandas==2.0.3
pip3 install peft==0.13.0
pip3 install pillow==10.4.0
pip3 install pycocoevalcap==1.2
pip3 install pycocotools==2.0.7
pip3 install tokenizers==0.20.0
pip3 install torch==2.4.1
pip3 install torchvision==0.19.1
pip3 install tqdm==4.66.5
pip3 install transformers==4.45.1
pip3 install openpyxl

# Run the Python script
python textLoader.py --path $PATH --model $MODEL --output $OUTPUT

# Deactivate the virtual environment
deactivate
