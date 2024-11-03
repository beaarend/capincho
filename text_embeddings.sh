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

# Install dependencies from requirements.txt
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Run the Python script
python textLoader.py --path $PATH --model $MODEL --output $OUTPUT

# Deactivate the virtual environment
deactivate
