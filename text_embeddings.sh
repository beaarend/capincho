#!/bin/bash
#SBATCH --job-name=my_python_job     # Job name
#SBATCH --output=output_%j.log       # Output log file (%j will be replaced by the job ID)
#SBATCH --error=error_%j.log         # Error log file (%j will be replaced by the job ID)
#SBATCH --ntasks=1                   # Number of tasks (processes)

# Access positional arguments
PATH=$1
MODEL=$2
OUTPUT=$3

# Load any necessary modules (Python module, if needed)
module load python/3.8                # Adjust the Python version as necessary

# Install dependencies from requirements.txt
pip install --upgrade pip             # Ensure pip is up-to-date
pip install -r requirements.txt

# Run the Python script
python textLoader.py --path $PATH --model $MODEL --output $OUTPUT

# Deactivate the virtual environment
deactivate
