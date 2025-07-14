#!/bin/bash                                                                                                                                             
#SBATCH --job-name slurm_cubes
#SBATCH --ntasks=1
#SBATCH --output /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_cubes.%a.%A.out
#SBATCH --error /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_cubes.%a.%A.err
#SBATCH --cpus-per-task=1                                                                                                      
#SBATCH --mem=12000M 
#SBATCH --array 1-1:1%1

export HEADASNOQUERY=""
export HEADASPROMPT="/dev/null/"

windows=(0 1 2 3 4 5 10 15 20)
offsets=(0 1 2 3 4 5)
#relevant_separations=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 35 40 45 50 55 60 70 80 90 100 110 126)
relevant_separations=(8 20 50 126 317 797)
sep_plot=20
threshold=6.
samplesUp=3
samplesDown=2
config_version="v5_20250621"
# The default is "False". The presence of the parameter activates it to "True". No possible --create_cubes False
# create_cubes=False


## WARNING: 
#    Use --arg value for flags that take multiple values.
#    Use --arg=value only when the value is a single item (like a string or number).

echo "Runing: python3 ./create_cubes.py --windows ${windows[@]} --offsets ${offsets[@]} --threshold=$threshold --samplesUp=$samplesUp --samplesDown=$samplesDown --config_version=$config_version --create_cubes=$create_cubes"

python3 ./create_cubes.py --windows "${windows[@]}" --offsets "${offsets[@]}" --threshold=$threshold --samplesUp=$samplesUp --samplesDown=$samplesDown --config_version=$config_version --relevant_separations "${relevant_separations[@]}" --sep_for_plot_mosaic=$sep_plot # --create_cubes

