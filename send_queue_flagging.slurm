#!/bin/bash                                                                                                                                             
#SBATCH --job-name slurm_flagging
#SBATCH --ntasks=1
#SBATCH --output /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_flagging.%a.%A.out
#SBATCH --error /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_flagging.%a.%A.err
#SBATCH --cpus-per-task=1                                                                                                      
#SBATCH --mem=12000M 
#SBATCH --array 1-9:1%9

export HEADASNOQUERY=""
export HEADASPROMPT="/dev/null/"

icpu=$((${SLURM_ARRAY_TASK_ID}-1))

fluxes_mcrab=(0.32 0.5 1. 3.2 10 31.6 100 316.2 1000)

flux=${fluxes_mcrab[$icpu]}
echo "Using Flux: $flux"

# leave default parameters
#datadir="/dataj6SIRENA/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/singles"
#secondary_samples = 1563
#verbose=0
#simEnergies=(0.2,  0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
#nsigmas=5 
#poly_order=8
#model="crab"

filter="nofilt"
focus="defoc"
config="v5_20250621"

if [ 1 -eq "$(echo "$flux < 1." | bc)" ]
then
    focus="infoc"
fi

if (( $(echo "$flux > 999" |bc -l) )); then
    filter="thickBe"
fi

echo "Runing: python3 ./execute_flagging.py --focus=$focus --filter=$filter --flux_mcrab=$flux"

python3 ./execute_flagging.py --focus=$focus --filter=$filter --flux_mcrab=$flux --config_version=$config







