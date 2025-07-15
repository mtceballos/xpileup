#!/bin/bash                                                                                                                                             
#SBATCH --job-name slurm_simulate_source
#SBATCH --ntasks=1
#SBATCH --output /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_simulate_src.%a.%A.out
#SBATCH --error /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_simulate_src.%a.%A.err
#SBATCH --cpus-per-task=1                                                                                                      
#SBATCH --mem=4000M 
#SBATCH --array 1-100:1%30
##SBATCH --array 101-200:1%30

recons=0
export HEADASNOQUERY=""
export HEADASPROMPT="/dev/null/"

icpu=${SLURM_ARRAY_TASK_ID}
((icpu_plus100=${icpu}+100))
Emin=2.0
Emax=10.0
model='crab'
focus='' # to be automatically determined based on Flux
fil='nofilt'
config_version="v5_20250621"

#filters=('nofilt' 'thinBe' 'thickBe')
fluxes_mcrab=(0.32 0.5 1. 3.2 10 31.6 100 316.2)
#fluxes_mcrab=(0.32 0.5 1. 3.2 10 31.6 100 316.2 1000)

for flux_mcrab in ${fluxes_mcrab[@]}; do
    python3 ./execute_simulate_source.py --sim_number=${icpu} --flux_mcrab=${flux_mcrab} --Emin=${Emin} --Emax=${Emax} --model=${model} --filter=${fil} --focus=${focus} --recons=${recons} --verbose=0 --config_version=${config_version}
done

python3 ./execute_simulate_source.py --sim_number=${icpu} --flux_mcrab=1000 --Emin=${Emin} --Emax=${Emax} --model=${model} --filter='thickBe' --focus=${focus} --recons=${recons} --verbose=0 --config_version=${config_version}

python3 ./execute_simulate_source.py --sim_number=${icpu_plus100} --flux_mcrab=1000 --Emin=${Emin} --Emax=${Emax} --model=${model} --filter='thickBe' --focus=${focus} --recons=${recons} --verbose=0 --config_version=${config_version}




