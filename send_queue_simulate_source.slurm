#!/bin/bash                                                                                                                                             
#SBATCH --job-name slurm_simulate_source
#SBATCH --ntasks=1
#SBATCH --output /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_simulate_src.%a.%A.out
#SBATCH --error /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_simulate_src.%a.%A.err
#SBATCH --cpus-per-task=1                                                                                                      
#SBATCH --mem=4000M 
#SBATCH --array 1-30:1%30

export HEADASNOQUERY=""
export HEADASPROMPT="/dev/null/"

icpu=${SLURM_ARRAY_TASK_ID}
Emin=2.0
Emax=10.0
model='crab'
focus='' # to be automatically determined based on Flux


#filters=('nofilt' 'thinBe' 'thickBe')
#fluxes_mcrab=(0.0001 0.0032 0.001 0.0032 0.01 0.032 0.1 0.32 1. 3.2 10 31.6 100 316.2 1000)
#flux_mcrab=${fluxes_mcrab[$icpu]}
flux_mcrab=0.5

#for fil in ${filters[@]}; do
#    python3 ./execute_simulate_source.py --sim_number=${icpu} --flux_mcrab=${flux_mcrab} --Emin=${Emin} --Emax=${Emax} --model=${model} --filter=${fil} --focus=${focus} --recons=1 --verbose=0
#done

fil='nofilt'
python3 ./execute_simulate_source.py --sim_number=${icpu} --flux_mcrab=${flux_mcrab} --Emin=${Emin} --Emax=${Emax} --model=${model} --filter=${fil} --focus=${focus} --recons=1 --verbose=0





