#!/bin/bash                                                                                                                                             
#SBATCH --job-name slurm_simulate_CrabSpec_1s
#SBATCH --ntasks=1
#SBATCH --output /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_simulate_1s.%a.%A.out
#SBATCH --error /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_simulate_1s.%a.%A.err
#SBATCH --cpus-per-task=1                                                                                                      
#SBATCH --mem=4000M 
##SBATCH --array 1-100:1%30

export HEADASNOQUERY=""
export HEADASPROMPT="/dev/null/"

# closest flux to EC's file is 950
# we use 1150 to match case of additional 5ct/s/pix due to MXS
flux_mcrab=1150
exposure=1.
nonxbgd="no"
XTalk="none"
python3 ./execute_simulate_CrabSpec_1s.py --flux_mcrab=${flux_mcrab} --exposure=${exposure} --nonxbgd=${nonxbgd} --XTalk=${XTalk} --verbose=0






