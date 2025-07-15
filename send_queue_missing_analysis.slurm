#!/bin/bash                                                                                                                                             
#SBATCH --job-name slurm_missing_analysis
#SBATCH --ntasks=1
#SBATCH --output /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_missing.%a.%A.out
#SBATCH --error /dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/slurm_missing.%a.%A.err
#SBATCH --cpus-per-task=1                                                                                                      
#SBATCH --mem=12000M 
#SBATCH --array 1-9:1%9

icpu=$((${SLURM_ARRAY_TASK_ID}-1))

csvs=('info_nofilt_infoc_global_0.320mCrab.csv' \
	  'info_nofilt_infoc_global_0.500mCrab.csv' \
	  'info_nofilt_defoc_global_1.000mCrab.csv' \
	  'info_nofilt_defoc_global_3.200mCrab.csv' \
	  'info_nofilt_defoc_global_10.000mCrab.csv' \
	  'info_nofilt_defoc_global_31.600mCrab.csv' \
	  'info_nofilt_defoc_global_100.000mCrab.csv' \
	  'info_nofilt_defoc_global_316.200mCrab.csv' \
	  'info_thickBe_defoc_global_1000.000mCrab.csv')
config_version="v5_20250621"
echo "Using ${csvs[$icpu]}"
python3 ./execute_missing_analysis.py --config_version=${config_version}  --csv=${csvs[$icpu]} --verbose=1





