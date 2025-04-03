import papermill as pm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Execute the jupyter notebook for missing photons analysis',
            prog='execute_missing_analysis.py')
    
    parser.add_argument('--csv', required=True,
                        help='CSV file with data for missing analysis')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbose?', choices=[0,1])

    # read input arguments
    args = parser.parse_args()
    csv = args.csv
    verbose = args.verbose

    nbname=f'./output_ipynb/output_{csv}.ipynb'

    params_dict = dict(
        global_csv_file = csv,
        verbose = verbose
    )
    print(f"Running jupyter-notebook for simulations with {params_dict}")
    pm.execute_notebook(
        'missing_analysis.ipynb', nbname,
        parameters = params_dict
    )
    print("=========================")
    print("Missing analysis finished" )
    print("=========================")

    
