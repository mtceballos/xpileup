import papermill as pm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Execute the jupyter notebook for stream Crab spectral simulation',
            prog='execute_simulate_CrabSpec_stream.py')
    
    parser.add_argument('--flux_mcrab', required=True,
                        help='Source flux in mCrab units', type=float)
    parser.add_argument('--exposure', default=1.,
                        help='exposure time', type=float)
    parser.add_argument('--nonxbgd', default="no",
                        help='Simulate non-Xray background?')
    parser.add_argument('--XTalk', default="none",
                        help='Type of XTalk to be simulated')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbose?', choices=[0,1])

    # read input arguments
    args = parser.parse_args()
    flux_mcrab = args.flux_mcrab
    exposure = args.exposure
    nonxbgd = args.nonxbgd
    XTalk = args.XTalk
    verbose = args.verbose

    nbname=f'./output_ipynb/output_Crab_{flux_mcrab}_{exposure}s_{nonxbgd}_{XTalk}.ipynb'

    params_dict = dict(
        flux_mcrab = flux_mcrab,
        exposure = exposure,
        nonxbgd = nonxbgd,
        XTalk = XTalk,
        verbose = verbose
    )
    print(f"Running jupyter-notebook for simulations with {params_dict}")
    pm.execute_notebook(
        'simulate_CrabSpec_stream.ipynb', nbname,
        parameters = params_dict
    )
    print("=====================")
    print("Simulations finished" )
    print("=====================")

    
