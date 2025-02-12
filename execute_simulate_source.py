import papermill as pm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Execute the jupyter notebook for Crab spectral simulation',
            prog='execute_simulate_crab.py')
    
    parser.add_argument('--sim_number', default=1, type=int,
                        help='simulation number')
    parser.add_argument('--flux_mcrab', required=True,
                        help='Source flux in mCrab units', type=float)
    parser.add_argument('--Emin', default=2.0, type=float,
                        help='Minimum of the flux energy range in keV')
    parser.add_argument('--Emax', default=10.0, type=float,
                        help='Maximum of the flux energy range in keV')
    parser.add_argument('--model', required=True,
                        help='Model to be used for the simulation',
                        choices=['crab', 'model1', 'model2', 'model3'])
    parser.add_argument('--filter', default='nofilt',
                        help='Filter to be used for the simulation, if any',
                        choices=['thinOpt', 'thickOpt', 'nofilt', 'thinBe', 'thickBe'])
    parser.add_argument('--focus', default='',
                        help='Focus to be used for the simulation. If not provided, option is selected automatically from the input flux',
                        choices=['infoc', 'defoc',''])
    parser.add_argument('--recons', default=0, type=int,
                        help='Do reconstruction?', choices=[0,1])
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbose?', choices=[0,1])

    # read input arguments
    args = parser.parse_args()
    sim_number = args.sim_number
    flux_mcrab = args.flux_mcrab
    Emin = args.Emin
    Emax = args.Emax
    model = args.model
    filter = args.filter
    focus = args.focus
    recons = args.recons
    verbose = args.verbose

    nbname=f'./output_ipynb/output_{model}_{flux_mcrab}_{Emin}_{Emax}_{filter}_{focus}_sim{sim_number}.ipynb'

    params_dict = dict(
        sim_number = sim_number,
        flux_mcrab = flux_mcrab,
        Emin = Emin,
        Emax = Emax,
        model = model,
        filter = filter,
        focus = focus,
        recons = recons,
        verbose = verbose
    )
    print(f"Running jupyter-notebook for simulations with {params_dict}")
    pm.execute_notebook(
        'simulate_source.ipynb', nbname,
        parameters = params_dict
    )
    print("=====================")
    print("Simulations finished" )
    print("=====================")

    
