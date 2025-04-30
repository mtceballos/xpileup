import papermill as pm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Execute the jupyter notebook for flagging analysis',
            prog='execute_flagging.py')
    
    parser.add_argument('--singlesdatadir', required=False,
                        default="/dataj6/ceballos/INSTRUMEN/EURECA/TN350_detection/2024_revision/singles",
                        help='CSV file with data for missing analysis')
    parser.add_argument('--simEnergies', required=False,type=float,
                        nargs='*', default=[0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                        help="list of single simulated energies")
    parser.add_argument('--model', required=False,
                        default="crab", help="source model", choices=['crab','extend'])
    parser.add_argument('--flux_mcrab', required=True, type=float,
                        help="Source flux in mCrab")
    parser.add_argument('--filter', required=False, default="nofilter",
                        choices=['thinOpt','thickOpt','nofilt','thinBe','thickBe'],
                        help="Filter used in source file simulations")
    parser.add_argument('--focus', required=False, default="infoc", choices=['infoc','defoc'],
                        help="Focus option in simulations")
    parser.add_argument('--secondary_samples', required=False, default="1563", type=int,
                        help="Number of samples to previous pulse for a secondary classification")
    parser.add_argument('--nsigmas', required=False, default=5, type=float,
                        help="Number of std deviations to create the singles area contour")
    parser.add_argument('--poly_order', required=False, default=8,type=int,
                        help="Order of polynomial to create singles area contour")
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbose?', choices=[0,1])

    # read input arguments
    args = parser.parse_args()
    datadir=args.singlesdatadir
    simEnergies=args.simEnergies
    model=args.model
    flux_mcrab=args.flux_mcrab
    filter=args.filter
    focus=args.focus
    secondary_samples=args.secondary_samples
    nsigmas=args.nsigmas
    poly_order=args.poly_order
    verbose=args.verbose


    nbname=f'./output_ipynb/output_flagging_{flux_mcrab}.ipynb'

    params_dict = dict(
        datadir=datadir,
        simEnergies=simEnergies,
        secondary_samples = secondary_samples,
        verbose = verbose,
        nsigmas = nsigmas,
        poly_order = poly_order, 
        model = model,
        flux_mcrab = flux_mcrab,
        filter = filter,
        focus = focus)
    
    print(f"Running jupyter-notebook for flagging with {params_dict}")
    pm.execute_notebook(
        'flag_multipulse.ipynb', nbname,
        parameters = params_dict
    )
    print("=========================")
    print("Flagging analysis finished" )
    print("=========================")

    
