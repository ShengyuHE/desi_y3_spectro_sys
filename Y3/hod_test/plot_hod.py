# Initiate the HOD instance with parameter file test_fit_param.yaml
import sys
sys.path.insert(0, '/global/homes/s/shengyu/.local/perlmutter/python-3.11/lib/python3.11/site-packages')
import copy
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from HODDIES import HOD
sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_ABACUS_Y3

# result_dir = '/global/homes/s/shengyu/project_rc/main/Y3/hod_test/results'
Z_VSMEAR = REDSHIFT_VSMEAR 
Z_CUBIC = REDSHIFT_ABACUS_Y3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zind", help="index of redshift", type = int, choices=[0,1,2], default=0)
    parser.add_argument("--tracer", help="tracer type to be selected", type = str, choices=['LRG','ELG','QSO'], default='LRG')
    parser.add_argument("--systematics", help="systematic type to be selected", type = str, choices=['standard','dv-obs'], default=['standard','dv-obs'], nargs = '+')
    parser.add_argument("--fitind", help="index of fitting", type = int, default=0)
    args = parser.parse_args()

    fit = 'minimizer'
    tracer = args.tracer
    result_dir = f'/pscratch/sd/s/shengyu/galaxies/hod/fit/results/minimiser/{tracer}'

    fit_ind = args.fitind
    tracer = args.tracer
    z_simu = Z_CUBIC[tracer][args.zind]
    (zmin, zmax) = Z_VSMEAR[tracer][args.zind]

    seed = fit_ind+123
    #Plot best fit from a precomputed bes fit run 
    for sys_model in args.systematics:
        args_hod = yaml.load(open(f'./parameter_files/{tracer}_HOD_fit_param.yaml', 'r'), Loader=yaml.FullLoader)
        args_hod['tracers'] = [tracer]
        args_hod['hcat']['Abacus']['z_simu'] = z_simu
        args_hod['fit_param']['zmin'] = zmin
        args_hod['fit_param']['zmax'] = zmax
        args_hod['fit_param']['zmax'] = zmax
        if sys_model == 'standard':
            args_hod['fit_param']['use_vsmear'] = False
        elif sys_model == 'dv-obs':
            args_hod['fit_param']['use_vsmear'] = True
        args_copy = copy.deepcopy(args_hod)
        HOD_obj = HOD(args=args_copy, path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')
        HOD_obj.initialize_fit(fix_seed=seed)
        bf_file_fn = result_dir+f'/fitmini_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}_{fit_ind:02}.npy'
        save_fn_fig = result_dir+ f'/plots/hodbf_{tracer}_z{zmin}-{zmax}_{sys_model}_{fit_ind:02}.png'
        HOD_obj.plot_bf_data(bf_file = bf_file_fn, save=save_fn_fig)