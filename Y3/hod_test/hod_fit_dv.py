
import os
import sys
import yaml
import argparse
from HODDIES import HOD
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_ABACUS_Y3

Z_VSMEAR = REDSHIFT_VSMEAR 
Z_CUBIC = REDSHIFT_ABACUS_Y3 

result_dir = '/pscratch/sd/s/shengyu/galaxies/hod/fit/results'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zind", help="index of redshift", type = int, choices=[0,1,2,3], default=0)
    parser.add_argument("--tracer", help="tracer type to be selected", type = str, choices=['LRG','ELG','QSO'], default='QSO')
    parser.add_argument("--systematics", help="systematic type to be selected", type = str, choices=['standard','dv-obs'], default=['standard','dv-obs'], nargs = '+')
    parser.add_argument("--fitnum", help="number of fitting", type = int, default=20)

    args = parser.parse_args()
    fit = 'minimizer'

    tracer = args.tracer
    z_simu = Z_CUBIC[tracer][args.zind]
    (zmin, zmax) = Z_VSMEAR[tracer][args.zind]

    # for z_simu,(zmin, zmax) in zip(Z_CUBIC[tracer],Z_VSMEAR[tracer]):
    for fit_ind in range(args.fitnum):
        seed = fit_ind+123
        for sys_model in args.systematics:
            args_hod = yaml.load(open(f'./parameter_files/{tracer}_HOD_fit_param.yaml', 'r'), Loader=yaml.FullLoader)
            args_hod['tracers'] = [tracer]
            args_hod['hcat']['Abacus']['z_simu'] = z_simu
            args_hod['fit_param']['zmin'] = zmin
            args_hod['fit_param']['zmax'] = zmax
            if sys_model == 'standard':
                args_hod['fit_param']['use_vsmear'] = False
            elif sys_model == 'dv-obs':
                args_hod['fit_param']['use_vsmear'] = True
            if 'minimizer' in  fit:
                fn_minimiser = result_dir+f'/minimiser/{tracer}/fitmini_{tracer}_z{zmin}-{zmax}_{sys_model}_{fit_ind:02}.npy'
                print(fn_minimiser, flush=True)
                if not os.path.exists(fn_minimiser):
                    print(f'[START HOD FITTING]: {fn_minimiser}')
                    HOD_obj= HOD(args = args_hod, path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')
                    options = {"maxiter":40, "popsize": 500, 'xtol':1e-6, 'workers':mpi_comm.Get_size(),  'backend':'mpi'}
                    result = HOD_obj.run_minimizer(minimizer_options= options, seed=seed, mpi_comm=mpi_comm, save_fn = fn_minimiser)
                    if mpi_rank==0:
                        save_fn_fig = result_dir+ f'/minimiser/{tracer}/plots/hodbf_{tracer}_z{zmin}-{zmax}_{sys_model}_{fit_ind:02}.png'
                        fig = HOD_obj.plot_bf_data(save=save_fn_fig)