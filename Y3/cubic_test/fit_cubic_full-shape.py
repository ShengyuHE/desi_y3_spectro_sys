import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import argparse
import numpy as np

# from mockfactory import Catalog
from cosmoprimo.fiducial import DESI, AbacusSummit
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.profilers import MinuitProfiler
from desilike.samplers.emcee import EmceeSampler
from desilike import setup_logging
setup_logging()  # for logging messages

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX, EDGES, GET_RECON_BIAS
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX
from fit_cubic_tools import get_observable_likelihood

Z_VSMEAR = REDSHIFT_LSS_VSMEAR # REDSHIFT_VSMEAR or REDSHIFT_LSS_VSMEAR (LSS z bins)
Z_CUBIC = REDSHIFT_LSS_CUBICBOX # REDSHIFT_CUBICBOX or REDSHIFT_LSS_CUBICBOX

def traning_emulators(task, data_args, fit_args):
    (likelihood, theory, tempalte) = get_observable_likelihood(task, data_args, fit_args)
    emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=3, method='finite')) # Taylor expansion, up to a given order
    emulator.set_samples() # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)
    emulator.fit()
    return emulator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    # parser.add_argument("--datDir", help="base directory for void data catalogs", default=None)
    # parser.add_argument("--outputDir", help="base directory for void random catalogs", default=None)
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['LRG','ELG','QSO'], default=['LRG','ELG','QSO'], nargs = '+')
    parser.add_argument("--sysmodels", help="spectroscopic systeamtics type to be fitted", type = str, choices=['standard', 'dv-obs'], default=['standard', 'dv-obs'], nargs = '+')
    args = parser.parse_args()

    fit = 'SF' # FM: Full-Modeling, SF: Shape-Fit
    task = f'{fit}fit_cubic_sys_LCDM'
    theory_model = 'vel'
    fit_cosmo = 'LCDM'
    bins_type = 'y3_sys'
    cov_type  = 'EZcov_fn_rsf' # EZcov_fn, EZcov_fn_rsf
    option = 'None' # _5.0logA, _4.5logA, _fix_ns, _dv_obs, None

    for tracer in args.tracers:
        for z_eff, (zmin, zmax) in zip(Z_CUBIC[tracer], Z_VSMEAR[tracer]):
            for sys_model in args.sysmodels:
                result_dir = '/pscratch/sd/s/shengyu/results/cubic'
                if 'rsf' in cov_type:
                    emulator_fn = result_dir+f'/emulator/emulator_{fit}_{tracer}_z{zmin}-{zmax}_{theory_model}_rscov.npy'
                    chain_fn = result_dir+f'/{fit}/{fit_cosmo}/chain_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}_{theory_model}_rscov.npy'
                    if option != 'None':
                        emulator_fn = result_dir+f'/emulator/extra/emulator_{fit}_{tracer}_z{zmin}-{zmax}_{theory_model}_rscov{option}.npy'
                        chain_fn = result_dir+f'/{fit}/{fit_cosmo}/extra/chain_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}_{theory_model}_rscov{option}.npy' 
                else:
                    emulator_fn = result_dir+f'/emulator/emulator_{fit}_{tracer}_z{zmin}-{zmax}_{theory_model}.npy'
                    chain_fn = result_dir+f'/{fit}/{fit_cosmo}/chain_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}_{theory_model}.npy'
                fit_args = {"corr_type": 'pk', "bins_type": bins_type, "cov_type": cov_type,  "theory_model": theory_model, "emulator_fn": emulator_fn}
                data_args = {"tracer": tracer, "z_eff": z_eff, "zmin": zmin, "zmax": zmax, "sys_model": sys_model, "grid_cosmo": "000",}
                if not os.path.exists(chain_fn):
                    print("sampling starts with data", data_args, flush=True)
                    (likelihood, _, _) = get_observable_likelihood(task, data_args, fit_args)
                    # MCMC sampling
                    interations = 60001 # save every 300 iterations
                    nwalkers= 64
                    sampler = EmceeSampler(likelihood, seed=41, nwalkers=nwalkers, save_fn = chain_fn)
                    sampler.run(check={'max_eigen_gr': 0.03, 'min_ess': 300}, max_iterations = interations) # save every 300 iterations