import os
os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"
import sys
import glob
import argparse
import numpy as np

from cosmoprimo.fiducial import DESI, AbacusSummit
from desilike.profilers import MinuitProfiler
from desilike.samples.profiles import Profiles
from desilike.samplers.emcee import EmceeSampler
from desilike.samples import plotting, Chain
from desilike import setup_logging
setup_logging()

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX
from fit_cubic_tools import load_bins, load_covbox_cov, get_observable_likelihood

Z_VSMEAR = REDSHIFT_LSS_VSMEAR # REDSHIFT_VSMEAR or REDSHIFT_LSS_VSMEAR (LSS z bins)
Z_CUBIC = REDSHIFT_LSS_CUBICBOX # REDSHIFT_CUBICBOX or REDSHIFT_LSS_CUBICBOX

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    # parser.add_argument("--datDir", help="base directory for void data catalogs", default=None)
    # parser.add_argument("--outputDir", help="base directory for void random catalogs", default=None)
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['LRG','ELG','QSO'], default=['LRG','ELG','QSO'], nargs = '+')
    parser.add_argument("--sysmodels", help="spectroscopic systeamtics type to be fitted", type = str, choices=['standard', 'dv-obs'], default=['standard', 'dv-obs'], nargs = '+')
    args = parser.parse_args()

    task = 'BAOfit_recon_cubic_sys'
    # fit_cubic_post_recon: fit spectroscopic systeamtics contaminated mocks BAO after reconstruction

    bins_type = 'y3_bao' # test, y3_bao
    cov_type = 'covbox' # covbox, rescalC
    
    if 'recon' in task:
        recon  = True

    for tracer in args.tracers:
        for z_eff, (zmin, zmax) in zip(Z_CUBIC[tracer], Z_VSMEAR[tracer]):
            for sys_model in args.sysmodels:
                base_dir = '/pscratch/sd/s/shengyu/results/cubic'
                # base_dir = './results'
                data_args = {"tracer": tracer, "z_eff": z_eff, "zmin": zmin, "zmax": zmax, "recon": recon, "sys_model": sys_model, "grid_cosmo": "000"}
                fit_args = {"corr_type": 'xi', "bins_type": 'y3_sys', "cov_type": 'cov_fn', "recon": recon}
                chain_fn = base_dir+f'/BAO/chain_recon_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sys_model}.npy'
                if not os.path.exists(chain_fn):
                    print("sampling starts with data", data_args, flush=True)
                    (likelihood, _, _) = get_observable_likelihood(task, data_args, fit_args)
                    nwalkers = 64
                    interations = 9001 # save every 300 iterations
                    # MCMC sampling
                    sampler = EmceeSampler(likelihood, seed=42, nwalkers=nwalkers, save_fn = chain_fn)
                    sampler.run(check={'max_eigen_gr': 0.02}, max_iterations = interations) # save every 300 iterations