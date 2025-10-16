import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from mockfactory import utils, DistanceToRedshift, Catalog, RandomBoxCatalog

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX, Y3_SMOOTHING
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX
from triumvirate.parameters import fetch_paramset_template, ParameterSet
from triumvirate.catalogue import ParticleCatalogue
from triumvirate.threept import compute_bispec_in_gpp_box, compute_3pcf_in_gpp_box

Z_VSMEAR = REDSHIFT_LSS_VSMEAR # REDSHIFT_VSMEAR or REDSHIFT_LSS_VSMEAR (LSS z bins)
Z_CUBIC = REDSHIFT_LSS_CUBICBOX # REDSHIFT_CUBICBOX or REDSHIFT_LSS_CUBICBO

def compute_bk(pos1, fn, mode=(0,0,0), edges=(0.0025, 0.3025, 60), ngrid = 512, **args):
    """
    Compute the bispectrum for a periodic box.
    Use triumvirate codes and the basic idea is Fourrier Transform from the bispectrum.
    Parameters
    ----------
    pos1 : array-like
        Positions of sample 1 (e.g., galaxies or halos).
    fn   ; 
    edges : list of arrays
        Bin edges in separation (smin, smax, nbins).
    boxsize : float
        Size of the simulation box.
    mode : tuple of int, optional
        Multipoles to project onto. Default is (0, 0, 0).
    ngrid: 
        Number of grid points along each axis for the Fast Fourier Transform (FFT). Default is 512.
    Returns
    -------
    result_threept : dict
        Dictionary containing the computed 3PCF and related quantities.
    """
    boxsize = args.get('boxsize', 2000)
    los = args.get('los', 'z')
    basedir = args.get('basedir', './test')
    (kmin, kmax, nbins) = edges
    (d_ell1, d_ell2, d_ELL) = mode
    mode_fn = f'{d_ell1}{d_ell2}{d_ELL}'
    fn_bk = fn.format(f'bk{mode_fn}_diag')
    if not os.path.exists(fn_bk):
        bispec_dict = fetch_paramset_template('dict')
        bispec_dict.update({
            'degrees': {'ell1': d_ell1, 'ell2': d_ell2, 'ELL': d_ELL},
            'boxsize': {'x': boxsize, 'y': boxsize, 'z': boxsize},
            'statistic_type': 'bk',
            'ngrid': {'x': ngrid, 'y': ngrid, 'z': ngrid},
            'num_bins': nbins,
            'range': [kmin, kmax],
        })
        bispec_param = ParameterSet(param_dict=bispec_dict)
        bispec_param.update(directories={'measurements':basedir+'/mpspk/'})
        CATALOGUE = ParticleCatalogue(pos1[0], pos1[1], pos1[2],  nz=len(pos1[0])/boxsize**3)
        result_bk = compute_bispec_in_gpp_box(CATALOGUE,paramset=bispec_param, save='.npz')
        fn_bk_temp = basedir+ f'/mpspk/bk{mode_fn}_diag.npz'
        os.rename(fn_bk_temp, fn_bk)
        print('save to', fn_bk, flush = True)
        return result_bk
    else:
        print(f'{fn_bk} exsits',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    # parser.add_argument("--datDir", help="base directory for void data catalogs", default=None)
    # parser.add_argument("--outputDir", help="base directory for void random catalogs", default=None)
    # parser.add_argument("--mockid", type=str, default="0-24", help="Mock ID range or list")
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['BGS'], nargs = '+')
    parser.add_argument("--mode", help="bispectrum mode to be selected", type = str, choices=["000", "200", "202"], default="000")

    args = parser.parse_args()
    task = 'compute_data_3pt' # compute_cov, compute_data_2pt, compute_data_3pt
    if task in ['compute_data_3pt']:
        mockid = '0-24'
    elif task in ['compute_cov']:
        mockid = '3000-4500'
    mode_map = {"000": (0, 0, 0),"200": (2, 0, 0),"202": (2, 0, 2)}
    mode_tuple = mode_map[args.mode]

    # Convert mockid string input to a list
    if '-' in mockid:
        start, end = map(int, mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, mockid.split(',')))
    for tracer in args.tracers:
        for z_cubic, (zmin, zmax) in zip(Z_CUBIC[tracer], Z_VSMEAR[tracer]):
            if task in ['compute_data_3pt']:
                boxsize = 2000.
                los     = 'z'
                for mock_id in mockids:
                    mock_id03 =  f"{mock_id:03}"
                    basedir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CubicBox/{tracer}/obs_z{zmin:.1f}-{zmax:.1f}/AbacusSummit_base_c000_ph{mock_id03}'
                    for sysmodel in ['standard', 'dv-obs']:
                        # print(tracer, zmin, zmax, mock_id03, sysmodel,flush=True)
                        data_args = {'boxsize':2000., 'loc':'z', 'basedir':basedir}
                        if tracer == 'BGS':
                            data_fn = basedir+f'/catalog_rsd_{tracer}_z{zmin:.1f}-{zmax:.1f}.fits'
                        else:
                            data_fn = basedir+f'/catalog_rsd_xi2d_{tracer}_z{zmin:.1f}-{zmax:.1f}_velbias_B_s_mockcov.fits'
                        data = Catalog.read(data_fn)
                        if sysmodel == 'standard':
                            data_positions = np.array([data['x'],data['y'], data['z']%boxsize-boxsize/2.0])
                        elif sysmodel == 'dv-obs':
                            data_positions = np.array([data['x'],data['y'], data['z_dv']])
                        data_positions = np.array(data_positions, dtype=np.float64)
                        output_fn = basedir+f'/mpspk/{{}}_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sysmodel}.npz'
                        if task == 'compute_data_3pt':
                            compute_bk(data_positions, output_fn, mode=mode_tuple, edges=(0.0025, 0.3025, 60), **data_args)

            elif task in ['compute_cov']:
                boxsize = 500.
                los     = 'z'
                for mock_id in mockids:
                    mock_id04 =  f"{mock_id:04}"
                    cov_dir = f'/global/cfs/cdirs/desi/cosmosim/AbacusHOD_mocks/v1/CovBox/{tracer}/z{z_cubic:.3f}'
                    if tracer == 'LRG':
                        catalog_fn = cov_dir + f'/catalog_rsd_xi2d_{tracer.lower()}_main_z{z_cubic:.1f}_velbias_mockcov_ph{mock_id04}.fits'
                    elif tracer == 'ELG':
                        catalog_fn = cov_dir + f'/catalog_rsd_xi2d_{tracer.lower()}_z{z_cubic:.1f}_velbias_B_mockcov_ph{mock_id04}.fits'
                    elif tracer == 'QSO':
                        catalog_fn = cov_dir + f'/catalog_rsd_xi2d_{tracer.lower()}_z{z_cubic:.1f}_velbias_B_mockcov_zerr_skip5_ph{mock_id04}.fits'
                    if not os.path.isfile(catalog_fn):
                        print(f"File not found: {catalog_fn}. Skipping mock_id={mock_id04}",flush=True)
                        continue
                    # compute bispectrum
                    basedir_bk = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CovBox/{tracer}/z{z_cubic:.3f}/bk'
                    fn_bk = basedir_bk+f'/{{}}_{tracer}_z{z_cubic:.1f}_ph{mock_id04}.npy'.format('bk000_diag')
                    if not os.path.exists(fn_bk):
                        CATALOGUE = ParticleCatalogue(data_positions[0],data_positions[1],data_positions[2], nz=len(data_positions[0])/boxsize**3)
                        bispec_dict = fetch_paramset_template('dict')
                        bispec_dict.update({
                            'degrees': {'ELL': 0, 'ell1': 0, 'ell2': 0},
                            'boxsize': {'x': boxsize, 'y': boxsize, 'z': boxsize},
                            'statistic_type': 'bispec',
                            'ngrid': {'x': 512, 'y': 512, 'z': 512},
                            'num_bins': 10,
                            'range': [0.005, 0.205],
                        })
                        bispec_param = ParameterSet(param_dict=bispec_dict)
                        bispec_param.update(directories={'measurements':basedir_bk+'/'})
                        result_bispec = compute_bispec_in_gpp_box(CATALOGUE,paramset=bispec_param, save='.npz')
                        fn_bk_temp = basedir_bk+'/bk000_diag.npz'
                        os.rename(fn_bk_temp, fn_bk)
                    # compute 3-point correlation function
                    basedir_3pcf = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CovBox/{tracer}/z{z_cubic:.3f}/zeta'
                    fn_3pcf = f'/{{}}_{tracer}_z{z_cubic:.1f}_ph{mock_id04}.npy'.format('zeta000_diag')
                    if not os.path.exists(fn_3pcf):
                        CATALOGUE = ParticleCatalogue(data_positions[0],data_positions[1],data_positions[2], nz=len(data_positions[0])/boxsize**3)
                        threept_dict = fetch_paramset_template('dict')
                        threept_dict.update({
                            'degrees': {'ELL': 0, 'ell1': 0, 'ell2': 0},
                            'boxsize': {'x': boxsize, 'y': boxsize, 'z': boxsize},
                            'statistic_type': '3pcf',
                            'ngrid': {'x': 512, 'y': 512, 'z': 512},
                            'num_bins': 10,
                            'range': [0,50],
                        })
                        threept_param = ParameterSet(param_dict=threept_dict)
                        threept_param.update(directories={'measurements':basedir_3pcf+'/'})
                        result_threept = compute_3pcf_in_gpp_box(CATALOGUE,paramset=threept_param, save='.npz')
                        fn_3pcf_temp = basedir_3pcf+'/zeta000_diag.npz'
                        os.rename(fn_3pcf_temp, fn_3pcf)