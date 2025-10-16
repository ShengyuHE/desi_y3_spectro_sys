import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from cosmoprimo.fiducial import DESI, AbacusSummit
from mockfactory import utils, DistanceToRedshift, Catalog, RandomBoxCatalog
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction
from pypower import CatalogFFTPower,mpi, setup_logging
from pycorr import TwoPointCorrelationFunction, setup_logging
setup_logging()

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX, Y3_SMOOTHING
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX, REDSHIFT_LSS_EZMOCKS
from helper import GET_RECON_BIAS
# from mpi4py import MPI
# mpicomm = mpi.COMM_WORLD
# mpiroot = 0

# basic settings
kedges   = np.arange(0.,0.4001,0.001); ells = (0, 2)
smuedges  = (np.linspace(0., 200, 201), np.linspace(-1., 1., 201))
slogmuedges= (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201))
rplogedges = (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201))

Z_VSMEAR = REDSHIFT_LSS_VSMEAR # REDSHIFT_VSMEAR or REDSHIFT_LSS_VSMEAR (LSS z bins)
Z_CUBIC = REDSHIFT_LSS_EZMOCKS # REDSHIFT_CUBICBOX or REDSHIFT_LSS_CUBICBO or REDSHIFT_LSS_EZMOCKS

def read_real_space_positions(data, redshift, cosmo = DESI()):
    Hz = cosmo.H0*cosmo.efunc(redshift)
    x = data['x']
    y = data['y']
    z = data['z']+data['vz']*(1+redshift)/Hz
    data_positions = np.array([x,y,z])
    data_positions = data_positions%boxsize-boxsize/2.0
    return data_positions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    # parser.add_argument("--datDir", help="base directory for void data catalogs", default=None)
    # parser.add_argument("--outputDir", help="base directory for void random catalogs", default=None)
    # parser.add_argument("--mockid", type=str, default="0-24", help="Mock ID range or list")
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['LRG','ELG','QSO'], default=['LRG','ELG','QSO'], nargs = '+')
    args = parser.parse_args()
    Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']
    task = 'compute_EZmocks' # compute_abacus, compute_covbox, compute_EZmocks
    corr_types = ['xi'] # for EZmocks and covcubic
    recons = [True]
    if task in ['compute_abacus']:
        mockid = '0-24'
    elif task in ['compute_covbox']:
        mockid = '3000-4500'
    elif task in ['compute_EZmocks']:
        mockid = '1-2000'
    # Convert mockid string input to a list
    if '-' in mockid:
        start, end = map(int, mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, mockid.split(',')))
    for tracer in args.tracers:
        for z_cubic, (zmin, zmax) in zip(Z_CUBIC[tracer], Z_VSMEAR[tracer]):
            for mock_id in mockids:
                mock_id04 =  f"{mock_id:04}"
                if tracer in ['BGS']:
                    boxsize  = 500.
                    base_dir = '/global/cfs/cdirs/desicollab/cosmosim/SecondGenMocks/EZmock/CubicBox_2Gpc' 
                if tracer in ['LRG','ELG','QSO']:
                    boxsize  = 1500.
                    base_dir = '/global/cfs/cdirs/desicollab/cosmosim/SecondGenMocks/EZmock/CubicBox_6Gpc'
                print(tracer, z_cubic, (zmin, zmax), mock_id04, recons, flush=True)
                for corr_type in corr_types:
                    output_dir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/EZmocks/{tracer}/z{z_cubic:.3f}/{corr_type}'
                    for recon in recons:
                        if recon == True:
                            output_fn = output_dir+f'/recon/{{}}_{tracer}_z{zmin:.1f}-{zmax:.1f}_{mock_id04}.npy'
                        elif recon == False:
                            output_fn = output_dir+f'/{{}}_{tracer}_z{zmin:.1f}-{zmax:.1f}_{mock_id04}.npy'
                        fn_mps = output_fn.format('xipoles')
                        if not os.path.exists(fn_mps):
                            catalog_fn = base_dir+f'/{tracer}/z{z_cubic:.3f}/{mock_id04}/EZmock_{tracer}_z{z_cubic:.3f}_AbacusSummit_base_c000_ph000_{mock_id04}.0.fits.gz'
                            data = Catalog.read(catalog_fn, filetype='fits')
                            data_positions = read_real_space_positions(data, z_cubic)
                            if recon == True:
                                f, bias, s = GET_RECON_BIAS(tracer = tracer, grid_cosmo = '000')
                                Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']
                                recon = Reconstruction(f=f, bias=bias, los='z', cellsize=20., boxsize=boxsize, boxcenter=[0.0, 0.0, 0.0], position_type='xyz', dtype='f8')
                                recon.assign_data(data_positions)
                                recon.set_density_contrast(smoothing_radius=s)
                                recon.run()
                                data_positions = recon.read_shifted_positions(data_positions, field='disp+rsd', dtype='f8')
                                data_positions = data_positions%boxsize-boxsize/2.0
                                shifted_positions = []
                                for i in range(4):
                                    randoms = RandomBoxCatalog(boxsize=boxsize, csize=data.csize)
                                    randoms_positions = randoms.get('Position').T
                                    del randoms['Position']
                                    randoms_positions_rec = recon.read_shifted_positions(randoms_positions, field='disp+rsd', dtype='f8')
                                    shifted_positions.append(randoms_positions_rec)
                                shifted_positions = np.concatenate(shifted_positions, axis=1)
                                shifted_positions = shifted_positions%boxsize-boxsize/2.0
                            elif recon == False:
                                shifted_positions = None
                            result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions, 
                                                                        shifted_positions1 = shifted_positions, 
                                                                        engine='corrfunc', boxsize=boxsize, 
                                                                        los='z', position_type='xyz',
                                                                        gpu=True, nthreads = 4)
                            result_mps.save(fn_mps)