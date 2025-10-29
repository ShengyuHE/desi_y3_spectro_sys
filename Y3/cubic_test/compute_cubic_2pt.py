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
kedges   = np.arange(0.,0.4001,0.001); ells = (0, 2, 4)
smuedges  = (np.linspace(0., 200, 201), np.linspace(-1., 1., 201))
slogmuedges= (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201))
rplogedges = (np.geomspace(0.01, 100., 100), np.linspace(-1., 1., 201))

Z_VSMEAR = REDSHIFT_VSMEAR # REDSHIFT_VSMEAR or REDSHIFT_LSS_VSMEAR (LSS z bins)
Z_CUBIC = REDSHIFT_CUBICBOX # REDSHIFT_CUBICBOX or REDSHIFT_LSS_CUBICBOX

def statistics_2pt(data_positions, shifted_positions, fn, **args):
    boxsize = args.get('boxsize', 2000)
    los = args.get('los', 'z')
    recon = args.get('recon', False)
    # compute mps
    fn_mps = fn.format('xipoles')
    if not os.path.exists(fn_mps):
        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions, 
                                                    shifted_positions1 = shifted_positions, 
                                                    engine='corrfunc', 
                                                    boxsize=boxsize, los=los, position_type='xyz',
                                                    gpu=True, nthreads = 4)
                                                #  mpiroot=mpiroot, mpicomm=mpicomm)
        result_mps.save(fn_mps)
    else:
        result_mps = TwoPointCorrelationFunction.load(fn_mps)
    # compute pk
    fn_pk = fn.format('pkpoles')
    if not os.path.exists(fn_pk):
        result_pk = CatalogFFTPower(data_positions1=data_positions, 
                                    shifted_positions1 = shifted_positions, 
                                    edges=kedges, ells=ells, interlacing=3, 
                                    boxsize=boxsize, nmesh=512, resampler='tsc',los=los, position_type='xyz',)
                                    # mpiroot=mpiroot, mpicomm=mpicomm)
        result_pk.save(fn_pk)
    else:
        result_pk = CatalogFFTPower.load(fn_pk)
    # compute mps log scales
    # if recon == False:
    #     fn_mpslog = fn.format('mpslog')
    #     if not os.path.exists(fn_mpslog):
    #         result_mps = TwoPointCorrelationFunction('smu', slogmuedges, data_positions1=data_positions,
    #                                                 engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz',
    #                                                 gpu=True, nthreads = 4,)
    #                                                 #  mpiroot=mpiroot, mpicomm=mpicomm)
    #         result_mps.save(fn_mpslog)
    #     else:
    #         result_mps = TwoPointCorrelationFunction.load(fn_mpslog)
    # # compute projected correlation function wp
    # if recon == False:
    #     fn_wplog = fn.format('wplog')
    #     if not os.path.exists(fn_wplog):
    #         result_wp = TwoPointCorrelationFunction('rppi', rplogedges, data_positions1=data_positions, 
    #                                                 engine='corrfunc', boxsize=boxsize, los=los, position_type='xyz', nthreads = 4,)
    #         result_wp.save(fn_wplog)
    #     else:
    #         result_wp = TwoPointCorrelationFunction.load(fn_wplog)
    
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
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['BGS','LRG','ELG','QSO'], nargs = '+')

    args = parser.parse_args()
    Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']
    task = 'compute_Abacus_2pt' # Abacus, Covbox, EZmocks, QSO_test
    corr_types = ['pk'] # for EZmocks and covcubic
    recons = [False]
    if 'Abacus' in task or 'test' in task:
        dataset = 'Abacus'
        mockid = '0-24'
    elif 'Covbox' in task:
        dataset = 'Covbox'
        mockid = '3000-4500'
    elif 'EZmocks' in task:
        dataset = 'EZmocks'
        mockid = '1-2000'
    # Convert mockid string input to a list
    if '-' in mockid:
        start, end = map(int, mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, mockid.split(',')))
    for tracer in args.tracers:
        for z_cubic, (zmin, zmax) in zip(Z_CUBIC[tracer], Z_VSMEAR[tracer]):
            if dataset in ['Abacus']:
                boxsize = 2000.
                los     = 'z'
                for mock_id in mockids:
                    mock_id03 =  f"{mock_id:03}"
                    basedir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CubicBox/{tracer}/obs_z{zmin:.1f}-{zmax:.1f}/AbacusSummit_base_c000_ph{mock_id03}'
                    for sysmodel in ['standard', 'dv-obs']:
                        for recon in recons:
                            data_args = {'boxsize':2000., 'loc':'z', 'recon':recon, 'basedir':basedir}
                            if tracer == 'BGS':
                                data_fn = basedir+f'/catalog_rsd_{tracer}_z{zmin:.1f}-{zmax:.1f}.fits'
                            else:
                                data_fn = basedir+f'/catalog_rsd_xi2d_{tracer}_z{zmin:.1f}-{zmax:.1f}_velbias_B_s_mockcov.fits'
                            data = Catalog.read(data_fn)
                            if sysmodel == 'standard':
                                data_positions = np.array([data['x'],data['y'], data['z']%boxsize-boxsize/2.0])
                            elif sysmodel == 'dv-obs':
                                data_positions = np.array([data['x'],data['y'], data['z_dv']])  
                            if recon == False:
                                output_fn = basedir+f'/mpspk/{{}}_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sysmodel}.npy'
                                shifted_positions = None
                            elif recon == True: 
                                output_fn = basedir+f'/mpspk/{{}}_recon_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sysmodel}.npy'
                                if not os.path.exists(output_fn.format('xipoles')) or not os.path.exists(output_fn.format('pkpoles')):
                                    f, bias, s = GET_RECON_BIAS(tracer = tracer, grid_cosmo = '000')
                                    Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']
                                    recon = Reconstruction(f=f, bias=bias, los='z', cellsize=20., boxsize=2000., boxcenter = [0,0,0], position_type='xyz', dtype='f8')
                                    recon.assign_data(data_positions)
                                    recon.set_density_contrast(smoothing_radius=s)
                                    recon.run()
                                    data_positions = recon.read_shifted_positions(data_positions, field='disp+rsd' , dtype='f8')
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
                                else:
                                    shifted_positions = None
                            statistics_2pt(data_positions, shifted_positions, output_fn, **data_args)

            elif data in ['Covbox']:
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
                        print(f"File not found: {catalog_fn}. Skipping mock_id={mock_id04}", flush=True)
                        continue
                    fn_mps = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CovBox/{tracer}/z{z_cubic:.3f}/xi/recon/{{}}_{tracer}_z{z_cubic:.1f}_ph{mock_id04}.npy'.format('xipoles')
                    # compute correlation
                    if not os.path.exists(fn_mps):
                        data = Catalog.read(catalog_fn)
                        data_positions = np.array([data['x'],data['y'], data['z']])
                        data_positions = data_positions%boxsize-boxsize/2.0
                        f, bias, s = GET_RECON_BIAS(tracer = tracer, grid_cosmo = '000')
                        Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']
                        recon = Reconstruction(f=f, bias=bias, los='z', cellsize=20., boxsize=boxsize, boxcenter = [0,0,0], position_type='xyz', dtype='f8')
                        recon.assign_data(data_positions)
                        recon.set_density_contrast(smoothing_radius=s)
                        recon.run()
                        data_positions = recon.read_shifted_positions(data_positions, field='disp+rsd' , dtype='f8')
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

                        result_mps = TwoPointCorrelationFunction('smu', smuedges, data_positions1=data_positions, 
                                                                    shifted_positions1 = shifted_positions, 
                                                                    engine='corrfunc', 
                                                                    boxsize=boxsize, los=los, position_type='xyz',
                                                                    gpu=True, nthreads = 4)
                        result_mps.save(fn_mps)
                    # compute pk
                    # fn_pk = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CovBox/{tracer}/z{z_cubic:.3f}/pk/{{}}_{tracer}_z{z_cubic:.1f}_ph{mock_id04}.npy'.format('pkpoles')
                    # if not os.path.exists(fn_pk):
                    #     data = Catalog.read(catalog_fn)
                    #     data_positions = np.array([data['x'],data['y'], data['z']])
                    #     # compute mps
                    #     result_pk = CatalogFFTPower(data_positions1=data_positions, 
                    #                                 edges=kedges, ells=ells, interlacing=3, 
                    #                                 boxsize=boxsize, nmesh=512, resampler='tsc',los='z', position_type='xyz',)
                    #                                 # mpiroot=mpiroot, mpicomm=mpicomm)
                    #     result_pk.save(fn_pk)

            elif data in ['EZmocks']:
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
                        if corr_type == 'xi':
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
                        elif corr_type == 'pk':
                            output_fn = output_dir+f'/{{}}_{tracer}_z{zmin:.1f}-{zmax:.1f}_{mock_id04}.npy'
                            fn_pk = output_fn.format('pkpoles')
                            if not os.path.exists(fn_pk):
                                catalog_fn = base_dir+f'/{tracer}/z{z_cubic:.3f}/{mock_id04}/EZmock_{tracer}_z{z_cubic:.3f}_AbacusSummit_base_c000_ph000_{mock_id04}.0.fits.gz'
                                data = Catalog.read(catalog_fn, filetype='fits')
                                data_positions = read_real_space_positions(data, z_cubic)
                                result_pk = CatalogFFTPower(data_positions1=data_positions, 
                                                            edges=kedges, ells=ells, interlacing=3, 
                                                            boxsize=boxsize, nmesh=512, resampler='tsc',los='z', position_type='xyz',)
                                                            # mpiroot=mpiroot, mpicomm=mpicomm)
                                result_pk.save(fn_pk)

            elif data in ['QSO_test']:
                boxsize = 2000.
                los     = 'z'
                (zmin, zmax) = (0.8, 2.1)
                for mock_id in mockids:
                    mock_id03 =  f"{mock_id:03}"
                    basedir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CubicBox/{tracer}/obsQSO_z{zmin}-{zmax}/AbacusSummit_base_c000_ph{mock_id03}'
                    for sysmodel in ['standard', 'dv-obs']:
                        for recon in recons:
                            data_args = {'boxsize':2000., 'loc':'z', 'recon':recon, 'basedir':basedir}
                            data_fn = basedir+f'/catalog_rsd_xi2d_{tracer}_z{zmin}-{zmax}_velbias_B_s_mockcov.fits'
                            data = Catalog.read(data_fn)
                            if sysmodel == 'standard':
                                data_positions = np.array([data['x'],data['y'], data['z']%boxsize-boxsize/2.0])
                            elif sysmodel == 'dv-obs':
                                data_positions = np.array([data['x'],data['y'], data['z_dv']])
                            if recon == False:
                                output_fn = basedir+f'/mpspk/{{}}_{tracer}_z{zmin:.1f}-{zmax:.1f}_{sysmodel}.npy'
                                shifted_positions = None
                            statistics_2pt(data_positions, shifted_positions, output_fn, **data_args)