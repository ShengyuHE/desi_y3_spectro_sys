import os
import sys
import fitsio
import logging
import argparse
import numpy as np
import mpytools as mpy
from astropy.table import Table

from cosmoprimo.fiducial import DESI, AbacusSummit
from mockfactory import utils, DistanceToRedshift, Catalog, RandomBoxCatalog
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction, utils, setup_logging

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, REDSHIFT_CUBICBOX, EDGES, GET_RECON_BIAS
from helper import REDSHIFT_LSS_VSMEAR, REDSHIFT_LSS_CUBICBOX
from Y3_redshift_systematics import vsmear, vsmear_modelling

Z_VSMEAR = REDSHIFT_LSS_VSMEAR # REDSHIFT_VSMEAR or REDSHIFT_LSS_VSMEAR (LSS z bins)
Z_CUBIC = REDSHIFT_LSS_CUBICBOX # REDSHIFT_CUBICBOX or REDSHIFT_LSS_CUBICBOX

boxsize = 2000.
cosmo = DESI()

def add_sys_spec_to_cat(catalog, tracer, zmin, zmax):
    catalog['z_dv'] = catalog['z'].copy()
    catalog['vz_dv'] = catalog['vz'].copy()
    dv = vsmear(tracer, zmin=zmin, zmax=zmax, Ngal = len(catalog), seed=z_index*10+mock_id*100+1234)
    catalog['vz_dv'] += dv
    zmid = (zmin+zmax)/2
    Hz = cosmo.H0*cosmo.efunc(zmid)
    catalog['z_dv'] += dv*(1+zmid)/Hz
    catalog['z_dv'] = catalog['z_dv']%boxsize-boxsize/2.0

def run_reconstruction(Reconstruction, positions, tracer, boxsize=None, nmesh=None, cellsize=20., smoothing_radius=None, convention='recsym', dtype='f8'):
    # RecSym = remove large scale RSD from randoms
    f, bias, s = GET_RECON_BIAS(tracer = tracer, grid_cosmo = '000')
    if smoothing_radius == None: smoothing_radius = s
    print(f"RECONTRUCTION: recon_sm{smoothing_radius:.0f}_IFFT_{convention}" )
    recon = Reconstruction(f=f, bias=bias, boxsize=boxsize, nmesh=nmesh, cellsize=cellsize, los='z', positions=positions, boxpad=1.2, position_type = 'xyz', dtype=dtype)
    recon.assign_data(positions)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    recon.run()
    field = 'rsd' if convention == 'rsd' else 'disp+rsd'
    field = 'disp+rsd' if convention == 'recsym' else 'disp'
    if Reconstruction is IterativeFFTParticleReconstruction:
        positions_rec = recon.read_shifted_positions('data', field=field, dtype=dtype)
    else:
        positions_rec = recon.read_shifted_positions(positions, field=field, dtype=dtype)
    positions_rec = positions_rec%boxsize-boxsize/2.0
    return positions_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nthreads", type = int, default = 4)
    # parser.add_argument("--datDir", help="base directory for void data catalogs", default=None)
    # parser.add_argument("--outputDir", help="base directory for void random catalogs", default=None)
    parser.add_argument("--tracers", help="tracer type to be selected", type = str, choices=['BGS','LRG','ELG','QSO'], default=['LRG'], nargs = '+')
    parser.add_argument("--mockid", type=str, default="0-24", help="Mock ID range or list")
    parser.add_argument("--return_ran", default=True, help="generate random catalogs")
    parser.add_argument("--recon", default=False, help="reconstruction")
    args = parser.parse_args()
    
    option = 'QSO_test'
    recon = False
    Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']

    # Convert mockid string input to a list
    if '-' in args.mockid:
        start, end = map(int, args.mockid.split('-'))
        mockids = list(range(start, end + 1))
    else:
        mockids = list(map(int, args.mockid.split(',')))
    if option == 'standard': 
        for tracer in args.tracers:
            for z_index,(zmin, zmax) in enumerate(Z_VSMEAR[tracer]):
                z_cubic = Z_CUBIC[tracer][z_index]
                for mock_id in mockids:
                    mock_id03 =  f"{mock_id:03}"
                    print('[BUILDING MOCKS]',tracer, z_cubic, (zmin, zmax),  mock_id03, flush=True)
                    z_dir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CubicBox/{tracer}'
                    if tracer == 'BGS':
                        catalog_fn = z_dir+f'/z{z_cubic:.3f}/AbacusSummit_base_c000_ph{mock_id03}/BGS_box_ph{mock_id03}.fits'
                    if tracer == 'LRG':
                        catalog_fn = z_dir+f'/z{z_cubic:.3f}/AbacusSummit_base_c000_ph{mock_id03}/catalog_rsd_xi2d_lrg_main_z{z_cubic:.1f}_velbias_B_s_mockcov.fits'
                    if tracer == 'ELG':
                        catalog_fn = z_dir+f'/z{z_cubic:.3f}/AbacusSummit_base_c000_ph{mock_id03}/catalog_rsd_xi2d_elg_z{z_cubic:.1f}_velbias_B_s_mockcov.fits'
                    if tracer == 'QSO':
                        catalog_fn = z_dir+f'/z{z_cubic:.3f}/AbacusSummit_base_c000_ph{mock_id03}/catalog_rsd_xi2d_qso_z{z_cubic:.1f}_velbias_B_s_mockcov_zerr_skip5.fits'
                    catalog = Catalog.read(catalog_fn, filetype='fits')
                    if recon == False:
                        dir = os.path.join(z_dir, f'obs_z{zmin:.1f}-{zmax:.1f}/AbacusSummit_base_c000_ph{mock_id03}')
                        if tracer == 'BGS':
                            save_fn = dir+f'/catalog_rsd_{tracer}_z{zmin:.1f}-{zmax:.1f}.fits'
                        else:
                            save_fn = dir+f'/catalog_rsd_xi2d_{tracer}_z{zmin:.1f}-{zmax:.1f}_velbias_B_s_mockcov.fits'
                        if not os.path.exists(save_fn):
                            if tracer == 'BGS':
                                # DESI BGS selection: R_MAG_ABS<-21.5
                                catalog = catalog[catalog['R_MAG_ABS'] < -21.5]
                                # in redshift space, assuming los = 'z'
                                zmid = (zmin+zmax)/2
                                Hz = cosmo.H0*cosmo.efunc(zmid)
                                catalog['z'] = catalog['z']+catalog['vz']*(1+zmid)/Hz
                                # move the box to the center
                                catalog['x'] = catalog['x']%boxsize-boxsize/2.0
                                catalog['y'] = catalog['y']%boxsize-boxsize/2.0
                                catalog['z'] = catalog['z']%boxsize-boxsize/2.0
                            add_sys_spec_to_cat(catalog, tracer, zmin, zmax)
                            catalog.write(save_fn)
                            print(f'[SAVE TO] {save_fn}', flush=True)
                    '''
                    elif recon == True:
                        dir = os.path.join(z_dir, f'obs_z{zmin:.1f}-{zmax:.1f}/AbacusSummit_base_c000_ph{mock_id03}/recon')
                        for sysmodel in ['standard', 'dv-obs']:
                            save_fn = dir+f'/catalog_rsd_recon_xi2d_{tracer}_z{zmin:.1f}-{zmax:.1f}_velbias_B_s_mockcov_{sysmodel}.dat.fits'
                            if not os.path.exists(save_fn):                                
                                Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}['IFFT']
                                if sysmodel == 'standard':
                                    data_positions = [catalog['x'], catalog['y'], catalog['z']]
                                    [catalog['x'], catalog['y'], catalog['z']] = run_reconstruction(Reconstruction, [catalog['x'], catalog['y'], catalog['z']], tracer, boxsize=boxsize)
                                    catalog.write(save_fn)
                                    print(f'save to {save_fn}', flush=True)
                                elif sysmodel == 'dv-obs':
                                    add_sys_spec_to_cat(catalog, tracer, zmin, zmax)
                                    [catalog['x'], catalog['y'], catalog['z_dv']] = run_reconstruction(Reconstruction, [catalog['x'], catalog['y'], catalog['z_dv']], tracer, boxsize=boxsize)
                                    catalog.write(save_fn)
                                    print(f'save to {save_fn}', flush=True)
                        for r_id in range(5):
                            ran_save_fn = dir+f'/catalog_rsd_recon_xi2d_{tracer}_z{zmin:.1f}-{zmax:.1f}_velbias_B_s_mockcov_{r_id}.ran.fits'
                            randoms = RandomBoxCatalog(boxsize=boxsize, csize=catalog.csize, seed=z_index*10+mock_id*100+1234 + r_id)
                            randoms_positions = randoms.get('Position').T
                            del randoms['Position']
                            randoms_positions_recon = run_reconstruction(Reconstruction, randoms_positions, tracer, boxsize=boxsize)
                            randoms['x'], randoms['y'], randoms['z'] = randoms_positions_recon
                            randoms.write(ran_save_fn)
                            print(f'save to {ran_save_fn}', flush=True)
                    '''
    elif option == 'QSO_test':
        z_index = 0
        (zmin, zmax) = Z_VSMEAR['QSO'][0]
        z_cubic = Z_CUBIC['LRG'][2]
        z_dir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/cosmosim/AbacusHOD_mocks_v1/CubicBox/LRG'
        for mock_id in mockids:
                mock_id03 =  f"{mock_id:03}"
                print(f'[BUILDING QSOTEST MOCKS] LRG3 {z_cubic} + dv_QSO_{(zmin, zmax)} {mock_id03}' , flush=True)
                catalog_fn = z_dir+f'/z{z_cubic:.3f}/AbacusSummit_base_c000_ph{mock_id03}/catalog_rsd_xi2d_lrg_main_z{z_cubic:.1f}_velbias_B_s_mockcov.fits'
                catalog = Catalog.read(catalog_fn, filetype='fits')
                dir = os.path.join(z_dir, f'obs_QSO/AbacusSummit_base_c000_ph{mock_id03}')
                save_fn = dir+f'/catalog_rsd_xi2d_LRG_z{zmin:.1f}-{zmax:.1f}_velbias_B_s_mockcov.fits'
                if not os.path.exists(save_fn):
                    add_sys_spec_to_cat(catalog, 'QSO', zmin, zmax)
                    catalog.write(save_fn)
                    print(f'[SAVE TO] {save_fn}', flush=True)
            
        