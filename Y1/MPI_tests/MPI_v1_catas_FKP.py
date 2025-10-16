# srun -N 1 -n 1 --exclusive -C cpu -t 04:00:00 --qos interactive --account desi python MPI_catas_FKP.py --tracer ELG_LOPnotqso --region NGC --parallel y --targDir /global/homes/s/shengyu/project_rc/main/make_LSS/example_files
import os
import glob
import time
import fitsio
import argparse
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

c = 299792 # speed of light in km/s
P0_values = {'ELG_LOPnotqso': 4000, 'LRG': 10000, 'QSO': 6000, 'BGS': 7000}
NRAN_values = {'ELG_LOPnotqso':10, 'LRG':8, 'QSO':4}

def NXnorm(catalog_fn, nz_fn):
    #Calculate the NXnorm for the catalog, norm = NX/n(z)
    catalog = Table(fitsio.read(catalog_fn)) 
    nz      = np.loadtxt(nz_fn)
    tree    = cKDTree(nz[:, 0].reshape(-1, 1))
    _, ind_rawNZ = tree.query(catalog['Z'].reshape(-1, 1))
    norm         = catalog['NX']/nz[ind_rawNZ,3]
    return norm

def FKPupdate(catalog_fn, nz_fn, catas_type, norm):
    T0 = time.time()
    catalog=Table(fitsio.read(catalog_fn))
    # Check for Z_{catas_type} existence
    if f'Z_{catas_type}' not in catalog.colnames:
        raise ValueError(f"Invalid Zcatas type: '{catas_type}'.")

    catalog[f'FKP_{catas_type}'] = catalog['WEIGHT_FKP'].copy()
    # Load the nz_catas and create the cKDTree
    nz = np.loadtxt(nz_fn)
    tree = cKDTree(nz[:, 0].reshape(-1, 1))
    # caluclate the completeness rescaling of nz for FKP weight
    dv = (catalog[f'Z_{catas_type}'] - catalog['Z']) / (1 + catalog['Z']) * c
    dz = catalog[f'Z_{catas_type}'] - catalog['Z']
    # tmp      = np.argsort(catalog['RA'], kind='mergesort')
    tmp      = np.argsort(catalog,order=['RA', 'DEC'])
    catalog  = catalog[tmp]
    norm     = norm[tmp]
    dv       = dv[tmp]
    NX       = catalog['NX'].copy()
    norm[norm==0] = np.nan
    # print('there are {} samples to find new FKP'.format(np.sum((dv!=0)&(np.isnan(norm)))), flush=True)
    for ID in np.where((dv!=0)&(np.isnan(norm)))[0]:
        if (2<ID)&(ID<len(catalog)-2):
            norm[ID] = np.nanmedian(norm[[ID-2,ID-1,ID+1,ID+2]])
        elif ID<2:
            norm[ID] = np.nanmedian(norm[[ID+1,ID+2]])
        elif ID>len(catalog)-2:
            norm[ID] = np.nanmedian(norm[[ID-2,ID-1]])
        # update NX for norm ==0
        ind_newNZ = np.argmin(abs(catalog[f'Z_{catas_type}'][ID]-nz[:,0]))
        NX[ID] = norm[ID]*nz[ind_newNZ,3]
    # update NX and WEIGHT_FKP columns for all catastrophics
    sel = dv != 0
    _, ind_newNZ = tree.query(catalog[f'Z_{catas_type}'][sel].reshape(-1, 1))
    NX[sel] = norm[sel] * nz[ind_newNZ, 3]
    catalog[f'FKP_{catas_type}'][sel] = 1 / (NX[sel] * P0 + 1)
    catalog[f'FKP_{catas_type}'][np.isnan(catalog[f'FKP_{catas_type}'])] = 1
    print('[FKP weight]: implement {} catastrophophics took time: {:.2f}s'.format(catas_type, time.time()-T0), flush=True)
    catalog.write(catalog_fn, overwrite=True)
    # print(f'{catas_type} catastrophics FKP corrected')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--targDir", help="base directory for input",default=None)
    parser.add_argument("--tracer", help="tracer type to be selected", choices=['ELG_LOPnotqso','LRG','QSO'],default=None)
    parser.add_argument("--region", help="region to be selected", choices=['NGC','SGC','NGC SGC'], default ='NGC SGC')
    parser.add_argument("--mock_type", help="mocks type", choices=['dat','ran','all'], default='all')
    parser.add_argument("--addcatas",help="apply catastrophics in the clean mock redshift 'Z', 'realistic' is the observed pattern, 'failures' is the upper limit 1%, slitless is the 5% assumed catastrophics",nargs='*',type=str,choices=['realistic','failures','slitless'], default=['realistic','failures'])
    parser.add_argument('--remove_zerror', help='the suffix of redshift column without the redsihft error', type=str, default=None)
    parser.add_argument("--parallel", help="MPI parallelization",choices=['y','n'], default='y')
    args = parser.parse_args()

    if args.remove_zerror == "None":
        args.remove_zerror = None

    T0 = time.time()

    filedir = args.targDir
    tracer = args.tracer
    mock_types = ['dat','ran'] if args.mock_type == 'all' else [args.mock_type]
    regions = ['NGC', 'SGC'] if args.region == 'NGC SGC' or args.region == 'SGC NGC' else [args.region]

    P0 = P0_values.get(tracer, None)
    NRAN = NRAN_values.get(tracer, None)

    if args.addcatas is not None:
        types = args.addcatas
    if args.remove_zerror is not None:
        types = args.remove_zerror

    if args.parallel == 'y':
        from mpi4py import MPI
        mpicomm = MPI.COMM_WORLD
        mpiroot = 0
        rank = mpicomm.Get_rank()
        size = mpicomm.Get_size()
        if rank == mpiroot:
            print("[FKP weight] Enable MPI parallization")
    else:
        rank = 0
        size = 1

    # Process `dat` mocks only on root rank 
    if rank == 0:
        for region in regions:
            nz_fiducial_fn = filedir + f'/{tracer}_{region}_nz.txt'
            catalog_fn = filedir + f'/{tracer}_{region}_clustering.dat.fits'
            norm = NXnorm(catalog_fn, nz_fiducial_fn)
            for catas_type in types:
                nz_fn = filedir + f'/{tracer}_{region}_nz_{catas_type}.txt'
                FKPupdate(catalog_fn, nz_fn, catas_type, norm)
                print(f'[FKP weight]: {tracer}_{region} dat mocks FKP_{catas_type} weight updated', flush=True)

    # collect job_imtes for `ran` mocks
    job_items = [
    (region, 'ran', 
     filedir + f'/{tracer}_{region}_{num}_clustering.ran.fits', 
     filedir + f'/{tracer}_{region}_nz.txt', num)
    for region in regions
    for num in range(4)
    ]
    
    # Divide work items among ranks
    job_items_indices = np.array_split(job_items, size)
    local_job_items = job_items_indices[rank]

    # Process `ran` mocks with multiple ranks
    for region, mock_type, catalog_fn, nz_fiducial_fn, num in local_job_items:
        norm = NXnorm(catalog_fn, nz_fiducial_fn)
        for catas_type in types:
            nz_fn = filedir + f'/{tracer}_{region}_nz_{catas_type}.txt'
            FKPupdate(catalog_fn, nz_fn, catas_type, norm)
            print(f'[Rank {rank}] [FKP weight]: {tracer}_{region} {num if num is not None else ""} mocks FKP_{catas_type} weight updated', flush=True)

    # Synchronize all ranks if in parallel mode
    if args.parallel == 'y':
        mpicomm.Barrier()
    if rank == 0:
        print('[FKP weight]: Implement all catastrophophics takes time: {:.2f}s'.format(time.time()-T0), flush=True)