import fitsio
import sys, os
import numpy as np
import LSS.common_tools as common    
from astropy.table import Table,join,Column

c = 299792

def contamination(fn, zmin=0.8, zmax=2.1, nz_dir=None):
    # read the mock
    data = Table(fitsio.read(fn))
    data.meta.clear()
    if not data.meta:
        # implement the redshift error
        sys.path.append('/global/cfs/projectdirs/desi/users/jiaxiyu/repeated_observations/')
        from Y3_redshift_systematics import vsmear
        # only vsmear the redshift at zmin<z<zmax
        if "Z_ini" not in data.colnames:
            data["Z_ini"] = data["Z"].copy()
        selz  = (zmin < data['Z_ini']) & (data['Z_ini'] <=zmax)
        # data['Z'] += add_dv* selz * (1 + data['Z']) / c
        add_dv = vsmear('QSO', float(zmin), float(zmax), len(data), zmode='LSS', dvmode='obs')
        data['Z'] = data["Z_ini"] + add_dv * selz * (1 + data["Z_ini"]) / c
        # save the data
        data.meta['conta'] = 'yes'
        data.write(fn,overwrite=True)
        print(f'mock contaminated: {fn}',flush=True)
        # update the nz file
        ## refer to https://github.com/desihub/LSS/blob/main/scripts/main/mkCat_main.py for all values
        if nz_dir != None:
            fcr = f'{nz_dir}/QSO_0_clustering.ran.fits'
            fcd = f'{nz_dir}/QSO_clustering.dat.fits'
            fout= f'{nz_dir}/QSO_nz.txt'
            common.mknz(fcd,fcr,fout,bs=0.02,zmin=0.6,zmax=2.4,randens=2500)
            print(f'nz updated: {fout}',flush=True)
    else:
        print(f'mock already contaminated: {fn}',flush=True)

def match_redshift(fn_target,fn_data, nz_dir=None, region=None):
    # read the contaminated file
    # data_contam = Table(fitsio.read(fn_data.replace('dvs_ro','global'),columns=['TARGETID','Z']))
    data_contam = Table(fitsio.read(fn_data, columns=['TARGETID','Z']))
    # read the file that should be updated, remove the old redshift column
    catalogue_old = Table(fitsio.read(fn_target))
    if "Z_ini" not in catalogue_old.colnames:
        catalogue_old["Z_ini"] = catalogue_old["Z"].copy()
    if "Z" in catalogue_old.colnames:
        catalogue_old.remove_column('Z')
    catalogue_old.meta.clear()
    if not catalogue_old.meta:
        # match by TARGETID (dat file) or TARGETID_DATA (ran file)
        match_cat = data_contam.copy()
        match_key = 'TARGETID'
        if fn_target.find('ran') != -1:
            match_key = 'TARGETID_DATA'
            match_cat.rename_column('TARGETID', match_key)
        # match the redshift of the data to the target mock
        catalogue_new = join(catalogue_old,match_cat,keys=[match_key],join_type='left')
        assert not np.ma.is_masked(catalogue_new['Z']), f"Error: masked values found in {fn_target}"
        # save the updated catalog
        catalogue_new.meta['conta'] = 'yes'
        catalogue_new.write(fn_target, overwrite=True)
        print(f'redshift updated: {fn_target}', flush=True)
        if nz_dir != None:
            fcr = f'{nz_dir}/QSO_{region}_0_clustering.ran.fits'
            fcd = f'{nz_dir}/QSO_{region}_clustering.dat.fits'
            fout= f'{nz_dir}/QSO_{region}_nz.txt'
            common.mknz(fcd,fcr,fout,bs=0.02,zmin=0.6,zmax=2.4,randens=2500)
            print(f'nz updated: {fout}',flush=True)
    else:
        print(f'redshift already updated: {fn_target}', flush=True)

if __name__ == '__main__':
    from multiprocessing import Pool
    zmin = 0.8
    zmax = 2.1
    Nmock= 25
    Nran = 18

    base_dir = os.environ["SCRATCH"]+ '/galaxies/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{}/mock{}/LSScats/dv_obs_z{:.1f}-{:.1f}'
    base_dir_all = [base_dir.format(imock, imock, zmin, zmax) for imock in range(Nmock)]
    fn_dat_ = base_dir+'/QSO_clustering.dat.fits'
    fn_GC_dat_ = base_dir + '/QSO_{}_clustering.dat.fits'

    # contaminate the mock
    ## if you need to implement it multiple times, you can repeat this line with different zmin,zmax
    # example: for zmin,zmax in zip([0.8,1.1,1.4,1.7],[1.1,1.4,1.7,2.1]): ... (the content until the print())
    fns_contam = [fn_dat_.format(imock, imock, zmin, zmax) for imock in range(Nmock)]
    with Pool(processes = Nmock) as pool:
        pool.starmap(contamination, zip(fns_contam, [zmin]*Nmock, [zmax]*Nmock, base_dir_all))
    # update: NGC, SGC dat file
    fns_target = [fn_GC_dat_.format(imock, imock, zmin, zmax, gc) for gc in ['NGC', 'SGC'] for imock in range(Nmock)]
    with Pool(processes= Nmock*2) as pool:
        pool.starmap(match_redshift, zip(fns_target, fns_contam*2, base_dir_all*2, ['NGC','SGC']*Nmock))
    print('All contaminations are finished', flush=True)