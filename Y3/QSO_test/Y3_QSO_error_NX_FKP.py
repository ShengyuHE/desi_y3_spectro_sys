import os
import LSS.common_tools as common    

#
zmin = 0.8
zmax = 2.1 

# dir_source = os.environ["SCRATCH"]+f'/galaxies/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{{}}/mock{{}}/LSScats/dv_obs_z0.8-2.1'
# fn_ = dir_source+'/QSO{}_clustering.{}.fits'


for imock in range(25):
    dir_source = f'/pscratch/sd/s/shengyu/galaxies/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{imock}/mock{imock}/LSScats/dv_obs_z0.8-2.1'
    common.addnbar(f'{dir_source}/QSO', bs=0.02,zmin=0.6,zmax=2.4,P0=6000,par='y')
    common.addnbar(f'{dir_source}/QSO_NGC', bs=0.02,zmin=0.6,zmax=2.4,P0=6000,par='y')
    common.addnbar(f'{dir_source}/QSO_SGC', bs=0.02,zmin=0.6,zmax=2.4,P0=6000,par='y')