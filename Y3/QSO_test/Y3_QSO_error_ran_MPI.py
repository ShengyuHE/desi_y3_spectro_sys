import os
import numpy as np

def MPI_match(fns_ran,fns_dat):
    from Y3_QSO_error import match_redshift

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    fn_dat_chunk    = fns_dat[rank::size]
    fn_ran_chunk = fns_ran[rank::size]

    for i, (fn_dat,fn_ran) in enumerate(zip(fn_dat_chunk,fn_ran_chunk)):
        try:
            match_redshift(fn_dat,fn_ran)
            # if i % 200 == 0:
                # print(f"[rank {rank}] processed {i+1}/{len(fn_dat)}", flush=True)
        except Exception as e:
            # Log and continue; avoids killing the whole allocation
            print(f"[rank {rank}] ERROR on {fn_dat}: {e}", flush=True)

    # Optional: synchronize, then have rank 0 print a final message
    comm.Barrier()
    if rank == 0:
        print("All ranks finished.", flush=True)

if __name__ == '__main__':
    #directory
    # path = '/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{}/mock{}/LSScats'
    zmin = 0.8
    zmax = 2.1 
    Nmock= 25
    Nran = 18

    base_dir = os.environ["SCRATCH"]+ '/galaxies/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{}/mock{}/LSScats/dv_obs_z{:.1f}-{:.1f}'
    base_dir_all = [base_dir.format(imock, imock, zmin, zmax) for imock in range(Nmock)]
    fn_dat_ = base_dir+'/QSO_clustering.dat.fits'
    fn_ran_ = base_dir+'/QSO_{}_clustering.ran.fits'

    # target random files (no NGC/SGC)
    fns_contam = [fn_dat_.format(imock, imock, zmin, zmax) for imock in range(Nmock)]

    # update: NGC+SGC, NGC, SGC ran file
    fns_target = [fn_ran_.format(imock,imock, zmin, zmax, iran) for iran in range(Nran) for imock in range(Nmock) ]+\
                 [fn_ran_.format(imock,imock, zmin, zmax, f'{gc}GC_{iran}') for gc in ['N','S'] for iran in range(Nran) for imock in range(Nmock)]
    MPI_match(fns_contam*3*Nran,fns_target)
