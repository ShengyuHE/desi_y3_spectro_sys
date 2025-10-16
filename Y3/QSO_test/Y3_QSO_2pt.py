import os
import sys
import argparse
import fitsio
import numpy as np
from astropy.table import Table, vstack
from scipy.interpolate import interp1d

from cosmoprimo.fiducial import AbacusSummit
from pypower import CatalogFFTPower, mpi, setup_logging
from pycorr import TwoPointCorrelationFunction, setup_logging
mpicomm = mpi.COMM_WORLD
setup_logging()

sys.path.append('/global/homes/s/shengyu/project_rc/main/Y3/')
from helper import REDSHIFT_VSMEAR, Y3_NRAN, Y3_ZRANGE
from helper import select_region, sky_to_cartesian

c = 299792 # speed of light in km/s
cosmo = AbacusSummit()

def get_edges(corr_type='smu', bin_type='lin'):
    if bin_type == 'log':
        sedges = np.geomspace(0.01, 100., 49)
    elif bin_type == 'lin':
        sedges = np.linspace(0., 200, 201)
    else:
        raise ValueError('bin_type must be one of ["log", "lin"]')
    if corr_type == 'smu':
        edges = (sedges, np.linspace(-1., 1., 201)) #s is input edges and mu evenly spaced between -1 and 1
    elif corr_type == 'rppi':
        if bin_type == 'lin':
            edges = (sedges, np.linspace(-40., 40, 101)) #transverse and radial separations are coded to be the same here
        else:
            edges = (sedges, np.linspace(-40., 40., 81))
    elif corr_type == 'theta':
        edges = (np.linspace(0., 4., 101),)
    else:
        raise ValueError('corr_type must be one of ["smu", "rppi", "theta"]')
    return edges

def get_positions_weights(catalog, tracer, region, zrange, weight_type = 'default', systematics_type = None):
    z2d = cosmo.comoving_radial_distance
    toret = []
    if isinstance(catalog, (tuple, list)):  # list of catalogs, one for each region
        for cat in catalog:
            toret += get_positions_weights(cat, tracer, region, zrange, weight_type = 'default', systematics_type = None)
    else:
        if systematics_type == 'dv-obs':
            import time
            from Y3_redshift_systematics import vsmear, vsmear_modelling
            random_seed = int(time.time() * 1000) % (2**32)
            zmid = (zrange[0]+zrange[1])/2
            dv = vsmear(tracer, zmin=zrange[0], zmax=zrange[1], Ngal = len(catalog), seed=random_seed)
            dz = dv*(1+zmid)/c # ∆z = ∆v*(1+z)/c
            catalog['Z'] += dz
        maskz = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1]) 
        mask  = maskz & select_region(catalog['RA'], catalog['DEC'], region=region)
        cat = catalog[mask]
        dist = z2d(cat['Z'])
        positions = [cat['RA'].data, cat['DEC'].data, dist]
        weights = np.ones(len(cat), dtype=bool)
        if 'default' in weight_type:
            weights = cat['WEIGHT'].data
        # toret.append((np.array(positions), np.array(weights)))
        toret.append((np.array(positions), np.array(weights)))
    return toret

def normalize_data_randoms_weights(data_weights, randoms_weights, weight_attrs=None, mpicomm=None):
    # Renormalize randoms / data for each input catalogs
    # data_weights should be a list (for each N/S catalogs) of weights
    import inspect
    from pycorr import mpi
    from pycorr.twopoint_counter import _format_weights, get_inverse_probability_weight
    if mpicomm is None:
        mpicomm = mpi.COMM_WORLD
    if weight_attrs is None: weight_attrs = {}
    weight_attrs = {k: v for k, v in weight_attrs.items() if k in inspect.getargspec(get_inverse_probability_weight).args}
    def sum_weights(*weights):
        sum_weights, formatted_weights = [], []
        for weight in weights:
            weight, nbits = _format_weights(weight, copy=True)  # this will sort bitwise weights first, then single individual weight
            iip = (get_inverse_probability_weight(weight[:nbits], **weight_attrs) if nbits else 1.) * weight[nbits]
            sum_weights.append(mpicomm.allreduce(np.sum(iip)))
            formatted_weights.append(weight)
        return sum_weights, formatted_weights
    data_sum_weights, data_weights = sum_weights(*data_weights)
    randoms_sum_weights, randoms_weights = sum_weights(*randoms_weights)
    all_data_sum_weights, all_randoms_sum_weights = sum(data_sum_weights), sum(randoms_sum_weights)
    for icat, rw in enumerate(randoms_weights):
        if randoms_sum_weights[icat] != 0:
            factor = data_sum_weights[icat] / randoms_sum_weights[icat] * all_randoms_sum_weights / all_data_sum_weights
            rw[-1] *= factor
    return data_weights, randoms_weights

def concatenate_data_randoms(data_positions_weights, *randoms_positions_weights, weight_attrs=None, concatenate=None, mpicomm=None):
    # data_positions_weights: list of (positions, weights) (for each region)
    # randoms_positions_weights: list of (positions, weights) (for each region)
    def _concatenate(array):
        if isinstance(array[0], (tuple, list)):
            array = [np.concatenate([arr[iarr] for arr in array], axis=0) if array[0][iarr] is not None else None for iarr in range(len(array[0]))]
        elif array is not None:
            array = np.concatenate(array)  # e.g. Z column
        return array
    data_positions, data_weights = tuple(_concatenate([pw[i] for pw in data_positions_weights]) for i in range(len(data_positions_weights[0]) - 1)), [pw[-1] for pw in data_positions_weights]
    if not randoms_positions_weights:
        data_weights = _concatenate(data_weights)
        return data_positions + (data_weights,)
    # import pdb; pdb.set_trace()
    list_randoms_positions = tuple([_concatenate([pw[i] for pw in rr]) for rr in randoms_positions_weights] for i in range(len(randoms_positions_weights[0][0]) - 1))
    list_randoms_weights = [[pw[-1] for pw in rr] for rr in randoms_positions_weights]
    for iran, randoms_weights in enumerate(list_randoms_weights):
        list_randoms_weights[iran] = _concatenate(normalize_data_randoms_weights(data_weights, randoms_weights, weight_attrs=weight_attrs)[1])
    data_weights = _concatenate(data_weights)
    list_randoms_positions_weights = list_randoms_positions + (list_randoms_weights,)
    if concatenate:
        list_randoms_positions_weights = tuple(_concatenate(rr) for rr in list_randoms_positions_weights)
    return data_positions + (data_weights,), list_randoms_positions_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs = '+')
    parser.add_argument("--nthreads", type = int, 
                        default = 4)
    parser.add_argument("--gpu", action = 'store_true')
    args = parser.parse_args()

    # tracers         = ['LRG', 'ELG_LOPnotqso', 'QSO']
    tracers         = ['QSO']
    weight_type     = 'default'
    corr_type       = 'smu'
    bin_type        = 'lin'
    nsplits         = 5
    selection_attrs = None

    dtype = 'f8'
    kwargs = {'dtype': dtype}
    my_script = True
    y3_scripts = False
    split_randoms_above = None
    if split_randoms_above is None: split_randoms_above = np.inf

    jobs = [(tracer, mock_id, zrange) for tracer in tracers for zrange in Y3_ZRANGE[tracer] for mock_id in range(0,1)]
    for tracer, mock_id, zrange in jobs:
        basedir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/DA2/Abacus_v4_1/altmtl{mock_id}/kibo-v1/mock{mock_id}/LSScats'
        # basedir = f'/pscratch/sd/s/shengyu/galaxies/catalogs/Y1/Abacus_v4_2/altmtl0/iron/mock0/LSScats'
        data_positions1 = data_weights1 = data_samples1 = data_positions2 = data_weights2 = data_samples2 = None
        randoms_z1 = randoms_positions1 = randoms_weights1 = randoms_samples1 = randoms_z2 = randoms_positions2 = randoms_weights2 = randoms_samples2 = None
        shifted_positions1 = shifted_weights1 = shifted_samples1 = shifted_positions2 = shifted_weights2 = shifted_samples2 = None
        jack_positions = None

        ran_mock_fns = [basedir+f'/{tracer}_{region}_{{}}_clustering.ran.fits'.format(i) for i in range(Y3_NRAN[tracer])]
        ran_mocks = vstack([Table(fitsio.read(v, columns=['RA', 'DEC', 'Z', 'WEIGHT'])) for v in ran_mock_fns])
        ran_mocks = np.array_split(ran_mocks, nsplits)
        randoms = [get_positions_weights(rr, tracer, region, zrange, weight_type = 'default', systematics_type = None) for rr in ran_mocks]

        # for sys_type in [None, 'dv-obs']:
        for sys_type in [None]:
            dat_mock_fn = basedir+f'/{tracer}_{region}_clustering.dat.fits'
            data_mock = Table(fitsio.read(dat_mock_fn, columns=['RA', 'DEC', 'Z', 'WEIGHT']))
            if sys_type == None:
                clustering_root = basedir+f'/xi/{corr_type}/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weight_type}_{bin_type}_nran{Y3_NRAN[tracer]}_split{nsplits}.npy'
                # clustering_root = f'/global/homes/s/shengyu/project_rc/main/Y3/bao_test/xi/mu/xi/{corr_type}/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weight_type}_{bin_type}_nran{Y3_NRAN[tracer]}_split{nsplits}.npy'
                data = get_positions_weights(data_mock, tracer, region, zrange, weight_type = 'default', systematics_type = None)
            elif sys_type == 'dv-obs':
                clustering_root = basedir+f'/xi/{corr_type}/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weight_type}_{bin_type}_nran{Y3_NRAN[tracer]}_split{nsplits}_dv-obs.npy'
                # clustering_root = f'/global/homes/s/shengyu/project_rc/main/Y3/bao_test/xi/mu/xi/{corr_type}/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weight_type}_{bin_type}_nran{Y3_NRAN[tracer]}_split{nsplits}_dv-obs.npy'
                data = get_positions_weights(data_mock, tracer, region, zrange, weight_type = 'default', systematics_type = sys_type)

            clustering_root = basedir+f'/xi/{corr_type}/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weight_type}_{bin_type}_nran{Y3_NRAN[tracer]}_split{nsplits}.npy'
            if not os.path.exists(clustering_root) and my_script:           
                print(f"\nProcessing for {tracer} tracer, {region} region, mock{mock_id}, z bin in [{zrange[0]}, {zrange[1]}] with {sys_type}", flush = True)
                (data_positions, data_weights), (randoms_positions, randoms_weights) = concatenate_data_randoms(data, *randoms)
                result = 0
                edges = get_edges(corr_type, bin_type)
                D1D2 = None
                for i in range(nsplits):
                    data_positions1 = data_positions
                    data_weights1 = data_weights
                    randoms_positions1 = randoms_positions[i]
                    randoms_weights1 = randoms_weights[i][0]
                    print("data:",data_positions1.shape, data_weights1.shape, flush=True)
                    print(f"{nsplits} split random:",randoms_positions1.shape, randoms_weights1.shape)
                    tmp = TwoPointCorrelationFunction(corr_type, edges, data_positions1=data_positions1, data_weights1=data_weights1,
                                        randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                                        shifted_positions1 = None, shifted_weights1 = None,
                                        engine='corrfunc', position_type = 'rdd', #los = 'firstpoint',
                                        nthreads=4, gpu = args.gpu,
                                        D1D2 = D1D2, selection_attrs = selection_attrs,
                                        )
                    D1D2 = tmp.D1D2
                    result += tmp
                result.save(clustering_root)
            break
            # clustering_root = basedir+f'/xi/{corr_type}/corr_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weight_type}_{bin_type}_nran{Y3_NRAN[tracer]}_split{nsplits}.npy'
            # if not os.path.exists(clustering_root) and y3_scripts:  
            #     (data_positions, data_weights), (randoms_positions, randoms_weights) = concatenate_data_randoms(data, *randoms)
            #     edges = get_edges(corr_type, bin_type)
            #     zedges = np.array(list(zip(edges[0][:-1], edges[0][1:])))
            #     mask = zedges[:, 0] >= split_randoms_above
            #     zedges = [zedges[~mask], zedges[mask]]
            #     split_edges, split_randoms = [], []
            #     for ii, zedge in enumerate(zedges):
            #         if zedge.size:
            #             split_edges.append([np.append(zedge[:, 0], zedge[-1, -1])] + list(edges[1:]))
            #             split_randoms.append(ii > 0)
            #     randoms_kwargs = dict(randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1, randoms_samples1=randoms_samples1,
            #                 randoms_positions2=randoms_positions2, randoms_weights2=randoms_weights2, randoms_samples2=randoms_samples2,
            #                 shifted_positions1=shifted_positions1, shifted_weights1=shifted_weights1, shifted_samples1=shifted_samples1,
            #                 shifted_positions2=shifted_positions2, shifted_weights2=shifted_weights2, shifted_samples2=shifted_samples2)
            #     results = []
            #     # nsplits = min(len(pw) if pw is not None else np.inf for pw in randoms_kwargs.values())
            #     if mpicomm.rank == 0:
            #         print('Using {:d} randoms splits for s > {:.0f}.'.format(nsplits, split_randoms_above), flush=True)
            #     for i_split_randoms, edges in zip(split_randoms, split_edges):
            #         result = 0
            #         D1D2 = None
            #         for isplit in range(nsplits if i_split_randoms else 1):
            #             tmp_randoms_kwargs = dict(randoms_kwargs)
            #             if i_split_randoms:
            #                 # On scales above split_randoms_above, sum correlation function over multiple randoms
            #                 for name, arrays in randoms_kwargs.items():
            #                     if arrays is None: continue
            #                     tmp_randoms_kwargs[name] = arrays[isplit]
            #                 if mpicomm.rank == 0:
            #                     print('Running split {:d} / {:d} for edges = {:.1f} - {:.1f}.'.format(isplit + 1, nsplits, edges[0][0], edges[0][-1]),flush=True)
            #             else:
            #                 for name, arrays in randoms_kwargs.items():
            #                     if arrays is None:
            #                         continue
            #                     elif 'samples' in name:
            #                         array = np.concatenate(arrays, axis=0)
            #                     else:  # e.g., list of positions / weights
            #                         array = [np.concatenate([arr[iarr] for arr in arrays], axis=0) for iarr in range(len(arrays[0]))]
            #                     tmp_randoms_kwargs[name] = array
            #             mesh_refine_factors = (4, 4, 2) if i_split_randoms else (2, 2, 1)
            #             tmp = TwoPointCorrelationFunction(corr_type, edges, data_positions1=data_positions, data_weights1=data_weights, data_samples1=data_samples1,
            #                                             data_positions2=data_positions2, data_weights2=data_weights2, data_samples2=data_samples2,
            #                                             engine='corrfunc', position_type='rdd', nthreads=4, gpu=args.gpu, **tmp_randoms_kwargs, **kwargs,
            #                                             D1D2=D1D2, selection_attrs=selection_attrs, mpicomm=mpicomm, mpiroot=None, mesh_refine_factors=mesh_refine_factors)
            #             D1D2 = tmp.D1D2
            #             if mpicomm.rank == 0:
            #                 print('Adding correlation function with edges = {}'.format(tmp.D1D2.edges[0]), flush=True)
            #             result += tmp
            #         results.append(result)
            #     corr = results[0].concatenate_x(*results)
            #     corr.save(clustering_root)

            print(f"\n 2pcf {tracer} tracer, mock{mock_id}, z bin in [{zrange[0]}, {zrange[1]}] complete", flush = True)
