##############################################################################
#
# Fourier maps calculations
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

import numpy as np
from .scatteringdata import ScatteringData


def calc_fourier_map(data: ScatteringData, map_type="obs", sharpen=True, resolution=0.3):
    """
    Compute a 3D Fourier map given a ScatteringData object
    :param data: a ScatteringData object with observed data, either
        q DiffractionDataSingleCrystal or PowderPatternDiffraction after
        extraction of intensities (profile fitting)
    :param map_type: either "obs" (the default), "diff" (difference)
        or "calc"
    :param sharpen: if True, normalise the structure factor Fourier coefficients
        by the average atomic scattering factor to sharpen the Fourier maps.
    :param resolution: approximate desired resolution for the map, in Angstroems
    :return: the 3D Fourier map, computed over one unit cell, with a resolution
        dictated by the largest HKL values. The map's origin is at the corner
        of the unit cell.
    """
    if "calc" not in map_type:
        obs2 = data.GetFhklObsSq()
        nb = len(obs2)
    else:
        nb = data.GetNbReflBelowMaxSinThetaOvLambda()
    calc = (data.GetFhklCalcReal() + 1j * data.GetFhklCalcImag())[:nb]
    c = data.GetCrystal()
    if sharpen:
        # Dictionary of scattering factor for all elements
        vsf = data.GetScatteringFactor()
        # Make this a dictionary against int_ptr, so it can match
        # the Crystal's ScatteringComponentList
        vsf = {k.int_ptr(): v for k, v in vsf.items()}
        norm_sf = np.zeros(nb)
        norm0 = 0
        for sc in c.GetScatteringComponentList():
            norm_sf += sc.mOccupancy * sc.mDynPopCorr * vsf[sc.mpScattPow.int_ptr()][:nb] ** 2
            norm0 += sc.mOccupancy * sc.mDynPopCorr * \
                     sc.mpScattPow.GetForwardScatteringFactor(data.GetRadiationType()) ** 2
        norm_sf = np.sqrt(norm_sf / norm0)
    # Scale obs and calc
    if "calc" not in map_type:
        scale_fobs = np.sqrt((abs(calc) ** 2).sum() / obs2.sum())
        print(" Fourier map obs scale factor:", scale_fobs)
    vol = c.GetVolume()
    spg = c.GetSpaceGroup()
    h, k, l = data.GetH()[:nb], data.GetK()[:nb], data.GetL()[:nb]
    # Map size to achieve resolution
    nx = int(np.ceil(c.a / resolution))
    ny = int(np.ceil(c.b / resolution))
    nz = int(np.ceil(c.c / resolution))
    # print("UnitCellMap::CalcFourierMap():",nx,",",ny,",",nz,",type:", map_type, );
    rhof = np.zeros((nz, ny, nx), dtype=np.complex64)
    for i in range(nb):
        norm = 1
        if sharpen:
            norm = 1 / norm_sf[i]
        if "calc" not in map_type:
            obs = scale_fobs * np.sqrt(obs2[i])
        acalc = np.abs(calc[i])
        for h0, k0, l0, fr, fi in spg.GetAllEquivRefl(h[i], k[i], l[i], False, data.IsIgnoringImagScattFact(),
                                                      calc.real[i], calc.imag[i]):
            if abs(2 * h0) < nx and abs(2 * k0) < ny and abs(2 * l0) < nz:
                # Integer indices
                ih = int(np.round(h0))
                ik = int(np.round(k0))
                il = int(np.round(l0))
                if "calc" in map_type.lower():
                    rhof[il, ik, ih] = (fr + 1j * fi) * norm / vol
                elif "obs" in map_type.lower():
                    rhof[il, ik, ih] = (fr + 1j * fi) * obs / acalc * norm / vol
                else:
                    rhof[il, ik, ih] = (fr + 1j * fi) * (obs - acalc) / acalc * norm / vol
            # if (i<5):
            #     print(int(h0)," ",int(k0)," ",int(l0),"(",spg.IsReflCentric(h0,k0,l0),"):"
            #           ,fr+1j*fi," :",rhof[il, ik, ih])
    if "obs" in map_type.lower() or "calc" in map_type.lower():
        # F000 for obs and calc maps
        nbsym = spg.GetNbSymmetrics(False, False)
        for sc in c.GetScatteringComponentList():
            sp = sc.mpScattPow
            rhof[0, 0, 0] += sp.GetForwardScatteringFactor(data.GetRadiationType()) * \
                             sc.mOccupancy * sc.mDynPopCorr * nbsym / vol
        # print("F000 =", rhof[0, 0, 0])
    return np.fft.fftn(rhof)  # , norm="backward"
