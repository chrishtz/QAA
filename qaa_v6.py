import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
# from libtiff import TIFF
from PIL import Image
import os

def qaa_v6(Rrs, wavel, bbw, aw, acoefs):
    # wavel = np.array([412, 443, 469, 488, 531, 547, 555, 645, 667, 678])
    idx412 = 0
    idx443 = 1
    idx488 = 3
    idx547 = 5
    idx555 = 6
    idx667 = 8

    # step 0
    if Rrs[idx667] > 20.0*np.power(Rrs[idx555], 1.5) or Rrs[idx667] < 0.9*np.power(Rrs[idx555], 1.7):
        Rrs[idx667] = 1.27*np.power(Rrs[idx555], 1.47) + 0.00018*np.power(Rrs[idx488]/Rrs[idx555], -3.19)

    rrs = Rrs / (0.52 + 1.7 * Rrs)

    # step 1  compute u
    g0 = 0.089
    g1 = 0.1245
    u= (-g0 + np.power(g0*g0 + 4.0*g1*rrs, 0.5)) / (2.0*g1)

    # step2 compute aref
    if Rrs[idx667] < 0.0015:
        numer = rrs[idx443] + rrs[idx488]
        denom = rrs[idx555] + 5.0 * rrs[idx667]*(rrs[idx667]/rrs[idx488])
        rho = np.log10(numer/denom)
        rho = acoefs[0] + acoefs[1]*rho + acoefs[2]*rho*rho
        aref = aw[idx555] + np.power(10, rho)
        idxref = idx555
        # bbpref = (u[6]*aref)/(1.0-u[6]) - bbw[6]
    else:
        aref = aw[idx667] + 0.39 * np.power(Rrs[idx667]/(Rrs[idx443]+Rrs[idx488]), 1.14)
        idxref = idx667

    # step3 compute bbpref
    bbpref = (u[idxref]*aref) / (1.0-u[idxref]) - bbw[idxref]

    # step4 compute eta(Y)
    rat = rrs[idx443]/rrs[idx555]
    Y = 2.0 * (1.0 - 1.2 * np.exp(-0.9*rat))

    # step5 bb(bb = bbp+bbw)
    bbp = bbpref*np.power(wavel[idxref]/wavel, Y)
    bb = bbp + bbw

    # step6 a
    a = ((1.0 - u)*bb)/u

    return bb, a, rrs

def qaa_decomp(rrs, wavel, a, aw):
    # step7
    idx412 = 0
    idx443 = 1
    idx488 = 3
    idx547 = 5
    idx555 = 6
    idx667 = 8
    rat = rrs[idx443]/rrs[idx555]
    symbol = 0.74 + (0.2 / (0.8+rat))

    # step8
    S = 0.015
    Sr = S + 0.002 / (0.6+rat)
    zeta = np.exp(Sr * (442.5-415.5))

    # step9 adg(443)
    denom = zeta - symbol
    dif1 = a[idx412] - symbol*a[idx443]
    dif2 = aw[idx412] - symbol*aw[idx443]
    adg443 = (dif1 - dif2) / denom

    # step10 adg & aph
    adg = adg443 * np.exp(Sr * (wavel[idx443] - wavel))
    aph = a - adg -aw

    # aph check
    x1 = aph[idx443] / a[idx443]
    if(x1<0.15 or x1>0.6):
        x2 = -0.8 + 1.4*(a[idx443] - aw[idx443])/(a[idx412] - aw[idx412])
        if(x2 < 0.15):
            x2 = 0.15
        if(x2 > 0.6):
            x2 = 0.6

        aph[idx443] = a[idx443] * x2
        adg443 = a[idx443] - aph[idx443] - aw[idx443]

        adg = adg443 * np.exp(Sr * (wavel[idx443] - wavel))
        aph = a - adg - aw

    return adg443, aph, adg


if __name__ =="__main__":

    filename = './ocdata/inputfiles/A2006167181000Rrs.L2.nc'
    nc_file = nc4.Dataset(filename, 'r')

    dim_dic = nc_file.dimensions

    variables = nc_file.groups['geophysical_data'].variables
    variables_s = nc_file.groups['sensor_band_parameters'].variables
    variables_n = nc_file.groups['navigation_data'].variables

    # nc_file.close()
    print("file reading done")

    nl = dim_dic['number_of_lines'].size
    pl = dim_dic['pixels_per_line'].size
    pcp = dim_dic['pixel_control_points'].size


    wavelength_all = variables_s['wavelength'][:]
    aw_all = variables_s['aw'][:]
    bbw_all = variables_s['bbw'][:]


    lon = variables_n['longitude'][:]
    lat = variables_n['latitude'][:]
    lonarray = np.array(lon)
    latarray = np.array(lat)

    # Modis
    acoefs = np.array([-1.204, -1.229, -0.395])
    nbands = 10
    wavel = np.array(wavelength_all[:nbands])
    aw = np.array(aw_all[:nbands])
    bbw = np.array(bbw_all[:nbands])

    # Modisa 2016/1 412 443 469 488 531 547 555 645 667 678
    Rrs_412 = variables['Rrs_412'][:]
    temp = np.array(Rrs_412)
    # temp = temp.astype(np.float64)

    nrows, ncols = temp.shape
    Rrs = temp[np.newaxis, :, :]

    # filter out the pixels
    rows_temparray, cols_temparray = np.where(temp!=-32767.0)
    rows_templist = list(rows_temparray)
    cols_templist = list(cols_temparray)
    rcpairs_templist = list(zip(rows_templist, cols_templist))

    for i in variables:
        if 'Rrs' in i and '412' not in i:
            var = np.array(variables[i][:])
            var_add = var[np.newaxis, :, :]
            Rrs = np.append(Rrs, var_add, axis=0)

    bb = np.full((nbands, nrows, ncols), np.nan)
    a = np.full((nbands, nrows, ncols), np.nan)
    aph = np.full((nbands, nrows, ncols), np.nan)
    adg = np.full((nbands, nrows, ncols), np.nan)
    adg443 = np.full((nrows, ncols), np.nan)

    for rcpair_i in rcpairs_templist:
        rowidx = rcpair_i[0]
        colidx = rcpair_i[1]
        Rrs_p = Rrs[:, rowidx, colidx]

        bb_p, a_p, rrs_p = qaa_v6(Rrs_p, wavel, bbw, aw, acoefs)
        adg443_p, aph_p, adg_p = qaa_decomp(rrs_p, wavel, a_p, aw)
        bb[:, rowidx, colidx] = bb_p
        a[:, rowidx, colidx] = a_p
        aph[:, rowidx, colidx] = aph_p
        adg[:, rowidx, colidx] = adg_p
        adg443[rowidx, colidx] = adg443_p

    print("qaa calculation done")

    # error analysis & visualization
    filename1 = './ocdata/inputfiles/A2006167181000l2prodadg443qaa.L2.nc'
    nc_file1 = nc4.Dataset(filename1, 'r')
    variables1 = nc_file1.groups['geophysical_data'].variables

    adg443qaaref = variables1['adg_443_qaa']
    adg443qaaref_array = np.array(adg443qaaref)
    adg443qaaref_array[adg443qaaref_array == -32767.0] = np.nan

    xx = np.arange(0, 5, 0.1)
    adg443flat = adg443.flatten()
    adg443refflat = adg443qaaref_array.flatten()

    adg443flat_cut = adg443flat.copy()
    adg443refflat_cut = adg443refflat.copy()
    adg443flat_cut[np.logical_or(adg443flat_cut < -5, adg443flat_cut > 5)] = np.nan
    adg443refflat_cut[np.logical_or(adg443refflat_cut < -5, adg443refflat_cut > 5)] = np.nan

    plt.scatter(adg443flat_cut, adg443refflat_cut, color='none', marker='o', edgecolors='r')
    plt.plot(xx, xx, 'b-*')
    plt.xlabel('adg443')
    plt.ylabel('adg443ref')
    plt.savefig('./ocdata/outputfiles/adg443.jpg')

    plt.show()


    # write files out
    # adg443[np.isnan(adg443)] = -32767.0
    # adg443qaaref_array[np.isnan(adg443qaaref_array)] = -32767.0

    f = nc4.Dataset('./ocdata/outputfiles/adg443.nc', 'w', format='NETCDF4')

    f.createDimension('number_of_lines', nl)
    f.createDimension('pixels_per_line', pl)
    f.createDimension('pixel_control_points', pcp)

    geophygrp = f.createGroup('geophysical_data')
    naviggrp = f.createGroup('navigation_data')


    adg443_nc = geophygrp.createVariable('adg443', 'f4', ('number_of_lines', 'pixels_per_line'))
    adg443ref_nc = geophygrp.createVariable('adg443ref', 'f4', ('number_of_lines', 'pixels_per_line'))
    longitude_nc = naviggrp.createVariable('longitude', 'f4', ('number_of_lines', 'pixel_control_points'))
    latitude_nc = naviggrp.createVariable('latitude', 'f4', ('number_of_lines', 'pixel_control_points'))


    adg443_t = np.rot90(adg443, 2)
    adg443qaaref_array_t = np.rot90(adg443qaaref_array, 2)
    lonarray_t = np.rot90(lonarray, 2)
    latarray_t = np.rot90(latarray, 2)
    adg443_nc[:, :] = adg443_t
    adg443ref_nc[:, :] = adg443qaaref_array_t
    longitude_nc[:, :] = lonarray_t
    latitude_nc[:, :] = latarray_t


    f.close()


    Image.fromarray(adg443_t).save('./ocdata/outputfiles/A2006167181000adg443.tif')
    Image.fromarray(adg443qaaref_array_t).save('./ocdata/outputfiles/A2006167181000adg443ref.tif')

    print("file written done")

    # adg443tif = Image.open('./ocdata/A2006167181000adg443.tif')
    # adg443reftif = Image.open('./ocdata/A2006167181000adg443ref.tif')
    # adg443tifarray = np.array(adg443tif)
    # adg443reftifarray = np.array(adg443reftif)
    #
    # print("done")

















