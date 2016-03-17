# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:48:48 2016

@author: ivaltchanov
"""


# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import math
from astropy.io import fits
import zipfile
import tarfile
from astropy.modeling import models, fitting
from astropy.stats import median_absolute_deviation
import copy 
import requests

# # Pointing offset corrector for SPIRE FTS sparse observations
# 
# ## Introduction
# 
# The SPIRE Fourier-Transform Spectrometer (FTS) has two wide frequency bands, SSW and SLW, which overlap in the region [] GHz. For a perfect point source the two bands should overlap nicely, with a ratio ~1, within the uncertainties. If there is a discrepancy in the overlap, then either the source is not point-like (extended or semi-extended) or the peak of the point source emission is not centred in the beam. It is also possible that a point-source is embedded in an extended background emission, which will also lead to a discontinueity in the overlap. In this case, the background subtraction must be done before running the Pointing Offset Corrector.
# 
# ## How the method works
# 
# If we have a source with surface brightness distribution $D_\nu(\Psi)$, and a beam $P_\nu(\Psi)$, where $\Psi$ is the spatial distribution, then we can write the forward coupling efficiency (see Ulich & Haas 1976) as:
# $$\eta_f(\nu)  =  \frac{\iint_{2\pi} P_\nu(\Psi-\Psi_0) D_\nu(\Psi) \mathrm{d}\Psi}{\iint_{2\pi} P_\nu(\Psi) \mathrm{d}\Psi},$$
# here $\Psi_0$ accounts for a pointing offset of the beam with respect to the source. Both $D_\nu(\Psi)$ and $P_\nu(\Psi)$ are normalised to have a peak of unity. 
# 
# Hence, the pointing corrected spectrum is (see Wu et al. 2013 for details):
# $$F_s = F_\mathrm{point} \eta_c(\nu,\Omega_\mathrm{source}) \frac{\Omega_\mathrm{source}}{\eta_f(\nu,\Omega_\mathrm{source})\Omega_\mathrm{beam}(\nu)},$$
# where $F_\mathrm{point}$ is the FTS pipeline provided final point-source calibrated spectrum in Jy,
# $\Omega_\mathrm{source}(\nu) = \iint_{2\pi} D_\nu(\Psi) \mathrm{d}\Psi$ is the source solid angle and
# $\Omega_\mathrm{beam}(\nu) = \iint_{2\pi} P_\nu(\Psi) \mathrm{d}\Psi$ is the beam solid angle, both in steradians. For point sources, the correction efficiency $\eta_c$ is assumed to be 1.
# 
# The beam distribution $P_\nu(\Psi)$ is presented in Makiwa et al. (2012) and is available as a calibration file.
# 
# The source distribution $D_\nu(\Psi)$ must be provided by the user, in this implementation we assume the source shape does not change with frequency.

# ## Extracting the beam maps from the SPIRE calibration tree
# 
# The SPIRE calibration tree is usually provided as a JAR file. Reading JAR files in python is like reading zip files. 

# In[7]:
def getSpireFtsLevel2(obsid, what='spss'):
    """
    Using the HTTP access to HAIO, retrieve all level-2 products in tar.gz file
    and extract only the requested fits file
    at then ind it puts the spectral structure in a dictionary
    """
    tarFile = "%i_level2.tar"%obsid
    haioRequest = "http://archives.esac.esa.int/hsa/aio/jsp/product.jsp?PROTOCOL=HTTP&OBSERVATION_ID=%i&PRODUCT_LEVEL=Level2"%obsid
    print ("Downloading level-2 data from the Herschel Science Archive. May take a while... be patient")
    r = requests.get(haioRequest)
    with open(tarFile, "wb") as tmp:
        tmp.write(r.content)
    # now read the downloaded tar file
    with tarfile.open(tarFile,'r') as tar:
        for member in tar.getmembers():
            if (what in member.name and '_spg_' in member.name):
                f=tar.extract(member)
                xx = fits.open(member.name)
    tar.close()
    spec = {}
    with xx as hdu:
        #
        for k in hdu:
            extname = k.name
            if ('S' in extname):
                spec[k.name] = {}
                spec[k.name]['wave'] = k.data["wave"]
                spec[k.name]['flux'] = k.data["flux"]
                spec[k.name]['fluxErr'] = k.data["error"]
    return spec
    
    
# In[2]:
def readSpireSparseSpec(spssFile):
    #
    spec = {}
    with fits.open(spssFile) as hdu:
        #
        for k in hdu:
            extname = k.name
            if ('S' in extname):
                spec[k.name] = {}
                spec[k.name]['wave'] = k.data["wave"]
                spec[k.name]['flux'] = k.data["flux"]
                spec[k.name]['fluxErr'] = k.data["error"]
    return spec
#
def plotSpireSparseSpec(specIn, onlyCentral=True):
    """
    """
    central = ['SSWD4','SLWC3']
    plt.figure(figsize=(8,5))
    for det in specIn.keys():
        if (det in central):
            plt.plot(specIn[det]['wave'],specIn[det]['flux'],'k-')
        else:
            plt.plot(specIn[det]['wave'],specIn[det]['flux'],'c-')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux density (Jy)')

# In[2]:

def getFtsBeam(jarFile, band='SSW'):
    # the FTS beam is a 3-D array, one frequency dimension and two spatial dimensions
    zf = zipfile.ZipFile(jarFile, 'r')
    try:
        lst = zf.infolist()
        for zi in lst:
            fn = zi.filename
            if fn.count('SCalSpecBeamProf_%s'%(band)):
                # QUESTION: can this be extracted on the fly? currently the files are extracted in the 
                # current folder
                bb = zf.extract(fn)
                beam = fits.open(bb)
                break
    finally:
        zf.close()
    #
    hh = beam['image'].header
    #
    nfq = hh['NAXIS3']
    fqStart = hh['CRVAL3']
    fqStep = hh['CDELT3']
    # the frequency axis
    fq = fqStart + np.arange(nfq)*fqStep
    #
    return fq,beam
# In[3]:

def fitBeamAndPlot(jarFile):
    """
    """
    fq = {}
    coef = math.sqrt(8.0*math.log(2.0))
    arc2sr = math.pow((3600.0*180.0/math.pi),2) # 1 sq.arcsec to steradian
    #
    beamFwhm = {}
    beamArea = {}
    plt.figure(figsize=(8,5))
    for arr in ["SSW","SLW"]:
        fq[arr], res = getFtsBeam(jarFile, band=arr)
        beamMap = res['image'].data
        beamArea[arr] = np.sum(beamMap,axis=(1,2))/arc2sr
        bshape = beamMap.shape
        nfq = bshape[0]
        #
        # now fit a 2-D Gaussian to the beam
        #
        x,y = np.mgrid[:bshape[1],:bshape[2]]
        # 
        # set up the initial values of the model
        sx = sy = 17.0/coef
        xc = yc = bshape[1]/2.0
        g_init = models.Gaussian2D(amplitude=1, x_mean=xc, y_mean=yc, x_stddev=sx, y_stddev=sy,theta=0)
        #
        fit_p = fitting.LevMarLSQFitter()
        #
        beamFwhm[arr] = np.zeros(nfq)
        for i in np.arange(nfq):
            p = fit_p(g_init, x, y, beamMap[i,:,:])
            # take the geometric mean of the fitted FWHM
            beamFwhm[arr][i] = coef*math.sqrt(p.x_stddev*p.y_stddev)
    # Plot the data with the best-fit FWHM
        plt.subplot(211)
        plt.plot(fq[arr], beamFwhm[arr], 'k-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('FWHM (arcsec)')
        plt.subplot(212)
        plt.plot(fq[arr], beamArea[arr], 'k-')    
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Solid angle (sr)')
        pass
    #
    return fq,beamArea,beamFwhm

# In[4]:

def makeSourceModel(model='gaussian', x_off=0.0, y_off = 0.0, x_fwhm=1.0, y_fwhm=1.0, theta=0.0, plotIt=0):
    #
    # create a source model brightness distribution
    # the image will be the same dimensions as the beamMap, this is hardcoded
    # x_off, y_off are the offsets from the centre of the beam map
    # the source model distributin is normalized to have integral of 1
    #
    coef = math.sqrt(8.0*math.log(2.0))
    sx = x_fwhm/coef
    sy = y_fwhm/coef
    # Hardcoded values!
    nx = ny = 257
    xc = yc = 128.5
    #
    g_model = models.Gaussian2D(amplitude=1, x_mean=xc-x_off, y_mean=yc-y_off, x_stddev=sx, y_stddev=sy,theta=0)
    x,y = np.mgrid[:nx,:ny]
    modelImage = g_model(x,y)
    modelImage = modelImage/np.sum(modelImage)
    if (plotIt):
        plt.figure(figsize=(8,5))
        plt.imshow(modelImage)
    return modelImage


# In[5]:

def calcCorrection(sourceModel,jarFile, plotIt=0):
    #
    # Calculates the forward coupling efficiency and the correction curve as explained in the introduction
    #
    #sshape = sourceModel.shape
    fq = {}
    xcorr = {}
    for arr in ["SSW","SLW"]:
        fq[arr], beamMap = getFtsBeam(jarFile, band=arr)
        nfq = len(fq[arr])
        beamImage = beamMap['image'].data
        corr = np.zeros(nfq)
        for i in np.arange(nfq):
            corr[i] = np.sum(beamImage[i,:,:] * sourceModel)
        xcorr[arr] = 1.0/corr
    if (plotIt):
        plt.figure(figsize=(8,5))
        for arr in ["SSW","SLW"]:
            plt.plot(fq[arr], xcorr[arr], 'k-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Correction')
    return  fq,xcorr
#

# In[6]:

def calcJump(spec, verbose=0):
    """
    Calculates the ratio in the overlap region
    spec must be a dictionary
    the jump is calculated only for the central detectors
    """
    overlap = [959.3,989.4] # GHz
    fx = {}
    for idet in ['SSWD4','SLWC3']:
        fqX = spec[idet]['wave']
        maskOver = (fqX >= overlap[0]) & (fqX <= overlap[1])
        fx[idet] = spec[idet]['flux'][maskOver]
    #
    ratio = fx['SLWC3']/fx['SSWD4']
    #
    # Find the median and Median Absolute Deviation of the ratio
    med = np.median(ratio)
    mad = median_absolute_deviation(ratio)
    if (verbose):
        print ("SLW/SSW median ratio in overlap = %5.3f +/- %5.3f"%(med,mad))
    return med, mad
    
def calcJumpTwo(freq, curve, verbose=0):
    """
    Calculates the ratio in the overlap region
    freq and curve  must be dictionaries with keys 'SSW' and 'SLW'
    """
    overlap = [959.3,989.4] # GHz
    fqL = freq['SLW']
    fqS = freq['SSW']
    slwOver = (fqL >= overlap[0]) & (fqL <= overlap[1])
    sswOver = (fqS >= overlap[0]) & (fqS <= overlap[1])
    ratio = curve['SLW'][slwOver]/curve['SSW'][sswOver]
    #
    # Find the median and Median Absolute Deviation of the ratio
    med = np.median(ratio)
    mad = median_absolute_deviation(ratio)
    if (verbose):
        print ("SLW/SSW median ratio in overlap = %5.3f +/- %5.3f"%(med,mad))
    return med, mad
    
#
#
# In[7]:
def generateGridOffsets(jarFile,modelName="gaussian", fwhm = 1,\
                        minOff=0.0, maxOff=10.0, ngrid=10, verbose=False, plotIt=False):
    '''
    Use calcCorrection() to generate the expected overlap ratios corresponding
    to a grid of pointing offsets
    - Input minOff and maxOff are currently hardcoded -
    Output:
        mediNorm - normalised grid of overlap ratios
        offGrid - grid of pointing offsets corresponding to mediNorm
    '''
    offGrid = np.linspace(minOff,maxOff,ngrid)
    # Grid range
    medi, medierr = np.zeros(ngrid), np.zeros(ngrid)
    # Errors not currently returned
    # Fill the grids
    for i in np.arange(ngrid):
        # Create the source model
        modeli = makeSourceModel(model='gaussian', x_off=offGrid[i], y_off = 0.0, x_fwhm=1.0, y_fwhm=1.0, theta=0.0, plotIt=0)
        fq, xcorr = calcCorrection(modeli,jarFile)
        # Find the corresponding median overlap ratio
        if (verbose):
            print ("Deriving the median overlap for offset %5.1f\""%offGrid[i])
        medi[i], medierr[i] = calcJumpTwo(fq,xcorr)
    # Normalise the median overlap ratio grid, so it is 1 at zero offset
    mediNorm  = medi[0]/medi
    if (plotIt):
        plt.figure(figsize=(8,5))
        plt.plot(offGrid, mediNorm, 'ko-')
        plt.xlabel('Offset (arcsec)')
        plt.ylabel('Median jump between bands')
    #     
    return [mediNorm,offGrid]
#
# now let's do it for both FTS arrays
#
# In[7]:

#
# the jar file is also available for download at 
#     ftp://ftp.sciops.esa.int/pub/hsc-calibration/latest_cal_tree/spire_cal_14_3.jar
# 
jar_file = '/Users/ivaltchanov/Dropbox/Work/SPIRE/spire_cal_14_2.jar'
#
# now generate the grid of offset for the model
#
ratx, offx = generateGridOffsets(jar_file)
#
# read the FTS fits file and extract the spectra of the central detectors
#
#wdir = '/Users/ivaltchanov/Tmp/HerschelData/ivaltcha25093308/'
#spss = 'hspirespectrometer1342259588_a1060001_spg_HR_20spss_1457702494491.fits.gz'
#spec = readSpireSparseSpec(wdir + spss)
spec = getSpireFtsLevel2(1342259588)
#
# get the initial ratio of the two bands:
#
xmed, xmad = calcJump(spec,verbose=True)
#
# interpolate to find the offset from the grid
#
result = np.interp(xmed,ratx,offx)

print ("Derived offset from interpolation: %f arcsec"%result)
plt.figure(figsize=(8,5))
plt.plot(offx, ratx, 'ko-')
plt.plot(result,xmed,'ro')
plt.xlabel('Offset (arcsec)')
plt.ylabel('Median jump between bands')
# In[ ]:
#
# now correct with the derived offset
#
myModel = makeSourceModel(model='gaussian', x_off=result)
freq, correc = calcCorrection(myModel,jar_file, plotIt=0)
#
specCorr = copy.deepcopy(spec)
central = ['SSWD4','SLWC3']
plt.figure(figsize=(8,5))
for idet in central:
    arr = idet[0:3]
    tmp = np.interp(spec[idet]['wave'].data,freq[arr],correc[arr])
    specCorr[idet]['flux'] = spec[idet]['flux']*tmp
    plt.plot(spec[idet]['wave'].data, spec[idet]['flux'].data, 'r-')
    plt.plot(specCorr[idet]['wave'], specCorr[idet]['flux'], 'g-')
    pass
#
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux Density (Jy)')
