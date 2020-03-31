import os
import sys
assert sys.version_info >= (3,0)
import numpy as np 
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
from scipy import stats
from multiprocessing import Pool,cpu_count
from datetime import datetime
import astropy.io.fits as fits
import pickle
from numba import njit
from datetime import datetime

def fl_files(fl):
    """
    Returns an array with the files in the given filelist
    """
    with open(fl,'r') as file:
        files = file.readlines()
    return files

def fl_files_w_path(fl,fitsDir='./fitsFiles/',fl_as_array=False):
    """
    Returns an array with the files to be processed
    in the given filelist with the given path.
    
    If filelist given as array, it is assumed that the entire array needs processed. Whereas if a filelist
    is given as a txt file, the following looks for a completed filelist as well to avoid reprocessing.
    """
    if fl_as_array:
        files = np.char.array(fl)
        
    else:
        with open(fl,'r') as file:
            files = file.readlines()
        fl_c = fl.replace('.txt',"")+"_completed.txt"
        if os.path.isfile(fl_c):
            """             
            While it seems roundabout to convert to empty pandas dataframes, it's about 20% faster
            than using numpy arrays as follows.

            unique,counts = np.unique(np.concatenate([files,completed]),return_counts=True)
            files = unique[counts==1]
            """
            completed = fl_files(fl_c)  

            dff = pd.DataFrame(index = files)
            dfc = pd.DataFrame(index = completed)
            df = dff.append(dfc)
            df = df[~df.index.duplicated(keep=False)]
            files = np.char.array(files) # allows appending the fits directory later

        else:
            files = np.char.array(files) # char.array allows appending the fits directory later
            fcreate = open(fl_c,'a')    # Creates a filelist to keep track of processed files.
            fcreate.close()
        
    return fitsDir+files

def read_kepler_curve(file):
    """
    Given the path of a fits file, this will extract the light curve and normalize it.
    """
    lc = fits.getdata(file)
    t = lc.field('TIME')
    f = lc.field('PDCSAP_FLUX')
    err = lc.field('PDCSAP_FLUX_ERR')
    
    err = err[np.isfinite(t)&np.isfinite(f)]
    f_copy = f[np.isfinite(t)&np.isfinite(f)]
    t = t[np.isfinite(t)&np.isfinite(f)]
    f = f_copy
    
    err = err/np.median(f)
    nf = f / np.median(f)
    
    return t, nf, err

def recover(fl,temp_file,fl_as_array=False):
    """
    Purpose:
        In the event that processing lightcurves fails in the middle, 
        this will read in the failsafe tmp file, evaluate which files were successfully processed, 
        and modify the files list to be processed.

        Removes files already processed from files array and
        removes the original filelist if all files have been processed.

        Each lightcurve appends the pickle file, so this could take awhile, but it's probably (no basis for this statement) faster than reprocessing.
    Args:
        fl (str,list,np.array) - filelist for processing. If a string, should be the path to the filelist containing each file name on a seperate line.
        temp_file - filepath to the temp file generated as a failsafe
        fl_as_array (boolean) - true if the filelist is given as a list or numpy array 
    Returns:
        files - an array of files remaining to be processed
        df - a datUnpicklingErroraframe of successfully processed files
        """
    if fl_as_array:files = fl
    else: 
        with open(fl,'r') as file:
            files = file.readlines()
    
    df = pd.DataFrame()
    with open(temp_file,'rb') as fr:
        try:
            while True:
                df = df.append(pickle.load(fr))
        except EOFError:
            pass
    pickle.dump(df,open(temp_file,'wb')) # resave the temp file as a singular entry for easier processing later if need be.
    """
    Dropping files that have already been processed from the files 
    """
    # Create a copy of df to manipulate
    dfc = df.copy()
    # create an empty dataframe with all the file names as indices (not just completed)
    dff = pd.DataFrame(index = files)
    dfc = dfc.append(dff)
    files = np.array(dfc[~dfc.index.duplicated(keep=False)].index)
    
    with open(fl.replace('.txt',"")+"_completed.txt",'a') as completed:
        completed.writelines(df.index)

    return files, df

@njit(parallel=True)
def easy_feats(t,nf,err):
    nf_mean = np.mean(nf)
    nf_med = np.median(nf)
    stds = np.std(nf) #f6
    meanmedrat = nf_mean / nf_med # F2
    varss = np.var(nf) # F4
    coeffvar = stds/nf_mean #F5

    posthreshold = nf_mean+4*stds
    negthreshold = nf_mean-4*stds

    numout1s = len(nf[np.abs(nf-nf_mean)>stds])
    numposoutliers = len(nf[nf>posthreshold])
    numnegoutliers = len(nf[nf<negthreshold])

    numoutliers=numposoutliers+numnegoutliers #F10

    mad = np.median(np.abs(nf-nf_med))

    # delta nf/delta t
    slopes = (nf[1:]-nf[:-1])/(t[1:]-t[:-1])
    meanslope = np.mean(slopes) #F12

    # Separating positive slopes and negative slopes
    # Should both include the 0 slope? It doesn't matter for calculating the means later on...
    pslope = slopes[slopes>=0]
    nslope = slopes[slopes<=0]
    # Looking at the average (mean) positive and negative slopes
    if len(pslope)==0:meanpslope=0
    else:meanpslope=np.mean(pslope) #F15

    if len(nslope)==0:meannslope=0
    else:meannslope=np.mean(nslope) #F16

    # Quantifying the difference in shape.
    # if meannslope==0 (i.e., if there are no negative slopes), g_asymm is assigned a value of 10
    # This value is chosen such that 
    # a) it is positive (where g_asymm is inherently negative), 
    # b) it is a factor larger than a random signal would produce (roughly equal average of positive and negative slopes -> g_asymm=-1)
    # c) it is not orders of magnitude larger than other data, which would affect outlier analysis
    if meannslope==0:g_asymm = 10
    else:g_asymm=meanpslope / meannslope #F17

    # Won't this be skewed by the fact that both pslope and nslope have all the 0's? Eh
    if len(nslope)==0:rough_g_asymm=10
    else:rough_g_asymm=len(pslope) / len(nslope) #F18

    # meannslope is inherently negative, so this is the difference btw the 2
    diff_asymm=meanpslope + meannslope #F19
    
    absslopes = np.abs(slopes)
    meanabsslope=np.mean(absslopes) #F21
    varabsslope=np.var(absslopes) #F22
    varslope=np.var(slopes) #F23

    # secder = Second Derivative
    # Reminder for self: the slope is "located" halfway between the flux and time points, 
    # so the delta t in the denominator is accounting for that.
    # secder = delta slopes/delta t, delta t = ((t_j-t_(j-1))+(t_(j+1)-t_j))/2
    # secder=[(slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2) for j in range(1, len(slopes)-1)]
    # after algebraic simplification:
    secder = 2*(slopes[1:]-slopes[:-1])/(t[1:-1]-t[:-2])

    # abssecder=[abs((slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2)) for j in range (1, len(slopes)-1)]
    # simplification:

    abssecder=np.abs(secder)
    absmeansecder=np.mean(abssecder) #F24

    if len(pslope)==0:pslopestds=0
    else:pslopestds=np.std(pslope)

    if len(nslope)==0:
        nslopesstds=0
        stdratio=10
    else:
        nslopestds=np.std(nslope)
        stdratio=pslopestds/nslopestds

    sdstds=np.std(secder)
    meanstds=np.mean(secder)

    num_pspikes=len(slopes[slopes>=meanpslope+3*pslopestds]) #F25
    num_nspikes=len(slopes[slopes<=meannslope-3*nslopestds]) #F26

    # 5/30/18, discovered a typo here. meanslope was missing an 'n', i.e. all data
    # processed prior to this date has num_nspikes defined as meanslope-3*nslopestds
    # which will overestimate the number of negative spikes since meanslope is inherently
    # greater than meannslope.

    num_psdspikes = len(secder[secder>=meanstds+4*sdstds]) #F27
    num_nsdspikes = len(secder[secder<=meanstds-4*sdstds]) #F28
    if nslopestds==0:
        stdratio=10
    else:
        stdratio = pslopestds / nslopestds #F29

    # The ratio of postive slopes with a following postive slope to the total number of points.
    pairs = np.where((slopes[1:]>0)&(slopes[:-1]>0))[0] # where positive slopes are followed by another positive slope
    pstrend=len(pairs)/len(slopes) #F30

    plusminus = np.where((slopes[1:]<0)&(slopes[:-1]>0))[0]
    num_pm = len(plusminus)

    # This looks up the local maximums. Adds a peak if it's the largest within 10 points on either side.
    # Q: Is there a way to do this and take into account drastically different periodicity scales?

    naivemax,nmax_times,nmax_inds = [],[],[]
    naivemins,nmin_times,nmin_inds = [],[],[]
    for j in range(len(nf)):
        nfj = nf[j]
        if j-10<0:
            jmin=0
        else:
            jmin=j-10
        if j+10>len(nf)-1:
            jmax=len(nf-1)
        else:
            jmax=j+10

        max_nf=nf[jmin]
        min_nf=nf[jmin]
        for k in range(jmin,jmax):
            if nf[k] >= max_nf:
                max_nf = nf[k]
            elif nf[k] <= min_nf:
                min_nf = nf[k]

        if nf[j]==max_nf:
            if len(nmax_inds)>0:
                if j-nmax_inds[-1]>10:
                    naivemax.append(nf[j])
                    nmax_times.append(t[j])
                    nmax_inds.append(j)
            else:
                naivemax.append(nf[j])
                nmax_times.append(t[j])
                nmax_inds.append(j)    
        elif nf[j]==min_nf:
            if len(nmin_inds)>0:
                if j-nmin_inds[-1]>10:
                    naivemins.append(nf[j])
                    nmin_times.append(t[j])
                    nmin_inds.append(j)
            else:
                naivemins.append(nf[j])
                nmin_times.append(t[j])
                nmin_inds.append(j)
            
    naivemax = np.array(naivemax)
    nmax_times = np.array(nmax_times)
    nmax_inds = np.array(nmax_inds)
    naivemins = np.array(naivemins)
    nmin_times = np.array(nmin_times)
    nmin_inds = np.array(nmin_inds)
    
    len_nmax=len(naivemax) #F33
    len_nmin=len(naivemins) #F34

    ppslopes = np.abs(naivemax[1:]-naivemax[:-1])/(nmax_times[1:]-nmax_times[:-1])

    if len(ppslopes)==0:
        ptpslopes = 0
    else:
        ptpslopes=np.mean(ppslopes) #F36

    maxdiff = nmax_times[1:]-nmax_times[:-1]
    
    
    emin = naivemins[::2] # even indice minimums
    omin = naivemins[1::2] # odd indice minimums
    meanemin = np.mean(emin)
    if len(omin)==0:
        meanomin=0
    else:
        meanomin = np.mean(omin)
    oeratio = meanomin/meanemin #F42

    #measures the slope before and after the maximums
    # reminder: 1 less slope than flux, slopes start after first flux
    # slope[0] is between flux[0] and flux[1]
        # mean of slopes before max will be positive
        # mean of slopes after max will be negative

    nmax_inds_subset = nmax_inds[(nmax_inds>5)&(nmax_inds<len(slopes)-5)]
    flatness = np.zeros(len(nmax_inds_subset))
    for i,j in enumerate(nmax_inds_subset):
        flatness[i] = np.mean(slopes[j-6:j-1])-np.mean(slopes[j:j+5])

    if len(flatness)==0: flatmean=0
    else: flatmean = np.mean(flatness) #F55

    # measures the slope before and after the minimums
    # trying flatness w slopes and nf rather than "corr" vals, despite orig def in RN's program
      # mean of slopes before min will be negative
      # mean of slopes after min will be positive


    nmin_inds_subset = nmin_inds[(nmin_inds>5)&(nmin_inds<len(slopes)-5)]
    tflatness = np.zeros(len(nmin_inds_subset))
    for i,j in enumerate(nmin_inds_subset):
        tflatness[i] = -np.mean(slopes[j-6:j-1])+np.mean(slopes[j:j+5])

    # tflatness for mins, flatness for maxes
    if len(tflatness)==0: tflatmean=0
    else: tflatmean = np.mean(tflatness) #F56

    # reminder: 1 less second derivative than slope (2 less than flux). secder starts after first slope.
    # secder[0] is between slope[0] and slope[1], centered at flux[1]

    nmax_inds_subset = nmax_inds[(nmax_inds>5)&(nmax_inds<len(secder)-5)]
    
    roundness = np.zeros(len(nmax_inds_subset))
    for i,j in enumerate(nmax_inds_subset):
        roundness[i] = np.mean(secder[j-6:j+6])*2
        
    if len(roundness)==0: roundmean=0
    else: roundmean = np.mean(roundness) #F57

    nmin_inds_subset = nmin_inds[(nmin_inds>5)&(nmin_inds<len(secder)-5)]
    troundness = np.zeros(len(nmin_inds_subset))
    for i,j in enumerate(nmin_inds_subset):
        troundness[i] = np.mean(secder[j-6:j+6])*2

    if len(troundness)==0: troundmean=0
    else: troundmean = np.mean(troundness) #F58

    if troundmean==0 and roundmean==0: roundrat=1
    elif troundmean==0 and roundmean>0: roundrat=123456
    elif troundmean==0 and roundmean<0: roundrat=-123456
    else: roundrat = roundmean / troundmean #F59

    if flatmean==0 and tflatmean==0: flatrat=1
    elif tflatmean==0: flatrat=-10 #Flatness ratio is (almost) inherently positive has an average value of 1. If there are no minima, a flatness ratio of -10 is assigned. See reasoning for g_asymm.
    else: flatrat = flatmean / tflatmean #F60"""
        
    return meanmedrat, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, mad, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, num_pm, len_nmax, len_nmin, ptpslopes, oeratio, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat, naivemax, maxdiff

def fancy_feats(t,nf,err,naivemax,maxdiff):
    nf_mean = np.mean(nf)
    nf_med = np.median(nf)
    
    # fancy meaning I can't throw these under a jit decorator.
    longtermtrend = np.polyfit(t, nf, 1)[0] # Feature 1 (Abbr. F1) overall slope
    yoff = np.polyfit(t, nf, 1)[1] # Not a feature, y-intercept of linear fit
    skews = stats.skew(nf) # F3
    corrnf = nf - longtermtrend*t - yoff #this removes any linear trend of lc so you can look at just troughs
    
    kurt = stats.kurtosis(nf) # F11
    
    slopes = (nf[1:]-nf[:-1])/(t[1:]-t[:-1])

    # by looking at where the 99th percentile is instead of just the largest number,
    # I think it avoids the extremes which might not be relevant (might be unreliable data)
    # Is the miniumum slope the most negative one, or the flattest one? Answer: Most negative
    maxslope=np.percentile(slopes,99) #F13
    minslope=np.percentile(slopes,1) #F14

    #corrslopes (corrected slopes) removes the longterm linear trend (if any) and then looks at the slope
    corrslopes = (corrnf[1:]-corrnf[:-1])/(t[1:]-t[:-1])
    skewslope = stats.skew(slopes) #F20
    
    # Checks if the flux crosses the 'zero' line.
    zcrossind = np.where(corrnf[:-1]*corrnf[1:]<0)[0]
    num_zcross = len(zcrossind) #F31
    
    if len(naivemax)>2:
        mautocorrcoef = np.corrcoef(naivemax[:-1], naivemax[1:])[0][1] #F35
    else:
        mautocorrcoef = 0

    if len(maxdiff)==0:
        periodicity=0
        periodicityr=0
        naiveperiod=0
    else:
        periodicity=np.std(maxdiff)/np.mean(maxdiff) #F37
        periodicityr=np.sum(abs(maxdiff-np.mean(maxdiff)))/np.mean(maxdiff) #F38
        naiveperiod=np.mean(maxdiff) #F39
    if len(naivemax)==0:
        maxvars=0
        maxvarsr=0
    else:
        maxvars = np.std(naivemax)/np.mean(naivemax) #F40
        maxvarsr = np.sum(abs(naivemax-np.mean(naivemax)))/np.mean(naivemax) #F41

    # amp here is actually amp_2 in revantese
    # 2x the amplitude (peak-to-peak really)
    amp = np.percentile(nf,99)-np.percentile(nf,1) #F43
    normamp = amp / nf_mean #this should prob go, since flux is norm'd #F44
    
    autocorrcoef = np.corrcoef(nf[:-1], nf[1:])[0][1] #F54

    sautocorrcoef = np.corrcoef(slopes[:-1], slopes[1:])[0][1] #F55
    # ratio of points within one fifth of the amplitude to the median to total number of points 
    mbp = len(nf[(nf<=(nf_med+0.1*amp))&(nf>=(nf_med-0.1*amp))]) / len(nf) #F45

    f595 = np.percentile(nf,95)-np.percentile(nf,5)
    f1090 =np.percentile(nf,90)-np.percentile(nf,10)
    f1782 =np.percentile(nf, 82)-np.percentile(nf, 17)
    f2575 =np.percentile(nf, 75)-np.percentile(nf, 25)
    f3267 =np.percentile(nf, 67)-np.percentile(nf, 32)
    f4060 =np.percentile(nf, 60)-np.percentile(nf, 40)
    mid20 =f4060/f595 #F46
    mid35 =f3267/f595 #F47
    mid50 =f2575/f595 #F48
    mid65 =f1782/f595 #F49
    mid80 =f1090/f595 #F50 

    
    percentamp = max(np.abs(nf-nf_med)/nf_med) #F51

    magratio = (max(nf)-nf_med) / amp #F52
    
    return longtermtrend, skews, kurt, maxslope, minslope, skewslope, num_zcross, mautocorrcoef, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, amp, normamp, mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef
    
def feats(t,nf,err):
    """
    Purpose:
        Calculates the features determined to be useful for machine learning applications to photometric data.
        Splits the computation into 2 categories, those which can be easily decorated with the njit decorator, and those which can be otherwise optimized.
        
    Args:
        t (np.array, dtype np.float64) - time
        nf (np.array, dtype np.float32) - normalized flux
        err (np.array, dtype np.float32) - normalized error
    
    returns:
        ndata (list, dtype varied) - the features in order as specified in the feature key (external to this code)
    """
    # features that were jit-able
    meanmedrat, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, mad, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes, stdratio, pstrend, num_pm, len_nmax, len_nmin, ptpslopes, oeratio, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat, naivemax, maxdiff = easy_feats(t,nf,err)
    
    # features that were not easily jit-able
    longtermtrend, skews, kurt, maxslope, minslope, skewslope, num_zcross, mautocorrcoef, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, amp, normamp, mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef = fancy_feats(t,nf,err,naivemax,maxdiff)
    
    ndata = [longtermtrend, meanmedrat, skews, varss, coeffvar, stds, \
                 numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, \
                 maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, \
                 diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, \
                 num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, \
                 num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, \
                 periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, \
                 amp, normamp, mbp, mid20, mid35, mid50, \
                 mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, \
                 flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat]
    return ndata

def feature_calc(lc):
    """
    Purpose:
        This method is for use when processing multiple lightcurves. It takes a single lightcurve of data at a time, processes it, 
        dumps the results as an indexed dataframe into a tmp file as a failsafe, and returns the file name as well as the features in a list to be parsed.
        
    Args:
        lc (list) - a list containing:
            lc[0] (str) - nfile - the name of the file being processed
            lc[1] (list/np.array) - t - array of times for lightcurve
            lc[2] (list/np.array) - nf - array of normalized fluxes for lightcurve
            lc[3] (list/np.array) - err - array of normalized errors for lightcurve
    Note: requires a global variable "tmpfile" to be defined, this should be a string specifying where data should be temporarily stored.   
    Returns:
        nfile - the name of the file sans path
        ndata - the feature data
    """
    
    nfile,t,nf,err = lc[0],lc[1],lc[2],lc[3]

    try:
        ndata = feats(t,nf,err)
            
        fts = ["longtermtrend", "meanmedrat", "skews", "varss", "coeffvar", "stds", \
               "numoutliers", "numnegoutliers", "numposoutliers", "numout1s", "kurt", "mad", \
               "maxslope", "minslope", "meanpslope", "meannslope", "g_asymm", "rough_g_asymm", \
               "diff_asymm", "skewslope", "varabsslope", "varslope", "meanabsslope", "absmeansecder", \
               "num_pspikes", "num_nspikes", "num_psdspikes", "num_nsdspikes","stdratio", "pstrend", \
               "num_zcross", "num_pm", "len_nmax", "len_nmin", "mautocorrcoef", "ptpslopes", \
               "periodicity", "periodicityr", "naiveperiod", "maxvars", "maxvarsr", "oeratio", \
               "amp", "normamp", "mbp", "mid20", "mid35", "mid50", \
               "mid65", "mid80", "percentamp", "magratio", "sautocorrcoef", "autocorrcoef", \
               "flatmean", "tflatmean", "roundmean", "troundmean", "roundrat", "flatrat"]
        # A failsafe, dumps the data to a temp file in case of failure during processing of another lightcurve.
        # Reading in the temp file is a huge pain though.
        df = pd.DataFrame([ndata],index=[nfile[nfile.find('kplr'):]],columns=fts)
        try:
            with open(tmpfile,'ab') as f:
                pickle.dump(df,f)
        except:
            tmpfile = 'tmp_data.p'
            with open(tmpfile,'ab') as f:
                pickle.dump(df,f)
             
        return [nfile[nfile.find('kplr'):],ndata]
        
    except TypeError:
        kml_log = 'kml_log'
        os.system('echo %s ... TYPE ERROR >> %s'%(nfile.replace(fitsDir,""),kml_log))
        return 

def import_lcs(nfile):
    """
    Purpose: 
        Read in a lightcurve, wrapper for read_kepler_curve
    Args:
        nfile (string) - file including path of lightcurve (as a fits file)
    Returns:
        t (np.array) - time array
        nf (np.array) - normalized flux array
        err (np.array) - normalized error array
    """
    try:
        t,nf,err = read_kepler_curve(nfile)
        t,nf,err = t.astype(np.float64),nf.astype(np.float32),err.astype(np.float32)
        
    except TypeError as err:
        # Files can be truncated by the zipping process.
        print("%s. Try downloading %s again."%(err,nfile))
        return
    
    #features = featureCalculation(nfile,t,nf,err)
    return [nfile,t,nf,err]

def join_data(full_features_array):
    file = []
    data = []
    fts = ["longtermtrend", "meanmedrat", "skews", "varss", "coeffvar", "stds", \
           "numoutliers", "numnegoutliers", "numposoutliers", "numout1s", "kurt", "mad", \
           "maxslope", "minslope", "meanpslope", "meannslope", "g_asymm", "rough_g_asymm", \
           "diff_asymm", "skewslope", "varabsslope", "varslope", "meanabsslope", "absmeansecder", \
           "num_pspikes", "num_nspikes", "num_psdspikes", "num_nsdspikes","stdratio", "pstrend", \
           "num_zcross", "num_pm", "len_nmax", "len_nmin", "mautocorrcoef", "ptpslopes", \
           "periodicity", "periodicityr", "naiveperiod", "maxvars", "maxvarsr", "oeratio", \
           "amp", "normamp", "mbp", "mid20", "mid35", "mid50", \
           "mid65", "mid80", "percentamp", "magratio", "sautocorrcoef", "autocorrcoef", \
           "flatmean", "tflatmean", "roundmean", "troundmean", "roundrat", "flatrat"]
    for i in full_features_array:
        file.append(i[0])
        data.append(i[1])
    df = pd.DataFrame(data,index=file,columns=fts)
    return df

def features_from_filelist(fl,fitDir,of,fl_as_array=False, numCpus = cpu_count(), verbose=False, prime_feats=True,tmp_file='tmp_data.p'):
    """
    Purpose:
        This method calculates the features of the given filelist from the fits files located in fitsDir.
        All output is saved to a pickle file called tmp_data.p (or else whatever is specified).
            Run clean_up(filelist, fits/file/directory) and save_output('output/file/path') 
            to clean up the filelist (makes a completed filelist) and to save to the desired location. 
            Note: save_output() replaces tmp_data.p 
    Args:
        fl (string or list) - filelist as a path to a text file containing the files to be processed, one per line, or a list of the files
        fitDir (string) - path to the fits directory, if path is specified in the filelist, give this a value of ''
        of (string) - output file, the path to the file where the output should be saved as a pickle
        fl_as_array (boolean) - if True specifies that fl is given as a list, if False specifies that fl is a filepath
        numCpus (integer) - Number of cpus on which to process the files
        verbose (boolean) - if True will output progress statements
        tmp_file (string) - specifies where to save the temporary data. Useful if processing multiple sets of data in parallel.
    Notes:
        Multiprocessing is used for light curve import only at this stage. It was formerly used
            to improve feature calculation time, but using Numba for several features significantly
            improved the overall calculation time and multiprocessing for feature calculation 
            has not provided any additional speed up in conjunction with Numba.

    Returns:
        Pandas dataframe of output for all files in the filelist
    """
    startTime = datetime.now()
    global fitsDir
    fitsDir = fitDir
    global tmpfile
    tmpfile = tmp_file
    
    # files with path.
    if verbose:
        if not fl_as_array:print("Reading %s..."%fl)
    files = fl_files_w_path(fl,fitsDir,fl_as_array)
    
    if numCpus == -1:useCpus=cpu_count()-1
    else:useCpus = min([len(files),cpu_count()-1,max(numCpus,1)])
    
    if verbose:
        print("Using %s cpus to calculate features..."%useCpus)
        print("Importing {} lightcurves...".format(len(files)))
    
    # Importing all the lightcurves first
    p = Pool(useCpus)
    lcs = p.map(import_lcs,files)
    p.close()
    p.join()

    if verbose:print("Lightcurve import took {}".format(datetime.now()-startTime))

    fts = ["longtermtrend", "meanmedrat", "skews", "varss", "coeffvar", "stds", \
           "numoutliers", "numnegoutliers", "numposoutliers", "numout1s", "kurt", "mad", \
           "maxslope", "minslope", "meanpslope", "meannslope", "g_asymm", "rough_g_asymm", \
           "diff_asymm", "skewslope", "varabsslope", "varslope", "meanabsslope", "absmeansecder", \
           "num_pspikes", "num_nspikes", "num_psdspikes", "num_nsdspikes","stdratio", "pstrend", \
           "num_zcross", "num_pm", "len_nmax", "len_nmin", "mautocorrcoef", "ptpslopes", \
           "periodicity", "periodicityr", "naiveperiod", "maxvars", "maxvarsr", "oeratio", \
           "amp", "normamp", "mbp", "mid20", "mid35", "mid50", \
           "mid65", "mid80", "percentamp", "magratio", "sautocorrcoef", "autocorrcoef", \
           "flatmean", "tflatmean", "roundmean", "troundmean", "roundrat", "flatrat"]
    df = pd.DataFrame(columns=fts)

    # Splitting up lcs into 10k lc chunks so that progress can be monitored
    nlcs = len(lcs)
    nchunks = int(nlcs/10000)
    len_last_chunk = nlcs%10000
    if verbose:print("Processing %s files..."%len(files))
    featsStartTime = datetime.now()
    if prime_feats == True:
        feats(lcs[0][1],lcs[0][2],lcs[0][3]) # Numba's njit needs primed unless already done
    for i in range(nchunks):
        chunkStartTime = datetime.now()
        # Method saves to tmp_data.p file after processing each lightcurve as a failsafe.
        full_features = [feature_calc(lc) for lc in lcs[i*10000:(i+1)*10000]]
        df = df.append(join_data(full_features))
        with open(tmpfile,'wb') as file:pickle.dump(df,file) # overwrites failsafe with single dataframe containing all processed data, easier to process if system failure later.
        del(full_features)
        
        if verbose:print("{}/{} completed. Time for chunk: {}".format((i+1)*10000,nlcs,datetime.now()-chunkStartTime))
    # last chunk
    chunkStartTime = datetime.now()
    full_features = [feature_calc(lc) for lc in lcs[nlcs-len_last_chunk:]]
    if verbose:print("{}/{} completed. Time for chunk: {}".format(nlcs,nlcs,datetime.now()-chunkStartTime))
    df = df.append(join_data(full_features))
    del(full_features)
    
    if verbose:
        print("Features have been calculated, total time to calculate features: {}".format(datetime.now()-featsStartTime))
        print("Saving output to %s"%of)
    with open(of,'wb') as file:pickle.dump(df,file)
    if verbose:print("Cleaning up...")
    os.remove(tmpfile)
    del(lcs,files,p,df)
    if verbose:print("Done.")
    
    if __name__=="__main__":
        return
    else:
        with open(of,'rb') as file:
            features = pickle.load(file)
        return features
    
if __name__=="__main__":
    """
    If this is run as a script, the following will parse the arguments it is fed, 
    or prompt the user for input.
    
    python keplerml.py path/to/filelist path/to/fits_file_directory path/to/output_file
    """
    # fl - filelist, a txt file with file names, 1 per line
    if sys.argv[1]:
        fl = sys.argv[1]
    else:
        fl = raw_input("Input path: ")

    if sys.argv[2]:
        fitsDir = sys.argv[2]
    else:
        fitsDir = raw_input("Fits files directory path: ")
    # of - output file
    if sys.argv[3]:
        of = sys.argv[3]
    else:
        of = raw_input('Output path: ')
        if of  == "":
            print("No output path specified, saving to output.csv in local folder.")
            of = 'output.csv'
    start = datetime.now()
    features_from_filelist(fl,fitsDir,of,verbose=True)
    print(datetime.now()-start)
    