import matplotlib
matplotlib.use('TkAgg')
import sys
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import colors
import matplotlib.cm as cmx
import matplotlib.gridspec as gridspec
# cuda needs cudatoolkit 9.0, 10.0 is incompatible.
from numba import cuda
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import astropy.io.fits as fits
from astroquery.mast import Observations # how to remotely access Kepler data from Mast
import shutil # to delete downloaded fits files once done

import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

###############################
### Functions for importing ###
###############################
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

def get_kepler_lcs(filenames):
    """
    Purpose:
        Downloads lightcurves
    Args:
        filenames - list of full filenames
    returns:
        paths_to_files - list containing the full paths to downloaded lightcurves.
    """
    obj_ids = [i[:13] for i in filenames]
    keplerObs = Observations.query_criteria(target_name=obj_ids, obs_collection='Kepler')
    keplerProds = Observations.get_product_list(keplerObs)
    yourProd = Observations.filter_products(keplerProds, extension=filenames)
    manifest = Observations.download_products(yourProd)
    
    """
    The process of downloading the files sorts them in numerical order messing up the 
    order, so I'm making the manifest a dataframe and sampling it
    one at a time from the filenames, not sure if there's a way to just sort the whole list
    based on the original order.
    """
    manifest = manifest.to_pandas(index='Local Path')
    paths_to_files = []
    for i,f in enumerate(filenames): # forcing the order to match given filenames list
        f_sampler = make_sampler([f]) # to find the right file from manifest
        filename = f_sampler(manifest).index[0] # full local filepath
        paths_to_files.append(filename)
        
    return paths_to_files

def rm_kepler_lcs():
    shutil.rmtree('./mastDownload')
    return

def make_sampler(inds=['8462852']): 
    """
    --------------------------------------------------------------------
    Purpose:
        Useful to generate samples across quarters with common sources 
        where data is contained as a Pandas dataframe, with indices set 
        to be arbitrary identifying labels (i.e. kplr008462852)
        
    Args:
        inds (Array of strings) - array of indices, as identifying 
            strings, to be pulled from a data frame, can be with or 
            without kplr prefix
    Returns:
        Function that will pull the data, indicated by inds, from a 
        dataframe df 
    
    To use:
        - Define array containing IDs of sources of interest as strings
        - Define a sample generator by calling make_sampler(inds=Array 
            of string IDs) 
        - Generate dataframe by calling new function.

    Example:
        # A sampler function for the specified sample
        tabby_sample = make_sampler(inds=['8462852']) 
        Q4_sample = tabby_sample(Q4.data) # The Quarter 4 data for the sample
        Q8_sample = tabby_sample(Q8.data) # The quarter 8 data for the sample
        etc.
    """
    return lambda df: df[df.index.str.contains('|'.join(inds))]

def subset_kepler_df(df,inds=[8462852]):
    """
    --------------------------------------------------------------------
    Purpose:
        A much more efficient sampler that requires inds to be fed in as 
        integers and the dataframe index to be the full identifying 
        file, i.e. 'kplr008462852_############llc.fits.'
    Args:
        df - dataframe to draw the subset from
        inds list(ints) - a list of Kepler IDs as integers
    Returns:
        subset_df (dataframe) - a pandas dataframe with only the 
            relevant data
    --------------------------------------------------------------------
    """
    df_copy = df.copy() # don't want to operate on the original dataframe
    df_copy['KIC'] = df_copy.index
    df_copy.index = [int(i[4:13]) for i in df_copy.index]
    subset_df = df_copy.loc[inds,:].copy()
    subset_df.set_index('KIC',inplace=True)
    return subset_df

#####################################
### Functions for data processing ###
#####################################
def data_scaler(df,nfeats=60):
    """
    Purpose:
        Data scaler this work, returns a dataframe w/ scaled features
    Args:
        df (pd.DataFrame) - dataframe with features to be scaled
        nfeats (int) - number of features in the dataframe, defaults to 60, should be changed if dataframe is a reduction.
    Returns:
        scaled_df (pd.DataFrame)- the dataframe of scaled data. If there were additional calculated columns in the original datafarme, they will not be present in the scaled dataframe.
    """
    data = df.iloc[:,0:nfeats]
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    scaled_df = pd.DataFrame(index=df.index,\
                             columns=df.columns[:nfeats],\
                             data=scaled_data)
    return scaled_df

def tsne_fit(data,perplexity='auto',scaled=False):
    """
    Performs a t-SNE dimensionality reduction on the data sample generated.
    Uses a PCA initialization and the perplexity given, or defaults to 1/10th the amount of data.
    
    Returns:
        tsneReduction (pd.DataFrame) - a dataframe containing the coordinates for the tsne reduction.
    """
    assert len(data)<20000, "Dataset too large, t-SNE cannot handle very large datasets. Try sampling a subset."
    if type(perplexity)==str:
        perplexity=len(data)/10
    if scaled:
        scaledData = data
    else:
        scaledData = data_scaler(data,nfeats=len(data.columns))
    
    tsne = TSNE(n_components=2,perplexity=perplexity,init='pca',verbose=True)
    fit=tsne.fit_transform(scaledData)
    # Goal is to minimize the KL-Divergence, so it can be useful to know what the resulting divergence is. 
    if sklearn.__version__ == '0.18.1':
        print("KL-Divergence was %s"%tsne.kl_divergence_ )
    tsneReduction = pd.DataFrame(index=data.index,data=fit,columns=['tsne_x','tsne_y'])
    return tsneReduction

def pca_red(data,var_rat=0.9,scaled=False,verbose=True):
    """
    Returns a pca reduction of the given data that explains a
    given fraction of the variance.
    Args:
        data (pd.DataFrame) - the data to be reduced (the features presumably), culled of irrelevant data (any calculated columns like scores or outlier labels)
        var_rat (float between 0 and 1) - the ratio of variance to be 
            explained by the transformed data
        scaled (boolean) - if False, data will be scaled using sklearn's preprocessing module StandardScaler
            
    Returns:
        reduced_data (pd.DataFrame) - dataframe of the reduced data with colums being the principle component inds
        prints the number of dimensions and the explained variance
    """
    if scaled:
        scaled_data = data
    else:
        if verbose: print("Scaling data using StandardScaler...")
        scaled_data = data_scaler(data,nfeats=len(data.columns))

    if verbose: print("Finding minimum number of dimensions to explain {:04.1f}% of the variance...".format(var_rat*100))
    
    for i in range(len(data.columns)):
        pca = PCA(n_components=i)
        pca.fit(scaled_data)
        if sum(pca.explained_variance_ratio_) >= var_rat:
            break
    if verbose:
        print("""
        Dimensions: {:d},
        Variance explained: {:04.1f}%
        """.format(i,sum(pca.explained_variance_ratio_)*100))
    
    reduced_data = pd.DataFrame(index=data.index, \
                                data=pca.transform(scaled_data))
    return reduced_data

##############################
### Functions for plotting ###
##############################
def colors_for_plot(inds,cmap='nipy_spectral'):
    """
    --------------------------------------------------------------------
    Args:
        inds (array of ints size n) - array to be converted into color 
            values
        cmap (str) - colormap of desired output
        
    Returns:
        colorVal (numpy array, size (n,4)) - numpy array containing 
            colors as rgba array or hex color values
    
    Note:    
        colormap 'color_blind' consists of 3 distinct colors 
            specifically chosen that show up well regardless of 
            color-sight
    --------------------------------------------------------------------
    """
    if cmap=='color_blind':
        # Custom set of colors to use that are color blind friendly
        color_blind = {0:"#009999", 1:"#FF6633", -1:"#333366"}
        colorVal = np.array([color_blind[i] for i in inds])
    else:
        cNorm  = colors.Normalize(min(inds), max(inds))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        colorVal = scalarMap.to_rgba(inds)
    return colorVal

def four_panel(data, title='Data',
               col_clus='Greens_d',shade_clus=False,
               col_out='Purples_d',shade_out=False,
               col_edge='Oranges_d',shade_edge=False,
               types='all_kde',alpha=1):
    """
    Four paneled figure separating out different designations in given data.
    Args:
        data (pandas dataframe) - dataframe to be plotted, must have columns 'db_out', 'tsne_x', and 'tsne_y'
        title (str) - Desired title for whole plot
        col_clus (str) - gradient for clustered kde
        shade_clus (boolean) - for kde plots, if False, only plots lines
        col_out (str) - see col_clus
        shade_out (boolean) - see shade_clus
        col_edge (str) - see col_clus
        shade_edge (boolean) - see shade_clus
        types (str) - What each panel should be, either KDE or scatter plot
        alpha (float, 0 to 1) - how opaque layers should be
    Returns:
        None - plots desired data
    """
    
    titlesize = 36
    ticksize = 30
    
    if types=='all_kde':
        types='kkkk'
    elif types =='all_scatter':
        types='ssss'
    comb_plot=types[3:] # the combined plot type (will be k (kde) or s (scatter))
    
    # If shading is enabled, reverse the default color gradient
    if shade_clus and col_clus=='Greens_d':
        col_clus='Greens'
    if shade_out and col_out=='Purples_d':
        col_out='Purples'
    if shade_edge and col_edge=='Oranges_d':
        col_edge='Oranges'
        
    sns.set_style('white')
    labels = data.db_out
    
    outliers = data[labels==-1]
    core = data[labels==0]
    edge = data[labels==1]
    
    # for scatter plots, creates common color map to be used in different plots w/ different data
    colorVal=colors_for_plot(labels,cmap='viridis')

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    plt.suptitle(title,fontsize=titlesize)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    panels = [ax1,ax2,ax3,ax4]
    
    # Panel 1
    panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,
          c=colorVal[data.db_out==0][core.tsne_x<40],t=types[0],ax=ax1,alpha=alpha)
    ax1.set_title('(a)',fontsize=titlesize,verticalalignment='bottom')
    ax1.tick_params(labelsize=ticksize,labelbottom='off')
    ax1.set_xlabel('')
    ax1.set_ylabel('t-SNE y',fontsize=titlesize)
    
    # Panel 2
    panel(outliers[outliers.tsne_x<40],cmap=col_out,shade=shade_out,
          c=colorVal[data.db_out==-1][outliers.tsne_x<40],t=types[1],ax=ax2,alpha=alpha)
    ax2.set_title('(b)',fontsize=titlesize,verticalalignment='bottom')
    ax2.tick_params(labelleft='off',labelbottom='off')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # Panel 3
    panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,
          c=colorVal[data.db_out==1][edge.tsne_x<40],t=types[2],ax=ax3,alpha=alpha)
    ax3.set_title('(c)',fontsize=titlesize,verticalalignment='bottom')
    ax3.tick_params(labelsize=ticksize)
    ax3.set_xlabel('t-SNE x',fontsize=titlesize)
    ax3.set_ylabel('t-SNE y',fontsize=titlesize)
    
    # Panel 4
    if comb_plot=='k':
        # All kde
        panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,t='k',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],cmap=col_out,shade=shade_out,t='k',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,t='k',ax=ax4,alpha=alpha)
    elif comb_plot=='s':
        # All scatter
        panel(core[core.tsne_x<40],c=colorVal[data.db_out==0][core.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],c=colorVal[data.db_out==-1][outliers.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],c=colorVal[data.db_out==1][edge.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        
    elif comb_plot=='ks':
        # Cluster kde, outliers scatter
        panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,t='k',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],c=colorVal[data.db_out==-1][outliers.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,t='k',ax=ax4,alpha=alpha)
    
    elif comb_plot=='kands':
        panel(core[core.tsne_x<40],c=colorVal[data.db_out==0][core.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],c=colorVal[data.db_out==-1][outliers.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],c=colorVal[data.db_out==1][edge.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        
        panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,t='k',ax=ax4,k_alpha=.5)
        panel(outliers[outliers.tsne_x<40],cmap=col_out,shade=shade_out,t='k',ax=ax4,k_alpha=.5)
        panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,t='k',ax=ax4,k_alpha=.5)
    #"""


    ax4.set_title('(d)',fontsize=titlesize,verticalalignment='bottom')
    ax4.tick_params(labelsize=ticksize,labelleft='off')
    ax4.set_xlabel('t-SNE x',fontsize=titlesize)
    ax4.set_ylabel('')
    
    ax1.set_xlim(ax4.get_xlim())
    ax1.set_ylim(ax4.get_ylim())
    ax2.set_xlim(ax4.get_xlim())
    ax2.set_ylim(ax4.get_ylim())
    ax3.set_xlim(ax4.get_xlim())
    ax3.set_ylim(ax4.get_ylim())
    
    return

def plot_lc(file,filepath='./fitsFiles/',local=True,c='blue',ax=False):
    """
    kid should be full id including time information
    Args:
        file (str) - filename starting with kplr ending in .fits
        filepath (str) - path to fits file
    Returns:
        None
    """
    if local:
        f=filepath+file
    else:
        filenames = [file]
        obj_ids = [i[:13] for i in tmp.index[:n]]
        keplerObs = Observations.query_criteria(target_name=obj_ids, obs_collection='Kepler')
        keplerProds = Observations.get_product_list(keplerObs)
        yourProd = Observations.filter_products(keplerProds, extension=file)
        manifest = Observations.download_products(yourProd)
        manifest = manifest.to_pandas(index='Local Path')
        f=manifest.index[0]

    if not ax:
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(111)
    
    t,nf,err=read_kepler_curve(f)

    x=t
    y=nf

    axrange=0.55*(max(y)-min(y))
    mid=(max(y)+min(y))/2
    yaxmin = mid-axrange
    yaxmax = mid+axrange
    if yaxmin < .95:
        if yaxmax > 1.05:
            ax.set_ylim(yaxmin,yaxmax)
        else:
            ax.set_ylim(yaxmin,1.05)
    elif yaxmax > 1.05:
        ax.set_ylim(.95,yaxmax)
    else:
        ax.set_ylim(.95,1.05)
        
    ax.set_xlim(min(x),max(x))
    color = c
    ax.plot(x, y, 'o',markeredgecolor='none', c=color, alpha=0.2)
    ax.plot(x, y, '-',markeredgecolor='none', c=color, alpha=0.7)
    #ax2.set_title(files[index][:13],fontsize = 20)
    ax.set_ylabel(r'$\frac{\Delta F}{F}$',fontsize=25)
    ax.tick_params(labelsize=20)
    #ax.set_title('KID'+str(int(file[4:13])),fontsize=25)
    
def four_Q_lc(kid,Qa,Qb,Qc,Qd):
    fig = plt.figure(figsize=(12,12))
    
    
    title_text = 'KID '+str(int(kid[4:13]))
    
    ax1 = fig.add_subplot(411)
    plt.title(title_text,fontsize=25)

    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    sampler =  make_sampler([kid])
    QaID = sampler(Qa.data)
    QbID = sampler(Qb.data)
    QcID = sampler(Qc.data)
    QdID = sampler(Qd.data)
    cs = colors_for_plot([QaID.db_out[0],QbID.db_out[0],QcID.db_out[0],QdID.db_out[0]],cmap='color_blind')
    
        
    plot_lc(QaID.index[0],filepath=Qa.fitsDir,c=cs[0],ax=ax1)
    plot_lc(QbID.index[0],filepath=Qb.fitsDir,c=cs[1],ax=ax2)
    plot_lc(QcID.index[0],filepath=Qc.fitsDir,c=cs[2],ax=ax3)
    plot_lc(QdID.index[0],filepath=Qd.fitsDir,c=cs[3],ax=ax4)
    ax4.set_xlabel('Time (Days)',fontsize=22)
    
    legend_cs = colors_for_plot([0,1,-1],cmap='color_blind')
    ax1.scatter([],[],c=legend_cs[-1],s=50,label='Outlier')
    ax1.scatter([],[],c=legend_cs[0],s=50,label='Core Cluster')
    ax1.scatter([],[],c=legend_cs[1],s=50,label='Edge Cluster')
    ax1.legend(loc='upper center',bbox_to_anchor=(0.5,1.05),ncol=3, fontsize=18)
    
    fig.tight_layout()
    
def plot_top_n(Q_coo,n=10,sortby='sAverage',top_n_df=False):
    """
    Purpose:
        Plots the top n outlier lightcurves for a given quarter. This assumes data is saved in the standard
        way for coo files, i.e. that feature data is saved as the attribute 'data' in a dataframe, that outlier scores are
        saved as 'scores', and that the indices of each are the full file names.
    Args:
        Q_coo (cluster outlier object) - The cluster outlier object containing the feature data and outlier scores attributes
        n (int) - number of plots to create of most outlying points
        sortby (optional, str) - how to sort scores, default is the sampled average of k=4 to 13
        top_n_df (optional, boolean) - whether or not to return a dataframe containing the features of the top outliers
    Returns:
        top_n_feats (optional, dataframe) - pandas dataframe containing the features of the top n outliers
    """

    tmp = Q_coo.scores.sort_values(by=sortby,ascending=False)
    filenames = list(tmp.index[:n])
    obj_ids = [i[:13] for i in tmp.index[:n]]
    keplerObs = Observations.query_criteria(target_name=obj_ids, obs_collection='Kepler')
    keplerProds = Observations.get_product_list(keplerObs)
    yourProd = Observations.filter_products(keplerProds, extension=filenames)
    manifest = Observations.download_products(yourProd)
    
    """
    The process of downloading the files sorts them in numerical order messing up the 
    order of most outlying to leas, so I'm making the manifest a dataframe and sampling it
    one at a time from the filenames, not sure if there's a way to just sort the whole list
    based on the original order.
    """
    manifest = manifest.to_pandas(index='Local Path')
    
    for i,f in enumerate(filenames): # forcing the order to match most to least outlying
        f_sampler = make_sampler([f]) # to find the right file from manifest
        filename = f_sampler(manifest).index[0] # full local filepath

        fig = plt.figure(figsize=(15,1))
        ax = fig.add_subplot(111)
        t,nf,err = read_kepler_curve(filename)
        ax.errorbar(t,nf,err)
        plt.title('KIC {}'.format(int(obj_ids[i][4:])))
        
    shutil.rmtree('./mastDownload') # removing the downloaded data
    
    if top_n_df:
        top_n_sampler = make_sampler(tmp[:n].index)
        top_n_feats = top_n_sampler(Q_coo.data)
        return top_n_feats
    else:
        return
    
def interactive_plot(df,pathtofits,clusterLabels):
    """
    Purpose:
        Plot an interactive representation of a set of labelled data.
        Clicking on a point in the scatterplot will plot the corresponding lightcurve below.
    Args:
        df (pd.DataFrame) - a dataframe containing the data to be plotted. The first 2 columns are plotted.
        pathtofits (str) - path to the fits file directory
        clusterLabels (str or list/list equivalent) - if str, will select the corresponding column from df.
            if list or list equivalent, then labels should be stored in order matching the dataframe
    """
    
    root = Tk.Tk()
    root.wm_title("Scatter")

    files = df.index
    if type(clusterLabels) == str:
        labels = df[clusterLabels]
    else:
        labels = clusterLabels

    colorVal = colors_for_plot(labels)

    data_out = df[labels==-1]
    files_out = data_out.index
    data_cluster = df[labels!=-1]
    files_cluster = data_cluster.index
    
    data = np.array(df.iloc[:,[0,1]])
    
    outX = data_out.iloc[:,0]
    outY = data_out.iloc[:,1]
    clusterX = data_cluster.iloc[:,0]
    clusterY = data_cluster.iloc[:,1]
    
    """--- Organizing data and Labels ---"""
    if df.index.str.contains('8462852').any():
        tabbyInd = list(df.index).index(df[df.index.str.contains('8462852')].index[0])            
    else:
        tabbyInd = 0
        
    fig = Figure(figsize=(20,10))
    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    # Toolbar to help navigate the data (pan, zoom, save image, etc.)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    gs = gridspec.GridSpec(2,100)
    with sns.axes_style("white"):
        # empty subplot for scattered data
        ax = fig.add_subplot(gs[0,:62])
        # empty subplot for lightcurves
        ax2 = fig.add_subplot(gs[1,:])
        # empty subplot for center detail
        ax3 = fig.add_subplot(gs[0,67:])
    @cuda.jit
    def distance_cuda(dx,dy,dd):
        bx = cuda.blockIdx.x # block in the grid
        bw = cuda.blockDim.x # size of a block
        tx = cuda.threadIdx.x # unique thread ID within a block
        i = tx + bx * bw
        if i>len(dd):
            return
        
        # d_ph is a distance placeholder
        d_ph = (dx[i]-dx[0])**2+(dy[i]-dy[0])**2
        dd[i]=d_ph**.5
        return
    
    def distances(pts,ex,ey):
        # Calculates distances between N points
        pts = np.array(pts)
        N=len(pts)
        # Allocate host memory arrays
        # Transpose pts array to n_dims x n_pts, each index of x contains all of a dimensions coordinates
        XT = np.transpose(pts)
        x = np.array(XT[0])
        x = np.insert(x,0,ex)
        y = np.array(XT[1])
        y = np.insert(y,0,ey)
        d = np.zeros(N)
        # Allocate and copy GPU/device memory
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)
        d_d = cuda.to_device(d)
        threads_per_block = 128
        number_of_blocks =int(N/128)+1 
        distance_cuda [number_of_blocks, threads_per_block](d_x,d_y,d_d)
        d_d.copy_to_host(d)
        return d[1:]   
    
    def calcClosestDatapoint(X, event):
        """Calculate which data point is closest to the mouse position.
        Args:
            X (np.array) - array of points, of shape (numPoints, 2)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        ex,ey = ax.transData.inverted().transform((event.x,event.y))
        
        #distances = [distance (XT[:,i], event) for i in range(XT.shape[1])]
        Ds = distances(X,ex,ey)
        return np.argmin(Ds)
    def drawData(index):
        # Plots the lightcurve of the point chosen
        ax2.cla()
        ### for fits files saved locally
        #f = pathtofits+df.index[index]
        #t,nf,err=read_kepler_curve(f)
                
        ### for fits files that need downloaded
        f = df.index[index] # full file name ****-****_llc.fits
        obj_id = f[:13] # observation id kplr********
        keplerObs = Observations.query_criteria(target_name=obj_id, obs_collection='Kepler')
        keplerProds = Observations.get_product_list(keplerObs)
        yourProd = Observations.filter_products(keplerProds, extension=f)
        manifest = Observations.download_products(yourProd)
        filename = manifest[0][0]            
        t,nf,err = qt.read_kepler_curve(filename)
        shutil.rmtree('./mastDownload') # removing the downloaded data
        
        x=t
        y=nf
        axrange=0.55*(max(y)-min(y))
        mid=(max(y)+min(y))/2
        yaxmin = mid-axrange
        yaxmax = mid+axrange
        if yaxmin < .95:
            if yaxmax > 1.05:
                ax2.set_ylim(yaxmin,yaxmax)
            else:
                ax2.set_ylim(yaxmin,1.05)
        elif yaxmax > 1.05:
            ax2.set_ylim(.95,yaxmax)
        else:
            ax2.set_ylim(.95,1.05)

        ax2.plot(x, y, 'o',markeredgecolor='none', c=colorVal[index], alpha=0.2)
        ax2.plot(x, y, '-',markeredgecolor='none', c=colorVal[index], alpha=0.7)
        #ax2.set_title(files[index][:13],fontsize = 20)
        ax2.set_xlabel('Time (Days)',fontsize=22)
        ax2.set_ylabel(r'$\frac{\Delta F}{F}$',fontsize=30)
        fig.suptitle(files[index][:13],fontsize=30)
        canvas.draw()
    def annotatePt(XT, index):
        """Create popover label in 3d chart
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        x2, y2 = XT[index][0], XT[index][1]
        # Either update the position, or create the annotation
        if hasattr(annotatePt, 'label'):
            annotatePt.label.remove()
            annotatePt.emph.remove()
        if hasattr(annotatePt, 'emphCD'):
            annotatePt.emphCD.remove()
        x_ax = ax.get_xlim()[1]-ax.get_xlim()[0]
        y_ax = ax.get_ylim()[1]-ax.get_ylim()[0]
        # Get data point from array of points X, at position index
        annotatePt.label = ax.annotate( "",
            xy = (x2, y2), xytext = (x2+.1*x_ax, y2+.2*y_ax),
            arrowprops = dict(headlength=20,headwidth=20,width=6,shrink=.1,color='black'))
        annotatePt.emph = ax.scatter(x2,y2,marker='o',s=50,c='orange')
        if files[index] in files_cluster:
            annotatePt.emphCD = ax3.scatter(x2,y2,marker='o',s=150,c='orange')
        else:
            annotatePt.emphCD = ax.scatter(x2,y2,marker='o',s=50,c='orange')
        canvas.draw()
    def onMouseClick(event, X):
        """Event that is triggered when mouse is clicked. Shows lightcurve for data point closest to mouse."""
        #XT = np.array(X.T) # array organized by feature, each in it's own array
        closestIndex = calcClosestDatapoint(X, event)
        drawData(closestIndex)
    def onMouseRelease(event, X):
        #XT = np.array(X.T)
        closestIndex = calcClosestDatapoint(X, event)
        annotatePt(X,closestIndex)
        #for centerIndex in centerIndices:
        #    annotateCenter(XT,centerIndex)
    def connect(X):
        if hasattr(connect,'cidpress'):
            fig.canvas.mpl_disconnect(connect.cidpress)
        if hasattr(connect,'cidrelease'):
            fig.canvas.mpl_disconnect(connect.cidrelease)
        connect.cidpress = fig.canvas.mpl_connect('button_press_event', lambda event: onMouseClick(event,X))
        connect.cidrelease = fig.canvas.mpl_connect('button_release_event', lambda event: onMouseRelease(event, X))
    def redraw():       
        # Clear the existing plots
        ax.cla()
        ax2.cla()
        ax3.cla()
        # Set those labels
        ax.set_xlabel("Reduced X",fontsize=18)
        ax.set_ylabel("Reduced Y",fontsize=18)
        # Scatter the data
        ax.scatter(outX, outY,c="red",s=30,alpha=.5)
        ax.hexbin(clusterX,clusterY,mincnt=5,bins="log",cmap="viridis",gridsize=35)
        hb = ax3.hexbin(clusterX,clusterY,mincnt=5,bins="log",cmap="viridis",gridsize=35)
        cb = fig.colorbar(hb)
        """
        ax.scatter(clusterX,clusterY,s=30,c='b')
        ax3.scatter(clusterX,clusterY)
        """
        ax3.set_title("Center Density Detail",fontsize=14)
        ax3.set_xlabel("Reduced X",fontsize=18)
        ax3.set_ylabel("Reduced Y",fontsize=18)
        ax.tick_params(axis='both',labelsize=18)
        ax2.tick_params(axis='both',labelsize=18)
        ax3.tick_params(axis='both',labelsize=18)
        #for centerIndex in centerIndices:
        #    annotateCenter(currentData1,centerIndex)
        if hasattr(redraw,'cidenter'):
                fig.canvas.mpl_disconnect(redraw.cidenter)
                fig.canvas.mpl_disconnect(redraw.cidexit)
        connect(data)
        annotatePt(data,tabbyInd)
        drawData(tabbyInd)
        #fig.savefig('Plots/Q16_PCA_kmeans/Tabby.png')
        canvas.draw()
    print("Plotting.")
    redraw()            
    def _delete_window():
        print("window closed.")
        root.destroy()
        sys.exit()
    root.protocol("WM_DELETE_WINDOW",_delete_window)
    root.mainloop()
    
###########################################################
### Class and methods for the weirdness profile/dossier ###
###########################################################
catalogs = {'koi_full':['./Variable Lists/list_koi_full.txt',',',2],
            'koi_confirmed':['./Variable Lists/list_koi_confirmed.txt',',',2],
            'koi_candidate':['./Variable Lists/list_koi_candidate.txt',',',2],
            'koi_fp':['./Variable Lists/list_koi_fp.txt',',',2],
            'EB':['./Variable Lists/list_EBs.csv',',',0],
            'HB':['./Variable Lists/list_kepler_heartbeats.txt',None,37],
            'flares':['./Variable Lists/kepler_solar_flares.txt',None,22],
            'no_signal':['./Variable Lists/list_kepler_nosig.txt',None,28],
            'periodic':['./Variable Lists/list_kepler_MSperiods.txt',None,32]}

class weirdnessProfile(object):
    def __init__(self,
                 KIC='8462852',
                 Qs=['Q4','Q8','Q11','Q16'],
                 analysis_path='/home/dgiles/Documents/KeplerLCs/output/Analysis/',
                 fitsDirPath='/home/dgiles/Documents/KeplerLCs/fitsFiles/',
                 verbose=True):
        """
        This object contains summary information about anomaly detection run on KIC lightcurves.
        Provide the KIC and the quarters for which analysis has been run, and this will create 
        a weirdness dossier, or profile for that KIC.
        Quarters must have been analyzed with an _analysis.csv file produced containing scores
        and 2 pca components for each KIC. 
        At the time of writing (3/8/19), only quarters 4, 8, 11, and 16 have been analyzed.
        
        This profiler cross references a number of catalogs whose members are stored in various
        formats. The 'catalogs' dictionary contains information on how to extract these filelists
        As a matter of convenience, copies of these analysis files have been created for each 
        catalog for each quarter rather than resample for each catalog.
        This will likely become unweildy as more catalogs and quarters are added (it already is),
        and methods will be updated, potentially impacting peformance.
        """
        if KIC[:3]=='KIC':
            self.KIC = str(int(KIC[3:]))
        elif KIC[:4]=='kplr':
            self.KIC = str(int(KIC[4:]))
        else:
            try:
                self.KIC = str(int(KIC))
            except:
                print("ID not recognized. Try without a prefix?")
        self.Qs = Qs
        self.analysis_path = analysis_path
        self.fitsDirPath = fitsDirPath
        self.sampler = lambda df: df[df.index.str.contains(self.KIC)]
        self.full_analysis_data = {Q:pd.read_csv(self.analysis_path+Q+'_analysis.csv',index_col=0) for Q in self.Qs}
        self.analysis_data = {Q:self.sampler(self.full_analysis_data[Q]) for Q in self.Qs}
        self.scores = self.scoreSummary()
        self.ranks = self.rankSummary()
        self.catalogs,self.ctlg_sampler = self.catalogCheck()
        self.ctlg_summary = self.catalogCompare()
        if verbose:self.printSummary()
        
    def scoreSummary(self):
        scores = pd.DataFrame()
        for Q in self.Qs:
            dftmp = self.analysis_data[Q].loc[:,['dist_score','PCA90_score','PCA95_score','PCA99_score']]
            dftmp.index = [Q]
            scores = scores.append(dftmp)
        return scores
    
    def rankSummary(self):
        # Returns the percentile of the object in relation to the rest of the quarter's data, generally more illuminating than the score
        full_ranks={}
        for Q in self.Qs:
            full_ranks[Q] = self.full_analysis_data[Q].loc[:,[]]
            full_ranks[Q]['Full'] = self.full_analysis_data[Q].dist_score.rank(ascending=False)
            full_ranks[Q]['Full_pctl'] = self.full_analysis_data[Q].dist_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
            full_ranks[Q]['PCA90'] = self.full_analysis_data[Q].PCA90_score.rank(ascending=False)
            full_ranks[Q]['PCA90_pctl'] = self.full_analysis_data[Q].PCA90_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
            full_ranks[Q]['PCA95'] = self.full_analysis_data[Q].PCA95_score.rank(ascending=False)
            full_ranks[Q]['PCA95_pctl'] = self.full_analysis_data[Q].PCA95_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
            full_ranks[Q]['PCA99'] = self.full_analysis_data[Q].PCA99_score.rank(ascending=False)
            full_ranks[Q]['PCA99_pctl'] = self.full_analysis_data[Q].PCA99_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
        self.full_ranks = full_ranks
        
        ranks = pd.DataFrame()
        for Q in self.Qs:
            dftmp = self.sampler(full_ranks[Q])
            dftmp.index = [Q]
            ranks = ranks.append(dftmp)
            
        return ranks
    
    def plotlcs(self):
        for Q in self.Qs:
            plot_lc(self.analysis_data[Q].index[0],self.fitsDirPath+Q+'fitsfiles/')
        return
    
    def pcaPlots(self):
        # Plots each quarter's data into the top 2 principle components from a PCA reduction
        for Q in self.Qs:
            x = self.full_analysis_data[Q].pca_x
            y = self.full_analysis_data[Q].pca_y
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            ax.scatter(x,y)
            ax.scatter(self.analysis_data[Q].pca_x,self.analysis_data[Q].pca_y)
            
    def catalogCheck(self):
        """
        Checks the available catalogs to see if the object is contained in them
        
        These catalogs are hardcoded in and based on lists compiled from various sources
        This isn't great practice, but these methods are created for a specific purpose,
        """
        in_catalogs = []
        ctlg_sampler = {}
        # Catalogs of individual classes are looped through here
        for ctlg in catalogs:
            import_ref = catalogs[ctlg]
            ctlg_list = np.genfromtxt(import_ref[0],delimiter=import_ref[1],skip_header=import_ref[2],usecols=(0),dtype=str)
            if self.KIC in ctlg_list:
                in_catalogs.append(ctlg)
                ctlg_sampler[ctlg] = make_sampler(ctlg_list)
                
        # Debosscher and SIMBAD have a bunch of different classifications, each need parsed out
        lines = np.genfromtxt('./Variable Lists/kepler_variable_classes.txt',dtype=str,skip_header=50)
        Debosscher_full_df = pd.DataFrame(data=lines[:,4],columns=["Class"],index=lines[:,0])
        if Debosscher_full_df.index.str.contains(self.KIC).any():
            deb_class = Debosscher_full_df.loc[self.KIC,'Class']
            ctlg_sampler['Deb_Class_%s'%deb_class]=make_sampler(Debosscher_full_df[Debosscher_full_df.Class==deb_class].index)
            in_catalogs.append('Deb_Class_%s'%deb_class)
            
        lines = np.genfromtxt('./Variable Lists/simbad.csv',delimiter=';',skip_header=7,dtype=str)[:,1:]
        simbad_full_df = pd.DataFrame(data=lines[:,1],columns=["Class"],index=lines[:,0])
        if simbad_full_df.index.str.contains(self.KIC).any():
            sim_class = simbad_full_df.loc[simbad_full_df.index.str.contains(self.KIC),'Class'][0]
            ctlg_sampler['SIMBAD_Class_%s'%sim_class]=make_sampler(simbad_full_df[simbad_full_df.Class == sim_class].index)
            in_catalogs.append('SIMBAD_Class_%s'%sim_class)
            
        return in_catalogs,ctlg_sampler
    
    def catalogCompare(self):
        # Produces summary dataframes for each catalog the object belongs to, 
        # the summary includes the mean score and the standard deviation of the scores of the catalog
        catalog_data_summary = {}
        for ctlg in self.catalogs:
            catalog_data_summary[ctlg] = pd.DataFrame()
            for Q in self.Qs:
                path = self.analysis_path+Q+'_'+ctlg+'_analysis.csv'
                ctlg_data = pd.read_csv(path,index_col=0)
                tmp_df = pd.DataFrame({'dist_score':ctlg_data.dist_score.mean(),
                                       'dist_std':ctlg_data.dist_score.std(),
                                       'PCA90_score':ctlg_data.PCA90_score.mean(),
                                       'PCA90_std':ctlg_data.PCA90_score.std(),
                                       'PCA95_score':ctlg_data.PCA95_score.mean(),
                                       'PCA95_std':ctlg_data.PCA95_score.std(),
                                       'PCA99_score':ctlg_data.PCA99_score.mean(),
                                       'PCA99_std':ctlg_data.PCA99_score.std()},
                                      index=[Q])
                catalog_data_summary[ctlg] = catalog_data_summary[ctlg].append(tmp_df)
        return catalog_data_summary
    def ctlgCmpViz(self,ctlg,scoreType='dist'):
        # Plots the score of the object against the score of the catalog with errorbars
        qs = [int(i[1:]) for i in self.scores.index]
        plt.scatter(qs,self.scores.loc[:,scoreType+'_score'])
        #plt.scatter(qs,self.ctlg_summary[ctlg].loc[:,scoreType+'_score'])
        plt.errorbar(qs,
                     self.ctlg_summary[ctlg].loc[:,scoreType+'_score'],
                     self.ctlg_summary[ctlg].loc[:,scoreType+'_std'])
        
    def printSummary(self):
        # Prints out the scores for the object in each given quarter, the catalogs it's contained in, 
        # and any significant score deviations from its identified catalogs
        print("Scores of KIC {}".format(self.KIC))
        display(self.scores)
        display(self.ranks)
        print("KIC{} contained in these catalogs: {}".format(self.KIC,self.ctlg_summary.keys()))
        for k in self.ctlg_summary.keys():
            for Q in self.Qs:
                if abs(self.ctlg_summary[k].loc[Q,'dist_score']-self.scores.loc[Q,'dist_score'])>self.ctlg_summary[k].loc[Q,'dist_std']:
                    print("Significant deviation (>1 sigma) from {} catalog in Quarter {}".format(k,Q))