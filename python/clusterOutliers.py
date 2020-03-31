import pandas as pd
import pickle
import keplerml
import km_outliers
import db_outliers
import quarterTools as qt
from bokeh.plotting import figure

from bokeh.layouts import row,gridplot,column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Button
from bokeh.events import Tap
from lightkurve import search_lightcurvefile

def import_gen(filedir="/home/dgiles/Documents/KeplerLCs/output/",suffix="_output.p",fitsdir="/home/dgiles/Documents/KeplerLCs/fitsFiles/",out_file_ext='.coo'):
    """
    Purpose:
        Creates a function to import quarters with common suffixes in a common directory (like "_output.csv" or "_PaperSample.csv"). 
    Args:
        filedir (str) - path to common directory
        fitsdir (str) - path to fitsfile directory (containing seperate directories for each quarter).
        suffix (str) - common suffix for output files
    Returns:
        lambda QN - a function which can be called with a specific Quarter specifified as a string.
        
    Example:
        importer = importGen(filedir='./output/',suffix='_output.p')
        Q1_cluster_object = importer('Q1')
        Q1_cluster_object is the clusterOutlier object for the Quarter 1 output features.
    """
    return lambda QN: clusterOutliers(filedir+QN+suffix,fitsdir+QN+"fitsfiles",output_file=filedir+QN+out_file_ext)

def load_coo(path_to_coo):
    with open(path_to_coo,'rb') as file:
        coo_file = pickle.load(file)
    return coo_file

class clusterOutliers(object):
    def __init__(self,feats,fitsDir,output_file='out.coo'):
        # feats may be provided as the dataframe itself or a filepath to a pickle or csv file containing a single dataframe.
        # the following interprets the form of feats given and defines self.data as the dataframe of the data.
        if type(feats) == pd.core.frame.DataFrame:
            self.data=feats
        else:
            try:
                with open(feats,'rb') as file:
                    self.data = pickle.load(file)
            except:
                try:
                    self.data = pd.read_csv(feats,index_col=0)
                except:
                    print("File format not recognized. Feature data must be stored as a dataframe in either a pickle or a csv.")
            assert type(self.data) == pd.core.frame.DataFrame, 'Feature data must be formatted as a pandas dataframe.'
        self.feats = self.data # self.feats is an alias for the data.
        self.scaled_data = qt.data_scaler(self.data)
        if fitsDir[-1]=="/":
            self.fitsDir = fitsDir
        else:
            self.fitsDir = fitsDir+"/"
        self.output_file = output_file
        self.files = self.data.index # A list of all files.
        # Initializing the data and files samples with 1000 entries.
        self.sample(
            1000,
            df='self',
            tabby=False,
            replace=True,
            rs=42) # Initializes self.dataSample and self.filesSample
        # Storing all reductions related to this object's data in its own reductions dictionary.
        self.reductions = dict()
        self.pca_red() # Initializes with pca reduction for plotting
    def sample(self, numLCs=10000, df='self',tabby=True, replace=True,rs=False):
        """
        Args:
            numLCs (int) - size of sample
            df ('self' or pd.DataFrame) - if self, will sample the object, if given a dataframe, will sample the given dataframe
            tabby (boolean) - if true, will ensure Boyajian's star is part of the sample.
            replace (boolean) - if true will replace the existing self.dataSample (used primarily for visualization)
            rs (boolean or int) - if int, will provide a set random state for the sample.
        Returns:
            sample (pd.DataFrame) - a randomly sampled subset from the givne dataframe
        """
        if type(df)==str:
            df = self.data
        
        assert (numLCs < len(df.index)),"Number of samples greater than the number of files."
        
        if type(rs)==int:
            sample = df.sample(n=numLCs,random_state=rs)
        else:
            sample = df.sample(n=numLCs)
        
        if tabby:
            if not sample.index.str.contains('8462852').any():
                sample = sample.drop(sample.index[0])
                sample = sample.append(self.data[self.data.index.str.contains('8462852')])
            
        if replace:
            self.dataSample = sample
            self.filesSample = sample.index          
        return sample
    
    def km_out(self,df='self',k=1):
        if type(df)==str:
            df = self.dataSample.iloc[:,0:60]
        labels = km_outliers.kmeans_w_outliers(df,k)
        self.KM_labels = labels
        return labels
    
    def db_out(self,df='self',neighbors=4,check_tabby=False,verbose=True):
        if type(df)==str:
            df = self.dataSample.iloc[:,0:60]
        labels = db_outliers.dbscan_w_outliers(data=df,min_n=neighbors,check_tabby=check_tabby,verbose=verbose)
        self.DB_labels = labels
        return labels
    
    def pca_red(self,df='self',red_name='PCA90',var_rat=0.9,scaled=False,verbose=True):
        if type(df)!=pd.core.frame.DataFrame:
            df = self.scaled_data
        pcaRed = qt.pca_red(df,var_rat,scaled,verbose)
        if type(df)!=pd.core.frame.DataFrame:
            self.reductions[red_name]=pcaRed
        return pcaRed
    
    def save(self,of=False):
        """
        Pickles the whole object
        Args:
            of (str) - Defaults to out.coo. File to save object to. Using .coo to demarcate Cluster-Outlier Objects. 
        """
        
        if type(of)!=str:
            of=self.output_file
        else:
            self.output_file=of
        try:
            with open(of,'wb') as file:
                pickle.dump(self,file)  
        except:
            print("Something went wrong, check output file path.")

    def plot(self,df=False,pathtofits=False,clusterLabels='db_out',reduced=False):
        if type(df)!=pd.core.frame.DataFrame:
            df = self.data
        if reduced:
            data = df.iloc[:,[0,1]]
        else:
            dataReduced = qt.pca_red(df)
            data = dataReduced.iloc[:,[0,1]]
            
        if type(pathtofits)!=str:
            pathtofits = self.fitsDir
            
        qt.interactive_plot(data,pathtofits,clusterLabels)
        
    def interactive_viz(self,Quarter,reduction):
        data = self.data

        
        red=self.reductions[reduction]
        x = red.iloc[:,0]
        y = red.iloc[:,1]
        files = self.data.index
        labels = np.array(['KIC '+str(int(lc[4:13])) for lc in files])
        
        ##### Bokeh Setup #####
        # Tools to use
        TOOLS="pan,wheel_zoom,reset,tap,box_select,poly_select"
        # Bokeh data
        # create a column data source for the plots to share
        s1 = ColumnDataSource(data={
            'x'     : x,
            'y'     : y,
            'desc'  : labels,
            'files' : files
        }) 

        s2=ColumnDataSource(data={
            't'  : [],
            'nf' : []
        })

        # Set up widgets
        button = Button(label='Plot Selected')
        # Set up callbacks
        def update_plot():
            try:
                inds = [s1.selected.indices[0]] # plotting multiple light curves is not supported at this time
            except:
                inds = [0] # if no point is selected, default behavior
            lcs = list(s1.data['desc'][inds])
            for i,ind in enumerate(inds): # framework to plot multiple lightcurves, not yet implemented
                lc = lcs[i]

                # download Kepler lighcurve via Lightkurve
                lcf = search_lightcurvefile(lc, mission="Kepler", quarter=Quarter).download()
                # use the normalized PDCSAP flux 
                nlc = lcf.PDCSAP_FLUX.normalize()
                t = nlc.time
                nf = nlc.flux

                s2.data = {'t':t,'nf':nf}
                plc.line('t','nf',source=s2)

        button.on_click(update_plot)

        # Set up layouts and add to document
        inputs = column(button)

        # create a new plot and add a renderer
        left = figure(tools=TOOLS, plot_width=1000, plot_height=600, title=None)
        left.circle('x', 'y', source=s1)
        left.on_event(Tap,update_plot)
        # Planning to incorporate a detailed view of the cluster center on a right plot in the future
        # create a plot for the light curve and add a line renderer
        plc = figure(tools=TOOLS,plot_width=1000,plot_height=200,title=None)
        plc.line('t','nf',source=s2)
        update_plot()

        #p = gridplot([[left,right]]) #future planning for multiple reductions
        layout = column(inputs,left, plc)
        return layout