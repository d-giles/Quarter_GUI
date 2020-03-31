import pickle
import sys
sys.path.append('python')
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.plotting import figure

from bokeh.layouts import row,gridplot,column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Button
from bokeh.events import Tap
from lightkurve import search_lightcurvefile

##### Data Import #####

with open('Q8_sample.coo','rb') as file:
    Q8_coo = pickle.load(file)

data = Q8_coo.data
Quarter = 8

pca=Q8_coo.reductions['PCA90']
x = pca.iloc[:,0]
y = pca.iloc[:,1]
files = data.index
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
curdoc().add_root(layout)