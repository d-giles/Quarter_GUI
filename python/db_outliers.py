"""
DBSCAN Clustering
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def eps_est(data,n=4,verbose=True):
    """
    Minimum data size is 1000
    Methodology improvement opportunity: Better elbow locater
    """
    
    if verbose:print("Calculating nearest neighbor distances...")
    nbrs = NearestNeighbors(n_neighbors=int(max(n+1,100)), algorithm='ball_tree',n_jobs=-1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    del nbrs
    distArr = distances[:,n] # distance array containing all distances to nth neighbor
    distArr.sort()
    pts = range(len(distArr))

    # The following looks for the first instance (past the mid point)
    # where the mean of the following [number] points
    # is at least (cutoff-1)*100% greater than the mean of the previous [number] points.
    # Number is currently set to be 0.2% of the total data
    # This works pretty well on data scaled to unit variance. Area for improvement though.
    number = int(np.ceil(len(data)/500))
    cutoff = 1.05
    if verbose:print("Finding elbow...")
    for i in range(int(np.ceil(len(pts)/2)),len(pts)-number):
        if np.mean(distArr[i+1:i+number])>=cutoff*np.mean(distArr[i-number:i-1]):
            dbEps = distArr[i]
            pt=pts[i]
            break

    if verbose:
        print("""
        Epsilon is in the neighborhood of {:05.2f}.
        """.format(dbEps))
    return dbEps,distArr

def dbscan_w_outliers(data,min_n=4,check_tabby=False,verbose=True):
    # numpy array of dataframe for fit later
    X=np.array([np.array(data.loc[i]) for i in data.index])
    if verbose:print("Estimating Parameters...")
    if len(X)>10000:
        # Large datasets have presented issues where a single high density cluster 
        # leads to an epsilon of 0.0 for 4 neighbors.
        # We adjust for this by calculating epsilon for a sample of the data,
        # then we scale min_neighbors accordingly
        if verbose:print("Sampling data for parameter estimation...")
        X_sample = data.sample(n=10000)
    else:
        X_sample = data
    dbEps,distArr = eps_est(X_sample,n=min_n,verbose=verbose)
    if len(X)>10000:
        if verbose:print("Scaling density...")
        min_n = int(len(X)/10000*min_n)
    
    if verbose:print("Clustering data with DBSCAN, eps={:05.2f},min_samples={}...".format(dbEps,min_n))
    #est = DBSCAN(eps=dbEps,min_samples=min_n,n_jobs=-1) # takes too long, deprecated
    #est.fit(X)
    #clusterLabels = est.labels_
    # Outlier score: distance to 4th neighbor?
    nbrs = NearestNeighbors(n_neighbors=min_n+1, algorithm='ball_tree',n_jobs=-1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    del nbrs
    distArr = distances[:,min_n]

    # The following determines the cluster edge members
    # by checking if any outlying points contain a clustered neighbor.
    # Necessary given the heuristic nature of epsilon, provides a buffer.
    # Optimization opportunity: this could be parellelized pretty easily
    d = {True:-1,False:0}
    clusterLabels = np.array([d[pt>dbEps] for pt in distArr]) 
    for i,label in enumerate(clusterLabels):
        # For all identified outliers (pts w/o enough neighbors):
        if label == -1:
            j=1
            # for the neighbors within epsilon
            while distances[i,j]<dbEps:
                # if a neighbor is labeled as part of the cluster,
                if clusterLabels[indices[i,j]] == 0:
                    # then this pt is an edge point
                    clusterLabels[i]=1
                    break
                j+=1
        
    if check_tabby:
        if data.index.str.contains('8462852').any():
            tabbyInd = list(data.index).index(data[data.index.str.contains('8462852')].index[0])
            if clusterLabels[tabbyInd] == -1:
                print("Tabby has been found to be an outlier in DBSCAN.")
            else:
                print("Tabby has NOT been found to be an outlier in DBSCAN")
        else:
            print("MISSING: Tabby is not in this data.")
                     
    numout = len(clusterLabels[clusterLabels==-1])
    numedge = len(clusterLabels[clusterLabels==1])
    if verbose:
        print("There are {:d} total outliers and {:d} edge members.".format(numout,numedge))

    return clusterLabels

if __name__=="__main__":
    """
    If this is run as a script, the following will parse the arguments it is fed, 
    or prompt the user for input.
    
    python db_outliers.py path/to/file n_features path/to/output_file
    """
    # f - file, pandas dataframe saved as csv with calculated features
    if sys.argv[1]:
        f = sys.argv[1]
    else:
        while not f:
            f = raw_input("Input path: ")
     
    print("Reading %s..."%f)
    
    df = pd.read_csv(f,index_col=0)
    
    if sys.argv[2]:n_feat=sys.argv[2]
    else: n_feat = raw_input("Number of features in data: ")
    if not n_feat:
        print("No features specified, assuming default number of features, 60.")
        n_feat = 60
           
    # of - output file
    if sys.argv[3]:
        of = sys.argv[3]
    else:
        of = raw_input("Output path: ")
    if not of:
        print("No output path specified, saving to 'output.npy' in local folder.")
        of = 'output'
    
    np.save(of,dbscan_w_outliers(df[:n_feat]))
    print("Done.")