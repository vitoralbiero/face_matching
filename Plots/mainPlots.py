'''
usage = python mainPlots.py resultsDir -a UMD STR RankOne_D -d Age_18_1819 Age_1213_17181920 -p cmcPlot rocPlot densityPlot -m 
Plot the following plots,

1. genuine and imposter distributions.
2. cmc curves
3. roc curves

The code will need the authenticScores.txt, imposterScores.txt and labels.txt
labels.txt should contain both the imposter and geunie score labels.

TODO - Add comments!

Author: Nisha Srinivas
@uncw
'''
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import argparse
import multiprocessing as mp
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from sklearn import metrics
from collections import defaultdict
import csv

logger = logging.getLogger(__name__)
formatter = '%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                   format = formatter)


#function reads the file in parallel
def load_files(fpath, chunkStart, chunkSize):
    
    with open(fpath) as tid:
        tid.seek(chunkStart)
        lines = tid.read(chunkSize).splitlines()
        scores = []
        probeIds = []
        galleryIds = []
        for eachline in lines:
            vals = eachline.rstrip('\n').split(' ')
            scores.append(vals)
    return scores

#function split the files into smaller chuks to be used by different processes
def splitfile(fname,size=1024*1024):
    fileEnd = os.path.getsize(fname)
    loopover = True
    with open(fname,'r') as fid:
        cursorpos = fid.tell()
        while loopover:
            begin = cursorpos
            fid.seek(size,1)
            fid.readline()
            cursorpos = fid.tell()
            yield begin, cursorpos-begin
            if cursorpos > fileEnd:
                loopover = False

#collecting the outputs of the jobs processed by different cores
def getScores(ffile,ncores):
    
    pool = mp.Pool(ncores)
    jobs = []
    for idx,(start,bSize) in enumerate(splitfile(ffile,1024*1024)):
        jobs.append(pool.apply_async(load_files,(ffile,start,bSize)))
    
    tmpScores = []
    
    for idx,eachjob in enumerate(jobs):
        tmpScores.extend(eachjob.get())

    logger.info('The number of score read from {} is {}'.format(ffile,len(tmpScores)))
    
    pool.close()
    
    return tmpScores

def plot_seaborn_histogram(aScores, iScores, ofpath, dname,ealgo):
     
    plt.rcParams["figure.figsize"] = [6, 5]

    sns.set_style("whitegrid")
    #print sns.plotting_context()
    #{'lines.linewidth': 1.5, 'lines.markersize': 6.0, 'ytick.major.size': 3.5, 'xtick.major.width': 0.8, 'axes.labelsize': 'medium', 'patch.linewidth': 1.0, 'ytick.minor.size': 2.0, 'font.size': 10.0, 'ytick.minor.width': 0.6, 'axes.titlesize': 'large', 'grid.linewidth': 0.8, 'xtick.major.size': 3.5, 'xtick.minor.size': 2.0, 'xtick.minor.width': 0.6, 'legend.fontsize': 'medium', 'ytick.labelsize': 'small', 'xtick.labelsize': 'medium', 'axes.linewidth': 0.8, 'ytick.major.width': 0.8}i##
    #sns.set_context(rc={'ytick.labelsize':'small','axes.labelsize': 'small','axes.titlesize': 'small', 'xtick.labelsize': 'small','font.size': 5.0})
    #print sns.plotting_context()
    fig = sns.distplot(aScores, color="blue", label='Genuine Scores')
    fig = sns.distplot(iScores,color="red", label='Imposter Scores')
    fig.set_xlabel('Scores',fontsize=12)
    fig.set_ylabel('Relative Frequency',fontsize=12)
    #fig.set_title('Scores Distributions ' + dname + ':' + ealgo, fontsize=14)
    fig.tick_params(labelsize=14,labelcolor="black")
    
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right", borderaxespad=0, ncol=1, fontsize=12, edgecolor='black', handletextpad=0.3)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    dprime = (abs(np.mean(aScores) - np.mean(iScores)) /
                np.sqrt(0.5 * (np.var(aScores) + np.var(iScores))))

    anc = AnchoredText("dprime = " + str(round(dprime,4)), loc="upper left", frameon=False)    
    fig.add_artist(anc)
    fig.figure.savefig(ofpath)
    plt.close()

def plot_histogram(aScores,iScores,ofpath,dname):

    #plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12

    plt.hist(aScores, bins='auto', histtype='step', density=True,
             label= 'Genuine', color='g', linewidth=1.5)
    plt.hist(iScores, bins='auto', histtype='step', density=True,
             label= 'Impostor', color='b', linestyle='dashed',
             linewidth=1.5)    

    legend1 = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                         mode="expand", borderaxespad=0, ncol=1, fontsize=12, edgecolor='black', handletextpad=0.3)

    # plt.gca().invert_xaxis()
    plt.ylabel('Relative Frequency')
    plt.xlabel('Scores')
    plt.title(dname)

    #plt.tight_layout(pad=0)
    #plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12

    d_prime = (abs(np.mean(aScores) - np.mean(iScores)) /
                np.sqrt(0.5 * (np.var(aScores) + np.var(iScores))))

    print('d-prime is {} '.format(d_prime))

    plt.grid(True)
    plt.savefig(ofpath,dpi=300)
    plt.close()

def plot_roc(data,fname):

    numAlgos = len(data.keys())
    colors = ['r','g','b','c','m','y','orange','brown','black']
    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12
    plt.grid(True, zorder=0, linestyle='dashed')
    plt.gca().set_xscale('log')
    begin_x = 1e-4
    end_x = 1e0
    #print(begin_x, end_x)

    ranges = end_x - begin_x
    
    
    for i,eachk in enumerate(sorted(data.keys())):
        rocmetrics = data[eachk]
        plt.plot(rocmetrics['fpr'],rocmetrics['tpr'], color=colors[i],label=eachk,linewidth=2.0)

    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([begin_x, end_x])
    plt.ylim([0.0, 1])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Accept Rate')
    plt.savefig(fname,dpi=300)
    plt.close()

def plot_cmc(data,rankn,fname,tname):
    numAlgos = len(data.keys())
    colors = ['r','g','b','c','m','y','orange','brown','black']
    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams['font.size'] = 12
    plt.grid(True, zorder=0, linestyle='dashed')

    with open(tname,'w') as csvid:
        csvwriter = csv.writer(csvid,delimiter=',')
        #print range(1,rankn+1)
        xvals = range(1,rankn+1)
        csvwriter.writerow(xvals)
        for i,eachk in enumerate(sorted(data.keys())):
            #print xvals, data[eachk], data[eachk][0:rankn]*100
            plt.plot(xvals,data[eachk][0:rankn]*100,colors[i],label=eachk,linewidth=2)
            csvwriter.writerow(data[eachk][0:rankn]*100)

    plt.legend(loc='lower right', fontsize=12)
    #plt.xlim([begin_x, end_x])
    plt.ylim([0.0, 100])
    # plt.ylim([0, 1])
    plt.ylabel('Retrieval Rate')
    plt.xlabel('Rank')
    plt.savefig(fname,dpi=300)
    plt.close()

def getRank(pidx):
    
    global aComps_g
    global iComps_g
    global probeIds_g

    acompScores = aComps_g[aComps_g[:,0] == probeIds_g[pidx]]
    acompScoresIdx = np.argmax(acompScores[:,2])
    acompScores = acompScores[acompScoresIdx]
    icompScores = iComps_g[iComps_g[:,0] == probeIds_g[pidx]]
    #find the number of comparisons greater than the match comparison
    tmprank = len(icompScores[icompScores[:,2] > acompScores[2]]) 

    return tmprank

def compute_ranks(aComps,iComps, idtoimgMappings,ncores):
    idtoimgMappings = dict(idtoimgMappings)
    global aComps_g
    global iComps_g
    global probeIds_g
    aComps_g = np.asarray(aComps,dtype=float)
    iComps_g = np.asarray(iComps,dtype=float)
    galleryIds = np.unique(np.concatenate((aComps_g[:,1],iComps_g[:,1])))
    probeIds_g = np.unique(np.concatenate((aComps_g[:,0],iComps_g[:,0])))
    #probeIds = list(probeIds)

    pool = mp.Pool(ncores)
    results = pool.map(getRank,range(0,probeIds_g.shape[0]))
    ranks = np.zeros(galleryIds.shape[0])
    for eachrow in results:
        ranks[eachrow] += 1
    ranks = np.cumsum(ranks)/float(probeIds_g.shape[0])   

    #print galleryIds.shape, probeIds.shape 
    #ranks = np.zeros(galleryIds.shape[0])
    #for pidx in range(0,probeIds.shape[0]):
    #    if pidx >= 0:
    #        acompScores = aComps[aComps[:,0] == probeIds_g[pidx]]
    #        acompScoresIdx = np.argmax(acompScores[:,2])
    #        acompScores = acompScores[acompScoresIdx]
    #        icompScores = iComps[iComps[:,0] == probeIds_g[pidx]]
    #        #find the number of comparisons greater than the match comparison
    #        tmprank = len(icompScores[icompScores[:,2] > acompScores[2]])  
    #        ranks[tmprank] += 1
    #ranks = np.cumsum(ranks)/float(probeIds.shape[0])
    
    return ranks
            
def parserArguments():

    description = '''Create the directory structure to store the results'''
    description = " ".join(description.split())
    epilog = '''Created by Nisha Srinivas - srinivasn@uncw.edu'''
    version = "0.0.0"
    parser = argparse.ArgumentParser(version=version, description=description, epilog=epilog)

    parser.add_argument('-n','--numCores',dest='numCores',type=int,default=8)
    parser.add_argument('-a', '--algolist', nargs='*', dest='algorithms', default=['UMD','STR','RankOne_D'],help='list all the algorithms')
    parser.add_argument('-d', '--datasetnames', nargs='*', dest='datasetnames', default=['Age_1213_17181920','Age_18_1819'],help='list all the datasets')
    parser.add_argument('-p', '--plots', nargs='*', dest='plotsofinterest', default=['cmcPlot','rocPlot', 'densityPlot'],help='list all the plots to plot')
    parser.add_argument('-e', '--expts', nargs='*', dest='expts',default=['expt1'],help='list all the experiments')
    parser.add_argument('-m', '--manual', action='store_true', dest='manualFD',help='If set to True it will use the manual face detection results and not the inbuilt one')
    parser.add_argument('rootDir', help='specify the root directory to store the results')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parserArguments()
    print args

    rootDir = args.rootDir
    datasetnames = args.datasetnames
    algorithms = args.algorithms
    plotsofinterest = args.plotsofinterest
    expts= args.expts

    for eachd in datasetnames:
        for eache in expts:
            if 'rocPlot' in plotsofinterest:
                rocDict = defaultdict(lambda: defaultdict(dict))
            if 'cmcPlot' in plotsofinterest:
                cmcDict = {}
            for eacha in algorithms:
                
                if args.manualFD:
                    suffix_ = 'manual'
                else:
                    suffix_ = 'auto'

                authenticFile = os.path.join(rootDir,eachd,'Algorithms',eacha, suffix_ + '_' + eachd + '_' + eacha + '_' + eache + '_authenticScores.txt')
                imposterFile = os.path.join(rootDir,eachd,'Algorithms',eacha, suffix_ + '_' + eachd + '_' + eacha + '_' + eache + '_imposterScores.txt')
                labelFile = os.path.join(rootDir,eachd,'Algorithms',eacha, suffix_ + '_' + eachd + '_' + eacha + '_' + eache + '_labels.txt')
                
                #authentic scores - a list of scores in string format are resturned
                authenticComparisons = getScores(authenticFile, args.numCores)
                authenticScores = [eachval[2] if (len(eachval) == 3) else eachval[0] for eachval in authenticComparisons]
                authenticScores = np.asarray(authenticScores,dtype=float)
                #imposter scores - a list of scores in string format are returned
                imposterComparisons = getScores(imposterFile,args.numCores)
                imposterScores = [eachval[2] if (len(eachval) == 3) else eachval[0] for eachval in imposterComparisons]
                imposterScores = np.asarray(imposterScores,dtype=float)
                #labels - a list of tuples is returned (a,b) -> a is the id and b is the image name
                labels = getScores(labelFile,args.numCores)
                print labels[0]                
                if 'densityPlot' in plotsofinterest:
                    ofile = os.path.join(os.path.join(rootDir,eachd,'Algorithms',eacha, suffix_ + '_' + eachd + '_' + eacha + '_' + eache + '_densityPlot.png'))
                    plot_seaborn_histogram(authenticScores,imposterScores,ofile,eachd,eacha)
                if 'rocPlot' in plotsofinterest:
                    authentic_y = np.ones(authenticScores.shape[0])
                    imposter_y = np.zeros(imposterScores.shape[0])
                    y = np.concatenate([authentic_y, imposter_y])
                    logger.info('The total number of comparisons are {}'.format(y.shape[0]))
                    scores = np.concatenate([authenticScores, imposterScores])
                    fpr, tpr, thr = metrics.roc_curve(y, scores, drop_intermediate=True)                         
                    rocDict[eacha]['fpr'] = fpr
                    rocDict[eacha]['tpr'] = tpr
                    rocDict[eacha]['thr'] = thr
                    authentic_y = None
                    imposter_y = None
                    y = None
                
                if 'cmcPlot' in plotsofinterest:
                    eranks = compute_ranks(authenticComparisons, imposterComparisons, labels, args.numCores)
                    cmcDict[eacha] = eranks
                    eranks = None

            if 'rocPlot' in plotsofinterest:
                figname = os.path.join(rootDir,eachd,'Algorithms', suffix_ + '_' + eachd + '_' + eache + '_rocPlot.png')
                #plot the roc curves for each experiment 
                plot_roc(rocDict,figname)
            
            if 'cmcPlot' in plotsofinterest:
                figname = os.path.join(rootDir,eachd,'Algorithms', suffix_ + '_' + eachd + '_' + eache + '_cmcPlot.png')
                txtname = os.path.join(rootDir,eachd,'Algorithms', suffix_ + '_' + eachd + '_' + eache + '_cmcRanks.txt')
                #plot the roc curves for each experiment 
                plot_cmc(cmcDict,10,figname,txtname)    
