#! /usr/bin/env python

'''


@author: Nisha Srinivas
@uncw
'''


import sys
import os
#import optparse
import argparse
import csv
from collections import defaultdict
#import numpy as np
from glob import glob  
import numpy as np
import logging

logger = logging.getLogger(__name__)
formatter = '%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                   format = formatter)



def _glob(path, exts):
    """Glob for multiple file extensions

    Parameters
    ----------
    path : str
    A file name without extension, or directory name
    exts : tuple
    File extensions to glob for

    Returns
    -------
    files : list
    list of files matching extensions in exts in path

    """
    path = os.path.join(path, "*") if os.path.isdir(path) else path + "*"
    return [f for files in [glob(path + ext) for ext in exts] for f in files]
    
def writeToFile(scoreList, filepath):
    with open(filepath,'w') as fid:
        writer = csv.writer(fid,delimiter = ',',lineterminator='\n')
        for eachrow in scoreList:
            writer.writerow(list(eachrow))

def writeToFileTxt(scoreList, filepath):
    with open(filepath,'w') as fid:
        for eachrow in scoreList:
            fid.write('{} {} {}\n'.format(eachrow[0],eachrow[1],eachrow[2]))            
    
def parseOptions():

    description = '''Convert RankOne output to ND Format'''

    description = " ".join(description.split())

    epilog = '''Created by Nisha Srinivas - srinivasn@uncw.edu'''

    version = "0.0.0"

    # Setup the parser
    parser = argparse.ArgumentParser(version=version,description=description,epilog=epilog)

    #Here are some templates for standard option formats.
    #parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
    #                 help="Increase the verbosity of the program")

    parser.add_argument("-c","--cores",type=int, dest="numCores", default=8,
                    help="define the number of cores to use")
    parser.add_argument('-a', '--algo-list', nargs='*', dest='algorithms',default=['RankOne_D'])
    parser.add_argument("-i","--inbuilt",action="store_true", dest="inbuiltFd",
                    help="use the files created by using the inbuilt face detector")
    parser.add_argument('resultsDir', help='specify the path to the algorithm results')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = parseOptions()
    
    resultsDir = args.resultsDir
    
    subdirs = glob(os.path.join(resultsDir,'*'))
    print subdirs
    datasetnames = [eachsubdir.split(os.path.sep).pop() for eachsubdir in subdirs]
    algorithms = args.algorithms

    for eachdataset in datasetnames:
        for eachalgo in algorithms:
        
            filesdir = os.path.join(resultsDir,eachdataset,'Algorithms',eachalgo)
            simfile = glob(os.path.join(filesdir,'*similarity_matrix*.csv'))
            print simfile
            
           
            genuineFile = os.path.join(filesdir,'manual_' + eachdataset + '_' + eachalgo + '_' + 'expt1_authenticScores.txt')
            imposterFile= os.path.join(filesdir,'manual_' + eachdataset + '_' + eachalgo + '_' + 'expt1_imposterScores.txt')

            genuinescoresFile = os.path.join(filesdir,'manual_' + eachdataset + '_' + eachalgo + '_' + 'expt1_matchScores.csv')
            imposterscoresFile= os.path.join(filesdir,'manual_' + eachdataset + '_' + eachalgo + '_' + 'expt1_nonmatchScores.csv')
            
            lFile = os.path.join(filesdir,'manual_' + eachdataset + '_' + eachalgo + '_' + 'expt1_labels.txt') 

            smmatrix = np.loadtxt(simfile[0],delimiter=',',dtype=str)
            
            rows,cols = smmatrix.shape

            assert(rows == cols)
            

            galleryIds = []
            probeIds = []

            for i in range(1,rows):
                if smmatrix[0,i].find('A') > 0:
                    galleryIds.append(i)
                else:
                    probeIds.append(i)
            
            galleryIds = np.asarray(galleryIds)
            probeIds = np.asarray(probeIds)

            smmatrix = np.delete(smmatrix,galleryIds,axis=1)
            smmatrix = np.delete(smmatrix,probeIds,axis=0)
            rows,cols = smmatrix.shape
            probeIds = smmatrix[0,1:]
            galleryIds = smmatrix[1:,0]  
            
            mappings = []
            ids = np.concatenate((probeIds,galleryIds),axis=0)
            
            with open(lFile,'w') as fid:
                for i in range(0,ids.shape[0]):
                    mappings.append((ids[i],i+1))
                    fid.write('{} {}\n'.format(i+1,ids[i]+'.jpg'))

            print('number of mappings is {}'.format(len(mappings)))
            print mappings[0]
            mappings = dict(mappings)
            print mappings.keys()[1:10]

            matchIdx = []
            matchIdxScores = []
            nonmatchIdx = []
            nonmatchIdxScores = []
            for r in range(1,rows):
                for c in range(1,cols):
                    gi = smmatrix[r,0]
                    pi = smmatrix[0,c]
                    score = smmatrix[r,c]
                    gidx = mappings[gi]
                    pidx = mappings[pi]
                    
                    gSub = gi.split('A')[0]
                    pSub = pi.split('B')[0]

                    if gSub == pSub :
                        matchIdxScores.append((pi+ '.jpg',gi+ '.jpg',score))
                        matchIdx.append((pidx,gidx,score))
                    else:
                        nonmatchIdxScores.append((pi+ '.jpg',gi+ '.jpg',score))
                        nonmatchIdx.append((pidx,gidx,score)) 
            print(len(matchIdx))
            print(len(matchIdxScores))
            print(len(nonmatchIdx))
            print(len(nonmatchIdxScores))

        
            
        writeToFile(matchIdxScores,genuinescoresFile)
        writeToFile(nonmatchIdxScores,imposterscoresFile)
        writeToFileTxt(matchIdx,genuineFile)
        writeToFileTxt(nonmatchIdx,imposterFile)
