#! /usr/bin/env python

'''
iusage: python create_similarity_matrices.py -a umd str -c 16 /home/nsriniva/raid_nsriniva/ChildFR/Datasets/umd/Results_Age_1213_17181920

This code reads the janice (umd and str) algorithms output and
converts it into a format to be used by Vitor's code.

@author: Nisha Srinivas
@uncw

'''

import time
import sys
import os
#import optparse
import argparse
import csv
from collections import defaultdict
import numpy as np
from glob import glob  
import numpy as np
import logging
import multiprocessing as mp


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

def getMappings(ifile,batchStart, batchSize):
    with open(ifile) as tid:
        tid.seek(batchStart)
        lines = tid.read(batchSize).splitlines()
        results = []
        probeList = []
        galleryList = []
        for eachline in lines:
            eline = eachline.split(',')
            #print (eline[0],eline[2])
            if eline[2].find('A.jpg') >= 0:
                galleryList.append((eline[0],eline[2]))
            elif eline[2].find('B.jpg') >= 0:
                probeList.append((eline[0],eline[2]))
            results.append((eline[0],eline[2]))
    return results, galleryList, probeList

def getScores(ifile,batchStart,batchSize,gmaps,pmaps):
    with open(ifile) as tid:
        tid.seek(batchStart)
        lines = tid.read(batchSize).splitlines()
        genuineComps = []
        imposterComps = []
        gComps = []
        iComps = []     
        for eachline in lines:
            comarisonType = None
            gid,lfgkdlkgfgfpid = None, None
            subjectIdProbe, subjectIdGalllery = None, None
            eline = eachline.split(',')
            # assume e[0] us probe and e[1] is gallery
            if eline[0] != eline[1]:
                if eline[0] in pmaps.keys() and eline[1] in gmaps.keys():
                    pid,gid = pmaps[eline[0]], gmaps[eline[1]]
                    subjectIdProbe = pid.split('B')[0]
                    subjectIdGallery = gid.split('A')[0]
                    if (subjectIdProbe == subjectIdGallery):
                        genuineComps.append((pid,gid,eline[3]))
                        gComps.append((eline[0],eline[1],eline[3]))
                    elif (subjectIdProbe != subjectIdGallery):
                        imposterComps.append((pid,gid,eline[3],eline[0],eline[1]))
                        iComps.append((eline[0],eline[1],eline[3]))
    return genuineComps, imposterComps, gComps, iComps

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
    
    description = '''creating similarity matrices'''

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
    parser.add_argument('-a', '--algo-list', nargs='*', dest='algorithms',default=['umd'])
    parser.add_argument("-i","--inbuilt",action="store_true", dest="inbuiltFd",
                    help="use the files created by using the inbuilt face detector")
    parser.add_argument('resultsDir', help='specify the path to the algorithm results')
    
    args = parser.parse_args()
        
    return args


    
if __name__ == '__main__':
    args = parseOptions()

    resultsDir = args.resultsDir
    algorithms = args.algorithms
    

    for algo in algorithms:
        algoResultsDir = os.path.join(resultsDir,algo)
        #find all subdirectories in results dir
        subdirs = glob(algoResultsDir + '/*')
        for eachdataset in subdirs:
            resultsDataDir = eachdataset.split(os.path.sep).pop()
            datasetname = eachdataset.split(os.path.sep).pop().split('Results_').pop()
            #print datasetname, eachalgo
            logger.info('Processing algorithm {} for dataset {}'.format(algo,datasetname))
            if args.inbuiltFd:
                mappingsFile = os.path.join(eachdataset, 'auto_modified_janice_detect_outfile.csv')
                fPath = os.path.join(eachdataset, 'auto_' + algo + '_janice_verify_results.csv')
                resultsFile = os.path.join(eachdataset,'auto_' + algo + '_reduced_janice_verify_results.csv')
                genuineFile = os.path.join(eachdataset,'auto_' + datasetname + '_' + algo + '_expt1_matchScores.csv')
                imposterFile = os.path.join(eachdataset, 'auto_' + datasetname + '_' + algo + '_expt1_nonmatchScores.csv')
                gFile =  os.path.join(eachdataset,'auto_' + datasetname + '_' + algo + '_expt1_authenticScores.txt')
                iFile = os.path.join(eachdataset, 'auto_' + datasetname + '_' + algo + '_expt1_imposterScores.txt')
                lFile = os.path.join(eachdataset, 'auto_' + datasetname + '_' + algo + '_expt1_labels.txt')
            else:
                mappingsFile = os.path.join(eachdataset, 'manual_modified_janice_detect_outfile.csv')
                fPath = os.path.join(eachdataset,'manual_' + algo + '_janice_verify_results.csv')
                resultsFile = os.path.join(eachdataset,'manual_' + algo + '_reduced_janice_verify_results.csv')
                genuineFile = os.path.join(eachdataset,'manual_' + datasetname + '_' + algo + '_expt1_matchScores.csv')
                imposterFile = os.path.join(eachdataset,'manual_' + datasetname + '_' + algo + '_expt1_nonmatchScores.csv')
                gFile =  os.path.join(eachdataset,'manual_' + datasetname + '_' + algo + '_expt1_authenticScores.txt')
                iFile = os.path.join(eachdataset, 'manual_' + datasetname + '_' + algo + '_expt1_imposterScores.txt')
                lFile = os.path.join(eachdataset, 'manual_' + datasetname + '_' + algo + '_expt1_labels.txt')
 
            pool = mp.Pool(args.numCores)
            mappingJobs = []
            for idx,(start,bSize) in enumerate(splitfile(mappingsFile,1024*1024)):
                mappingJobs.append(pool.apply_async(getMappings,(mappingsFile,start,bSize)))
            
            mappings = []
            gmappings = []
            pmappings = []
            for eachjob in mappingJobs:
                r, g, p = eachjob.get()
                mappings.extend(r)
                gmappings.extend(g)
                pmappings.extend(p)

            pool.close()
        
            mappings.pop(0)
            ##convert list to dictionary 
            with open(lFile,'w') as fid:
                for eachmapping in mappings:
                    fid.write('{} {}\n'.format(eachmapping[0],eachmapping[1]))
                    
            mappings = dict(mappings)
            gmappings = dict(gmappings)
            pmappings = dict(pmappings)
            logger.info('The number of keys in the gallery are {}'.format(len(gmappings.keys())))
            logger.info('The number of keys in the gallery are {}'.format(len(pmappings.keys())))
            
            pool = mp.Pool(args.numCores)
            jobs = []

            start_time = time.time()
            for idx, (start,bSize) in enumerate(splitfile(fPath)):
                jobs.append(pool.apply_async(getScores,(fPath,start,bSize,gmappings,pmappings)))
   
            genuineMatches = []
            imposterMatches = []
            genuineIds = []
            imposterIds = []
            for idx,eachjob in enumerate(jobs):
                gen, imp, gc, ic = eachjob.get()
                genuineMatches.extend(gen)
                imposterMatches.extend(imp)
                genuineIds.extend(gc)
                imposterIds.extend(ic)
            pool.close()
            logger.info('Time taken to read the file and process it is {}'.format((time.time() - start_time)))
            logget.info('The number of genuine and imposter comparisons are {} and {}'.format(len(genuineMatches),len(imposterMatches)))
            ##write the match scores and non-matches to it's own file.
   
            writeToFileTxt(genuineMatches,genuineFile)
            writeToFileTxt(imposterMatches,imposterFile)
            writeToFileTxt(genuineIds, gFile)
            writeToFileTxt(imposterIds, iFile)
            logger.info('--------') 
