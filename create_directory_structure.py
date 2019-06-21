'''

usage :  python create_directory_structure.py /home/nsriniva/raid_nsriniva/ND_Project -a UMD STR RankOne_D -d Age_1213_17181920 Age_18_1819

We will create a directory structure to
to save the face recognition results.

assume,
root dir = /home/nsriniva/ND_Project/Results
fr_results_dir = root_dir + 'FR_Results' #indicates face recognition results directory
dataset_fr_results_dir = fr_results_dir + datasetname
algorithms_fr_results_dir = dataset_fr_results_dir + 'Algorithms'
individual_algorithms_dir = algorithms_fr_results_dir + 'STR' or algorithms_fr_results_dir + 'UMD' or algorithms_fr_results_dir + 'RankOne_D'

Root_Dir
    |----FR_Results
            |----Dataset1
            |        |----Algorithms 
            |                |----STR
            |                |----UMD
            |                |----RankOne   
            |----Dataset2
            |        |----Algorithms 
            |                |----STR
            |                |----UMD
            |                |----RankOne            
            :
            :

            |----DatasetN
            |        |----Algorithms 
            |                |----STR
            |                |----UMD
            |                |----RankOne


Once the directories are created copy tht genuine, imposter and labels text files from each algorithm to their respective directories.
This seem tedious at first but it greatly helps while analyzing and plotting the results.

Author: Nisha Srinivas
@uncw
'''

import os
import argparse
import logging
import sys

logger = logging.getLogger(__name__)
formatter = '%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                   format = formatter)


def parseArguments():
    
    description = '''Create the directory structure to store the results'''
    description = " ".join(description.split())
    epilog = '''Created by Nisha Srinivas - srinivasn@uncw.edu'''
    version = "0.0.0"
    parser = argparse.ArgumentParser(version=version, description=description, epilog=epilog)
    
    parser.add_argument('-a', '--algolist', nargs='*', dest='algorithms', default=[])
    parser.add_argument('-d', '--datasetnames', nargs='*', dest='datasetnames', default=[])
    parser.add_argument('rootDir', help='specify the root directory to store the results')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parseArguments()
    
    rootDir = args.rootDir
    algorithms = args.algorithms
    datasetnames = args.datasetnames

    logger.info('The root directory is {}'.format(rootDir))
    
    for eachd in datasetnames:
        for eacha in algorithms:
            dirofInterest = os.path.join(rootDir,'FR_Results',eachd,'Algorithms',eacha)
            if not os.path.isdir(dirofInterest):
                os.makedirs(dirofInterest)

