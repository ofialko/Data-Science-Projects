
from config.logconfig import logging
from dsets2D import CifarDataset
import argparse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Precache dataets')
parser.add_argument('path_data', type=str, 
                    help='path to data')

args = parser.parse_args()
path_data  = args.path_data

if __name__ == "__main__":
    log.info("Starting {}".format(__name__))

    trn_ds = CifarDataset(path_data, isTrainSet_bool=True)
    val_ds = CifarDataset(path_data, isTrainSet_bool=False)