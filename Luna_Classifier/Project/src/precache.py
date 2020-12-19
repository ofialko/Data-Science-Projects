
from torch.utils.data import DataLoader
from src.config.logconfig import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from collections import namedtuple

from src.dsets import LunaDataset
from src.utils import enumerateWithEstimate
    
cli_args = namedtuple('cli_args',['batch_size', 'num_workers'])
cli_args.batch_size = 10
cli_args.num_workers = 4
    
def main():
        log.info("Starting {}, {}".format(__name__, cli_args))

        prep_dl = DataLoader(
            LunaDataset(),
            batch_size =cli_args.batch_size,
            num_workers=cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            prep_dl,
            "Stuffing cache",
            start_ndx=prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass
        
if __name__ == "__main__":
    main()