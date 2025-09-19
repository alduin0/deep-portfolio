import logging
import pathlib as pl
import datetime
import json
import torch

from src.func_create_dataset import create_dataset

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logfilename = f"mainlog-{timestamp}.log"
logging.basicConfig(level=logging.DEBUG,
                    filename=pl.Path.cwd().joinpath('logs', logfilename),
                    encoding='utf-8',
                    format='%(asctime)s - PID %(process)d - %(name)s - %(levelname)s - %(message)s',
                    filemode='a')

mylog = logging.getLogger(__name__)

def main():
    # ========================= READING CONFIG FILE ====================================================================
    mylog.info("Reading in external config file")
    path_config = pl.Path(__file__).parent.joinpath('input', 'config.json')
    if pl.Path(path_config).exists():
        with open(path_config) as myfile:
            config = json.load(myfile)
        mylog.info("Finished reading in external config file!")
    else:
        mylog.error(f"No external config file found at: {path_config}")

    # ========================= CREATING PYTORCH DATALOADERS ===========================================================
    # ========================= TRAINING - VALIDATION - TEST ===========================================================
    mydata = create_dataset(ticker=config['tickers'],
                            split=config['pars-data']['data-split'],
                            seq_len=config['pars-data']['size-history'],
                            days=config['pars-data']['size-series'],
                            )

    loader_train = torch.utils.data.DataLoader(mydata['dataset_train'],
                                               batch_size=config['pars-learning']['size-batch'],
                                               shuffle=False,
                                               num_workers=config['pars-learning']['num-workers'],
                                               )
    loader_val   = torch.utils.data.DataLoader(mydata['dataset_val'],
                                               batch_size=config['pars-learning']['size-batch'],
                                               shuffle=False,
                                               num_workers=config['pars-learning']['num-workers'],
                                               )
    loader_test   = torch.utils.data.DataLoader(mydata['dataset_test'],
                                               batch_size=config['pars-learning']['size-batch'],
                                               shuffle=False,
                                               num_workers=config['pars-learning']['num-workers'],
                                               )

    return 0


if __name__ == '__main__':
    main()