import logging
import pathlib as pl
import datetime
import json

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
    path_config = pl.Path(__file__).parent.joinpath('inpu', 'config.json')
    if pl.Path(path_config).exists():
        with open(path_config) as myfile:
            config = json.load(myfile)
        mylog.info("Finished reading in external config file!")
    else:
        mylog.error(f"No external config file found at: {path_config}")


    return 0


if __name__ == '__main__':
    main()