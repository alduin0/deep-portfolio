from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import inspect
import logging
import torch


mylog = logging.getLogger(__name__)

def create_dataset(ticker:tuple[str]|list[str]=('DB1.DE', '^STOXX50E'),
                   split:tuple[int]|list[int]=(80, 15, 5),
                   seq_len:int=5,
                   days:int=90,
                   )->dict:
    """
    hello dox
    """
    if sum(split) != 100:
        mylog.error(f"Invalid split values (train, val, test) = {split}")
        raise ValueError('split must add up to 100')
    mylog.info(f"Called function: {inspect.stack()[0].function}")

    # ================================ GETTING MULTITICKER DATA AT ONCE ================================================
    mytickers = yf.Tickers(ticker)
    histories = mytickers.history(period=f"{days}d")
    # instead of dropping nan rows, implement data filling later
    histories = histories.drop(columns=['High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
    rows_before = len(histories)
    histories = histories.dropna()
    rows_after = len(histories)
    mylog.info(f" {(rows_after - rows_before) / rows_before * 100:.2f} [%] were removed by dropna().")
    mylog.info(f"Using {rows_after} rows for upcoming data preparation.")
    # logging the timespan covered in terms of dates in real world
    dates = histories.index
    mylog.info(f"Used data covers dates from {dates.min()} to {dates.max()}")
    histories = histories['Open']
    # ============================= SCALING DATA TO [0,1] ==============================================================
    for key in histories.keys():
        histories[key] = MinMaxScaler(feature_range=(0,1)).fit_transform(histories[key].values.reshape(-1,1))
    hist_val = histories.values
    print(hist_val.shape)
    # ============================== SLICING FRAME BLOCKS INTO HISTORY SEQUENCES =======================================
    index_splits = np.array((split[0], split[0]+split[1]))/100 * rows_after
    index_splits = [int(val) for val in index_splits]

    data_train = hist_val[:index_splits[0],:]
    data_val   = hist_val[index_splits[0]:index_splits[1], :]
    data_test  = hist_val[index_splits[1]::, :]
    index_splits.append(data_test.shape[0])




    return {}