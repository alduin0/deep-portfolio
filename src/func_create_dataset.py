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
    Creates torch datasets for a given train-validate-test split. Raw data is fetched from
    yahoo finance according to argument 'days' passed to function. The number of preceeding
    days used as features can be set by seq_len. seq_len = 5 means that five days will
    be used as features and the following day as target.

    Parameters:
    -----------
    ticker : tuple[str] | list[str]
           tuple or list of strings encoding valid yahoo finance ticker symbols
    split : tuple[int] | list[int]
          tuple or list of ints depciting the demanded (train, val, test) split
    seq_len : int
            int setting the length of historic dates used for predicting the upcoming
            single date
    days : int
         size of trading days pulled from yahoo finance as raw data

    Returns:
    --------
    dict
        dictionary of torch datasets for training, validation, and testing data.
    """
    if sum(split) != 100:
        mylog.error(f"Invalid split values (train, val, test) = {split}")
        raise ValueError('split must add up to 100')
    mylog.info(f"Called function: {inspect.stack()[0].function}")

    # ================================ GETTING MULTITICKER DATA AT ONCE ================================================
    mytickers = yf.Tickers(ticker)                                             # assign yahoo finance ticker
    histories = mytickers.history(period=f"{days}d")                           # pull historic data
    # instead of dropping nan rows, implement data filling later
    histories = histories.drop(columns=['High', 'Low', 'Close',                # drop not needed
                                        'Volume', 'Dividends',                 # columns of data
                                        'Stock Splits'])                       # for saving ram
    rows_before = len(histories)                                               # percentage info
    histories = histories.dropna()                                             # of unusable rows
    rows_after = len(histories)                                                # for user
    mylog.info(f" {(rows_after - rows_before) / rows_before * 100:.2f} [%] were removed by dropna().")
    mylog.info(f"Using {rows_after} rows for upcoming data preparation.")
    # logging the timespan covered in terms of dates in real world
    dates = histories.index                                                    # info on
    mylog.info(f"Used data covers dates from {dates.min()} to {dates.max()}")  # timespan
    histories = histories['Open']                                              # covered
    # ============================= SCALING DATA TO [0,1] ==============================================================
    for key in histories.keys():                                               # transforming
        histories[key] = MinMaxScaler(feature_range=(0,1)).fit_transform(      # data to normalised
                                      histories[key].values.reshape(-1,1))     # format for speed
    hist_val = histories.values                                                #
    # ============================== SLICING FRAME BLOCKS INTO HISTORY SEQUENCES =======================================
    index_splits = np.array((split[0], split[0]+split[1]))/100 * rows_after    # indices where
    index_splits = [int(val) for val in index_splits]                          # to split arrays

    data_train = hist_val[:index_splits[0],:]                                  # separating raw data
    data_val   = hist_val[index_splits[0]:index_splits[1], :]                  # into given train,
    data_test  = hist_val[index_splits[1]::, :]                                # val, test split

    num_ticker = len(ticker)                                                   # defining sizes
    size_train = data_train.shape[0]                                           # and counts for
    size_val =   data_val.shape[0]                                             # allocation of
    size_test = rows_after - size_train - size_val                             # memory later
    size_cols = num_ticker * seq_len                                           #

    # allocating memory taking sequence lengths into account
    x_train = np.zeros((size_train - seq_len, size_cols), dtype=np.float64)
    y_train = np.zeros((size_train - seq_len, num_ticker), dtype=np.float64)
    x_val = np.zeros((size_val - seq_len, size_cols), dtype=np.float64)
    y_val = np.zeros((size_val - seq_len, num_ticker), dtype=np.float64)
    x_test = np.zeros((size_test - seq_len, size_cols), dtype=np.float64)
    y_test = np.zeros((size_test - seq_len, num_ticker), dtype=np.float64)

    # assigning data from data arrays to their flattened analoga
    # in respective arrays for train and target
    for index in range(x_train.shape[0]):
        x_train[index,:] = data_train[index:index+seq_len, :].flatten()
        y_train[index,:] = data_train[index+seq_len, :]

    for index in range(x_val.shape[0]):
        x_val[index,:] = data_val[index:index+seq_len, :].flatten()
        y_val[index,:] = data_val[index+seq_len, :]

    for index in range(x_test.shape[0]):
        x_test[index,:] = data_test[index:index+seq_len, :].flatten()
        y_test[index,:] = data_test[index+seq_len, :]

    # converting numpy arrays to torch datasets
    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train),
                                                   torch.from_numpy(y_train))
    dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(x_val),
                                                 torch.from_numpy(y_val))
    dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test),
                                                  torch.from_numpy(y_test))

    mydict = {'dataset_train':dataset_train,
              'dataset_val':dataset_val,
              'dataset_test':dataset_test}
    return mydict