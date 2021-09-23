import pandas as pd
import numpy as np


def snr_cnr_on_dataframe(df: pd.DataFrame, id_col: str, class_col_name: str,
             positive_classes: list, target_cols: list, abs_cnr: bool = True, anonimize: bool = True) -> (pd.DataFrame, pd.DataFrame):
  """
  Simple method for SNR and CNR calculation over a pandas DataFrame.\n
  :param df: pandas dataframe stores all data
  :param id_col: name of the column which contains the ID
  :param class_col_name: name of the column which contains the factor variable
  :param positive_classes: positive factor values. rows with these factor value will acts as signal - and the others are the background
  :param target_cols: list of target columns (e.g. series names)
  :param abs_cnr: switch for modified cnr calculation (absolute value in the numerator)
  :param anonimize: drop the id col
  :returns: snr and cnr dataframe
  """

  if not isinstance(df, pd.DataFrame):
    raise TypeError

  if not any([isinstance(c, str) for c in [id_col, class_col_name]]):
    raise TypeError

  if not any([isinstance(c, list) for c in [positive_classes, target_cols]]):
    raise TypeError

  snr_df = pd.DataFrame(columns=[id_col] + target_cols)
  cnr_df = pd.DataFrame(columns=[id_col] + target_cols)

  binary_col = "TYPE"
  df[binary_col] = df[class_col_name].apply(lambda x: x in positive_classes)

  id_list = sorted(df[id_col].unique())

  for _id in id_list:
    _df = df[df[id_col] == _id]
    signal_index = _df[binary_col]
    bg_index = _df[binary_col] == False
    _snr = {id_col: _id}
    _cnr = {id_col: _id}

    for series in target_cols:
      signal_mean = _df[signal_index][series].mean()
      background_mean = _df[bg_index][series].mean()
      background_sd = _df[bg_index][series].std()

      _snr[series] = signal_mean / background_sd

      if abs_cnr:
        _cnr[series] = abs(signal_mean - background_mean) / background_sd
      else:
        _cnr[series] = (signal_mean - background_mean) / background_sd

    snr_df = snr_df.append(_snr, ignore_index=True)
    cnr_df = cnr_df.append(_cnr, ignore_index=True)

  if anonimize:
    return snr_df[target_cols], cnr_df[target_cols]
  else:
    return snr_df, cnr_df


def pooled_snr_cnr_on_df(df: pd.DataFrame, class_col_name: str,
             positive_classes: list, target_cols: list, abs_cnr: bool = True) -> (dict, dict):
  """
  Function to calculate snr and cnr on a dataframe while aint give a shit about the origin and meaning of the data -  NO ID COL.\n
  :param df: pandas dataframe stores all data
  :param class_col_name: name of the column which contains the factor variable
  :param positive_classes: positive factor values. rows with these factor value will acts as signal - and the others are the background
  :param target_cols: list of target columns (e.g. series names)
  :param abs_cnr: switch for modified cnr calculation (absolute value in the numerator)
  :param anonimize: drop the id col
  :returns: snr_dict and cnr_dict
  """

  if not isinstance(df, pd.DataFrame):
    raise TypeError

  if not any([isinstance(c, list) for c in [positive_classes, target_cols]]):
    raise TypeError

  snr_dict = {}
  cnr_dict = {}

  binary_col = "TYPE"
  df[binary_col] = df[class_col_name].apply(lambda x: x in positive_classes)


  signal_index = df[binary_col]
  bg_index = df[binary_col] == False

  for series in target_cols:
    signal_mean = df[signal_index][series].mean()
    background_mean = df[bg_index][series].mean()
    background_sd = df[bg_index][series].std()

    snr_dict[series] = signal_mean / background_sd

    if abs_cnr:
      cnr_dict[series] = abs(signal_mean - background_mean) / background_sd
    else:
      cnr_dict[series] = (signal_mean - background_mean) / background_sd


  return snr_dict, cnr_dict


def synthetic_test():
    # create sample data
  np.random.seed(1862)

  pcol = "patient_id"
  ccol = "class"
  slist = [ f"series_{str(f).zfill(2)}" for f in range(1,13)]
  N = 100

  _df = pd.DataFrame(columns=[pcol]+[ccol]+slist)
  _positive_classes = ["A","B","C"] #positive class
  _all_classes = ["A","B","C","D","E","F","G"]
  _ids = [f"patient_{str(f).zfill(2)}" for f in range(1,10)]
  _df[pcol] = np.random.choice(_ids,N)
  _df[ccol] = np.random.choice(_all_classes,N)
  for s in slist:
    _mean = np.random.randint(80,120)
    _df[s] = np.random.rand(N) 

    # add some nan
    _df[s] = _df[s].apply(lambda x : np.random.randn()*5+_mean if x > 0.1 else np.nan)

  print(_df.sample(5))

  snr_df, cnr_df = snr_cnr_on_dataframe(_df,pcol,ccol,_positive_classes,slist)
  print(snr_df)
  print(cnr_df)

def pooled_test():
  df = pd.read_csv("df_metastases_only.csv")
  target_cols = list(df.columns[5:-1])
  print(f"\nsignal: metastases_melanom vs background: metastases_lung")
  snr, cnr = pooled_snr_cnr(df,"class_segmentation",["metastases_melanom"],target_cols)
  print("snr")
  print(snr)
  print("cnr")
  print(cnr)

  print(f"\nsignal: metastases_lung vs background: metastases_melanom")
  snr, cnr = pooled_snr_cnr(df,"class_segmentation",["metastases_lung"],target_cols)
  print("snr")
  print(snr)
  print("cnr")
  print(cnr)

if __name__ == '__main__':
  pooled_test()