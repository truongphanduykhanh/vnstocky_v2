'''
This script is to generate features data for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-09-20'


import os

import pandas as pd
import json


def get_tickers(folder: str = '../financeInfo') -> list[str]:
    '''
    Get all tickers in a folder

    Parameters
    ----------
    folder : str
        Path to folder wanted to get tickers. Ex: '../data/excelfull'

    Returns
    -------
    list of str
        List of tickers. Ex: ['A32', 'AAM', 'AAT', ...]
    '''
    file_names = pd.Series(os.listdir(folder))
    file_names = file_names.sort_values().str.split('_')
    tickers = [file_name[0] for file_name in file_names]
    tickers = list(set(tickers))
    tickers.sort()
    return tickers


def get_fs(
    ticker: str,
    fs: str,
    term: str,
    levels: list[int] = [0, 1, 2],
    name: str = 'NameEn',
    raw_path: str = '../financeInfo/',
    ticker_col: str = 'Ticker',
    time_col: str = 'Feat_Time'
) -> pd.DataFrame:
    '''Get financial statement of a ticker.
    Financial statement can be 'KQKD', 'CDKT', 'LC', 'CSTC' or 'CTKH'.
    '''

    if fs not in ['KQKD', 'CDKT', 'LC', 'CSTC', 'CTKH']:
        raise ValueError("fs must be one of ['KQKD', 'CDKT', 'LC', 'CSTC', 'CTKH'].")
    if term not in ['Quarter', 'Annual']:
        raise ValueError("term must be one of ['Quarter', 'Annual'].")
    if name not in ['NameEn', 'Name', 'ReportNormID', 'ID']:
        raise ValueError("name must be one of ['NameEn', 'Name', 'ReportNormID, 'ID'].")

    file_alike = f'{ticker}_{fs}_{term}_'  # string of ticker, fs and term
    all_files = os.listdir(raw_path)  # all file names
    files = [file for file in all_files if file.startswith(file_alike)]  # select file names
    paths = [f'{raw_path}{file}' for file in files]  # create paths from selected file names

    if len(paths) == 0:
        return None

    fs_df_terms = []  # list of terms (pages), corresponding to json files
    for path in paths:
        with open(path) as f:
            json_raw = json.load(f)

        # only the first two elements of the json are matter
        jason_fs = json_raw[1]
        jason_meta = json_raw[0]

        try:  # KQKD, CDKT, LC, CSTC
            fs_names = list(jason_fs.keys())
            # KQKD, CDKT, LC -> return list of one element
            # CSTC -> return list of 10 elements, which are groups of ratios.
            # E.g. Profitability ratios, Liquidity ratios, etc
            # CTKH -> AttributeError: 'list' object has no attribute 'keys'
            fs_df_names = []
            for fs_name in fs_names:  # loop because fs_names may have many elements (i.e. CSTC)
                fs_df = pd.json_normalize(jason_fs[fs_name])
                fs_df = (
                    fs_df
                    .loc[lambda df: df['Levels'].isin(levels)]
                    .set_index(name)
                    .loc[:, 'Value1':'Value4']  # every json file contains four terms
                    .dropna(axis=1, how='all')  # some terms may not have any data
                    .transpose()
                )
                # change col names to terms
                terms = pd.json_normalize(jason_meta)['PeriodEnd'].astype(str)
                terms_no = len(fs_df)
                fs_df = fs_df.set_axis(terms.tail(terms_no), axis=0)
                fs_df_names.append(fs_df)  # append if there are many fs_names (CSTC)

            # concatenate if there are many fs_names (CSTC)
            fs_df_names = pd.concat(fs_df_names, axis=1)
            fs_df_terms.append(fs_df_names)  # append all the pages (json files)

        except AttributeError:  # CTKH
            fs_df = pd.json_normalize(jason_fs)
            fs_df = (
                fs_df
                .loc[lambda df: df['Levels'].isin(levels)]
                .set_index(name)
                .loc[:, 'Value1':'Value4']  # every json file contains four terms
                .dropna(axis=1, how='all')  # some terms may not have any data
                .transpose()
            )
            # change col names to terms
            terms = pd.json_normalize(jason_meta)['YearPeriod'].astype(str)
            terms_no = len(fs_df)
            fs_df = fs_df.set_axis(terms.tail(terms_no), axis=0)
            fs_df_terms.append(fs_df)  # append all the pages (json files)

    fs_df_terms = pd.concat(fs_df_terms, axis=0)  # concatenate all the pages (json files)
    # there is an error: '202112' should be '202012
    fs_df_terms.rename(index={'202112': '202012'}, inplace=True)
    # the pages (json files) are duplicated at the end terms
    fs_df_terms = fs_df_terms[~fs_df_terms.index.duplicated(keep='first')]
    fs_df_terms = fs_df_terms.sort_index(ascending=False)
    fs_df_terms = fs_df_terms.reset_index()
    fs_df_terms = fs_df_terms.rename(columns={'PeriodEnd': time_col, 'YearPeriod': time_col})
    fs_df_terms.insert(0, ticker_col, ticker)
    fs_df_terms.columns.name = None
    return fs_df_terms


class FSFeatures:

    @staticmethod
    def calculate_roll_mean(
        df: pd.DataFrame,
        window: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calculate moving average (roll mean) of financial statement.'''
        meta = df[meta_cols]
        roll_mean = (
            df
            .drop(meta_cols[1], axis=1)
            .groupby(meta_cols[0])
            .rolling(window)
            .mean()
            .shift(-window + 1)
            .reset_index(drop=True)
        )
        roll_mean = roll_mean.add_suffix(f'_Mean_{window}Q')
        roll_mean = pd.concat([meta, roll_mean], axis=1)
        return roll_mean

    @staticmethod
    def shift_data(
        df: pd.DataFrame,
        periods: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Shift financial statement periods quarters back to history.'''
        meta = df[meta_cols]
        shift = (
            df
            .drop(meta_cols[1], axis=1)
            .groupby(meta_cols[0])
            .shift(-periods)
            .reset_index(drop=True)
        )
        shift = pd.concat([meta, shift], axis=1)
        return shift

    @staticmethod
    def calculate_momentum(
        df: pd.DataFrame,
        window: int,
        periods: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calculate the growth rate (momentum) of financial statement'''
        meta = df[meta_cols]

        roll_mean = FSFeatures.calculate_roll_mean(df, window)
        shift = FSFeatures.shift_data(roll_mean, periods)

        roll_mean = roll_mean.drop(meta_cols, axis=1)
        shift = shift.drop(meta_cols, axis=1)

        momentum = roll_mean.div(shift)
        momentum = momentum.add_suffix(f'_Momen_{periods}Q')
        momentum = pd.concat([meta, momentum], axis=1)
        return momentum

    @staticmethod
    def calculate_momentum_loop(
        df: pd.DataFrame,
        window_list: list[int] = [1, 2, 4],
        periods_list: list[int] = [1, 2, 4],
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calulate growth rate (momentum) over different windows and periods settings.'''
        meta = df[meta_cols]
        roll_mean_momentum = []
        for window in window_list:
            for periods in periods_list:
                momentum_window_periods = FSFeatures.calculate_momentum(df, window, periods)
                momentum_window_periods = momentum_window_periods.drop(meta_cols, axis=1)
                roll_mean_momentum.append(momentum_window_periods)
        roll_mean_momentum = pd.concat([meta] + roll_mean_momentum, axis=1)
        return roll_mean_momentum

    @staticmethod
    def get_common_size(
        df: pd.DataFrame,
        master_col: str,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ):
        '''Calulate common size of financial statement.'''
        df_common = (
            df
            .set_index(meta_cols)
            .divide(df.set_index(meta_cols)[master_col], axis=0)
            .add_suffix('_Common')
            .reset_index()
        )
        return df_common

    @staticmethod
    def calculate_momentum_common(
        df: pd.DataFrame,
        window: int,
        periods: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''Calculate the growth rate (momentum) of financial statement common size'''
        meta = df[meta_cols]
        roll_mean = FSFeatures.calculate_roll_mean(df, window)
        shift = FSFeatures.shift_data(roll_mean, periods)
        roll_mean = roll_mean.drop(meta_cols, axis=1)
        shift = shift.drop(meta_cols, axis=1)

        momentum = roll_mean.subtract(shift)
        momentum = momentum.add_suffix(f'_Momen_{periods}Q')
        momentum = pd.concat([meta, momentum], axis=1)
        return momentum

    @staticmethod
    def calculate_momentum_common_loop(
        df: pd.DataFrame,
        window_list: list[int] = [1, 2, 4],
        periods_list: list[int] = [1, 2, 4],
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        '''
        Calulate growth rate (momentum) of financial statement common size
        over different windows and periods settings.
        '''
        meta = df[meta_cols]
        roll_mean_momentum = []
        for window in window_list:
            for periods in periods_list:
                momentum_window_periods = FSFeatures.calculate_momentum_common(df, window, periods)
                momentum_window_periods = momentum_window_periods.drop(meta_cols, axis=1)
                roll_mean_momentum.append(momentum_window_periods)
        roll_mean_momentum = pd.concat([meta] + roll_mean_momentum, axis=1)
        return roll_mean_momentum
