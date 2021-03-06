"""This script is to generate features data for stock trading.
"""
__author__ = 'Khanh Truong'
__date__ = '2021-09-20'


import os

import pandas as pd
import json


def get_tickers(folder: str = '../financeInfo') -> list[str]:
    """Get all tickers in a folder.

    Args:
        folder (str, optional): Path to folder wanted to get tickers. Defaults to '../financeInfo'.

    Returns:
        list[str]: List of tickers. Ex: ['A32', 'AAM', 'AAT', ...].
    """
    file_names = pd.Series(os.listdir(folder))
    file_names = file_names.sort_values().str.split('_')
    tickers = [file_name[0] for file_name in file_names]
    tickers = list(set(tickers))
    tickers.sort()
    return tickers


def get_fs(
    ticker: str,
    fs: str,
    term: str = 'Quarter',
    levels: list[int] = [0, 1, 2],
    name: str = 'NameEn',
    raw_path: str = '../financeInfo/',
    ticker_col: str = 'Ticker',
    time_col: str = 'Feat_Time'
) -> pd.DataFrame:
    """Get financial statement of a ticker.

    Args:
        ticker (str): Company ticker that want to get financial statement for.
        fs (str): Type of financial statement. Can be 'KQKD', 'CDKT', 'LC', 'CSTC' or 'CTKH'.
        term (str): 'Quarter' or 'Annual'. Defaults to 'Quarter'.
        levels (list[int], optional): The headline levels in financial statement that want to get.
            The more levels, the more detail. Ranging from 0 to 6. Only applicable for 'CDKT'.
            Defaults to [0, 1, 2].
        name (str, optional): Headline languages. 'NameEn' for English. 'Name' for Vietnamese.
            'ReportNormID' for code. 'ID' for 1, 2, 3, etc. Defaults to 'NameEn'.
        raw_path (str, optional): Path to folder stored the raw files.
            Defaults to '../financeInfo/'.
        ticker_col (str, optional): Name of the column ticker in output dataframe.
            Defaults to 'Ticker'.
        time_col (str, optional): Name of the column time in output dataframe.
            Defaults to 'Feat_Time'.

    Raises:
        ValueError: Choose one of ['KQKD', 'CDKT', 'LC', 'CSTC', 'CTKH']
        ValueError: Choose one of ['Quarter', 'Annual']
        ValueError: Choose one of ['NameEn', 'Name', 'ReportNormID', 'ID']

    Returns:
        pd.DataFrame: financial statement
    """
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
            # some tables have duplicated columns. Firstly, remove column with all NA,
            # Then keep the first duplicated column if still duplicated.
            fs_df_names = fs_df_names.dropna(axis=1)
            fs_df_names = fs_df_names.loc[:, ~fs_df_names.columns.duplicated()]
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
            # some tables have duplicated columns. Firstly, remove column with all NA,
            # Then keep the first duplicated column if still duplicated.
            fs_df = fs_df.dropna(axis=1)
            fs_df = fs_df.loc[:, ~fs_df.columns.duplicated()]
            fs_df_terms.append(fs_df)  # append all the pages (json files)

    fs_df_terms = pd.concat(fs_df_terms, axis=0)  # concatenate all the pages (json files)
    # there is an error: '202112' should be '202012
    fs_df_terms.rename(index={'202112': '202012'}, inplace=True)
    # the pages (json files) are duplicated at the end terms
    fs_df_terms = fs_df_terms[~fs_df_terms.index.duplicated(keep='first')]
    fs_df_terms = fs_df_terms.loc[:, ~fs_df_terms.columns.duplicated(keep='first')]
    fs_df_terms = fs_df_terms.sort_index(ascending=False)
    fs_df_terms = fs_df_terms.reset_index()
    fs_df_terms = fs_df_terms.rename(columns={'PeriodEnd': time_col, 'YearPeriod': time_col})
    fs_df_terms.insert(0, ticker_col, ticker)
    fs_df_terms.columns.name = None
    return fs_df_terms


def get_fs_multiple(tickers: list[str], fs: list[str]) -> pd.DataFrame:
    """Return multiple financial statements of many tickers at once.

    Args:
        tickers (list[str]): tickers.
        fs (list[str]): names of financial statements.

    Raises:
        ValueError: Choose among ['KQKD', 'CDKT', 'LC', 'CSTC', 'CTKH'].

    Returns:
        pd.DataFrame: Financial statements of many tickers.
    """
    if set(fs) > set(['KQKD', 'CDKT', 'LC', 'CSTC', 'CTKH']):
        raise ValueError("fs must be among ['KQKD', 'CDKT', 'LC', 'CSTC', 'CTKH'].")
    fs_multiple = []
    for fs_str in fs:
        print(f'Getting {fs_str}...')
        fs_dfs = []
        for ticker in tickers:
            fs_df = get_fs(ticker=ticker, fs=fs_str)
            fs_dfs.append(fs_df)
        fs_dfs = pd.concat(fs_dfs, axis=0).set_index(['Ticker', 'Feat_Time'])
        fs_multiple.append(fs_dfs)
    fs_multiple = pd.concat(fs_multiple, axis=1).sort_index(ascending=[True, False]).reset_index()
    return fs_multiple


class FSFeatures:

    @staticmethod
    def calculate_roll_mean(
        df: pd.DataFrame,
        window: int,
        meta_cols: list[str] = ['Ticker', 'Feat_Time']
    ) -> pd.DataFrame:
        """Calculate moving average (roll mean) of financial statement.

        Args:
            df (pd.DataFrame): Features data.
            window (int): Number of periods to calculate the average mean.
            meta_cols (list[str], optional): Columns to be excluded from calculating average mean.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Moving average features.
        """
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
        """Shift financial statement periods quarters back to history.

        Args:
            df (pd.DataFrame): Data shifted.
            periods (int): Number of periods shifted.
            meta_cols (list[str], optional): Columns that to be excluded from the shifted.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Shifted data.
        """
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
        """Calculate the growth rate (momentum) of financial statement.

        Args:
            df (pd.DataFrame): Data to be calculated the momentum.
            window (int): Number of periods in rolling mean.
            periods (int): Number of gap periods to be divided.
            meta_cols (list[str], optional): Columns to be excluded from the calculation.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Momentum data.
        """
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
        """Calulate growth rate (momentum) over different windows and periods settings.

        Args:
            df (pd.DataFrame): Data to be calculated the momentum.
            window_list (list[int], optional): Number of periods in rolling mean.
                Defaults to [1, 2, 4].
            periods_list (list[int], optional): Number of gap periods to be divided.
                Defaults to [1, 2, 4].
            meta_cols (list[str], optional): Columns to be excluded from the calculation.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Momentum data.
        """
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
    ) -> pd.DataFrame:
        """Calulate common size of financial statement.

        Args:
            df (pd.DataFrame): Financial statement.
            master_col (str): Denominator column.
            meta_cols (list[str], optional): Columns to be excluded from the calculation.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Common size data.
        """
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
        """Calculate the growth rate (momentum) of financial statement common size.

        Args:
            df (pd.DataFrame): Data to be calculated the momentum.
            window (int): Number of periods in rolling mean.
            periods (int): Number of gap periods to be deducted.
            meta_cols (list[str], optional): Columns to be excluded from the calculation.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Momentum data.
        """
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
        """ Calulate growth rate (momentum) of financial statement common size over different
            windows and periods settings.

        Args:
            df (pd.DataFrame): Data to be calculated the momentum.
            window_list (list[int], optional): Number of periods in rolling mean.
                Defaults to [1, 2, 4].
            periods_list (list[int], optional): Number of gap periods to be deducted.
                Defaults to [1, 2, 4].
            meta_cols (list[str], optional): Columns to be excluded from the calculation.
                Defaults to ['Ticker', 'Feat_Time'].

        Returns:
            pd.DataFrame: Momentum data.
        """
        meta = df[meta_cols]
        roll_mean_momentum = []
        for window in window_list:
            for periods in periods_list:
                momentum_window_periods = FSFeatures.calculate_momentum_common(df, window, periods)
                momentum_window_periods = momentum_window_periods.drop(meta_cols, axis=1)
                roll_mean_momentum.append(momentum_window_periods)
        roll_mean_momentum = pd.concat([meta] + roll_mean_momentum, axis=1)
        return roll_mean_momentum
