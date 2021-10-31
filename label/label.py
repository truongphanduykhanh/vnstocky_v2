'''
This script is to generate label for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-10-31'

import os
from datetime import datetime

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple


def get_last_date_of_month(
    month: Union[str, int],
    look_back_periods: int = 1,
    gap: int = 1,
    input_format: str = '%Y%m',
    output_format: str = '%Y%m%d'
) -> list[str]:
    """Get last date of a month.

    Args:
        month (Union[str, int]): Input month. Ex: '202110'.
        look_back_periods (int, optional): Number of periods wanted to look back. Defaults to 1.
        gap (int, optional): Gap in months between periods. Defaults to 1.
        input_format (str, optional): Format of input month. Defaults to '%Y%m'.
        output_format (str, optional): Format of output date. Defaults to '%Y%m%d'.

    Returns:
        list[str]: List of last dates. Ex: ['20210731', '20210430', '20210131', ...]
    """
    month = datetime.strptime(str(month), input_format)
    last_dates = []
    for i in range(look_back_periods):
        date = month - relativedelta(months=i * gap)
        date = date.replace(day=28) + relativedelta(days=4)
        date = date - relativedelta(days=date.day)
        date = date.strftime(output_format)
        last_dates.append(date)
    if len(last_dates) == 1:
        last_dates = last_dates[0]
    return last_dates


def group_dates(
    dates: list[Tuple[str, int]],
    last_month: Tuple[str, int],
    gap: int,
    term: int,
    dates_month_format: str = '%Y%m%d'
) -> list[Union[str, float]]:
    """Group dates into last dates of periods.

    Args:
        dates (list[Tuple[str, int]]): Input dates. Ex: ['20210731', '20210730', ...]
        last_month (Tuple[str, int]): Last month wanted to group.
        gap (int): Gap in months between periods.
        term (int): Term in months from last_date. Term must not be greater than gap.
        dates_month_format (str, optional): Format of dates. Defaults to '%Y%m%d'.

    Returns:
        list[Union[str, float]]: Groups of dates. Ex: ['20210731', '20210731', ...]
    """
    # covert input dates to datetime64[D]
    if '%d' in dates_month_format:
        dates = [datetime.strptime(str(input_date), dates_month_format) for input_date in dates]
        dates = [datetime.strftime(input_date, '%Y-%m-%d') for input_date in dates]
    else:  # in case input dates are months
        dates = [
            get_last_date_of_month(
                month=date,
                input_format=dates_month_format,
                output_format='%Y-%m-%d'
            ) for date in dates]
    dates = np.array(dates, dtype='datetime64[D]')

    # create list of last dates and covert to datetime64[D]
    last_dates = get_last_date_of_month(
        month=last_month,
        look_back_periods=100,
        gap=gap,
        input_format=dates_month_format,
        output_format='%Y-%m-%d'
    )
    last_dates = np.array(last_dates, dtype='datetime64[D]')

    # calculate gap between last dates and input dates
    # the output is an 2D array whose shape is len(input_dates) x len(last_dates)
    day_from_last_to_input = np.subtract.outer(last_dates, dates)
    day_from_last_to_input = np.array(day_from_last_to_input, dtype='int')

    # replace any gap between 0 and term * 30 days by the corresponding last date
    # otherwise, replace by 0 (later, the array will be summed by axis=0)
    for i, last_date in enumerate(last_dates):
        condition = (0 <= day_from_last_to_input[i]) & (day_from_last_to_input[i] <= term * 30 - 1)
        day_from_last_to_input[i][condition] = int(str(last_date).replace('-', ''))
        day_from_last_to_input[i][~condition] = 0

    # sum the array
    group_dates = np.sum(day_from_last_to_input, axis=0)
    if '%d' in dates_month_format:
        group_dates = [str(group_date) if group_date != 0 else np.nan for group_date in group_dates]
    else:
        group_dates = [str(group_date)[0:6] if group_date != 0 else np.nan for group_date in group_dates]
    return group_dates


def get_mean_price(
    raw_df: pd.DataFrame,
    last_month: Tuple[str, int],
    last_month_format: str = '%Y%m%d',
    gap: int = 3,
    term: int = 1,
    ticker_col: str = '<Ticker>',
    date_col: str = '<DTYYYYMMDD>',
    date_col_format: str = '%Y%m%d',
    price_col: str = '<Close>'
) -> pd.DataFrame:
    '''
    Get label data frame from raw data.

    Parameters
    ----------
    last_date : str
        Last date that wanted to group to. Ex: '20210731'
    label_col : str
        Column name of label. Ex: 'Label_Norminator'
    term : int
        Term in months from last_date. Ex: 1
        Term must not be greater than gap.
    gap : int
        Gap in months between periods. Ex: 3

    Returns
    -------
    mean_price : pandas.DataFrame
        Data frame that has mean price of each ticker at each date group.
        Ex. columns names: ['Ticker', 'Label_Time', 'Label_Norminator']
    '''
    mean_price = (
        raw_df
        # group dates
        .assign(Label_Time=lambda df: (
            group_dates(df[date_col], last_month=last_month, gap=gap, term=term)))
        # calculate mean closing prices
        .groupby([ticker_col, 'Label_Time'])
        .agg({price_col: 'mean'})
        .sort_index(ascending=[True, False])
        .reset_index()
        .rename(columns={price_col: 'Label', ticker_col: 'Ticker'})
    )
    return mean_price

    def get_label(
        self,
        last_date_nominator='20210731',
        last_date_denominator='20210131',
        term_nominator=1,
        term_denominator=1,
        gap=6,
        id_col='Ticker',
        label_time_col='Label_Time',
        feat_time_col='Feat_Time',
        label_col='Return'
    ):
        '''
        Get label data frame from raw data.

        Parameters
        ----------
        last_date_nominator : str
            Last date of nominator that wanted to group to. Ex: '20210731'
        last_date_denominator : str
            Last date of denominator that wanted to group to. Ex: '20210430'
        term_nominator : int
            Term in months from last_date_nominator. Ex: 1
            Term must not be greater than gap.
        term_denominator : int
            Term in months from last_date_denominator. Ex: 1
            Term must not be greater than gap.
        gap : int
            Gap in months between periods. Ex: 3
        id_col : str
            Column name of tickers. Ex: 'Ticker'
        label_time_col : str
            Column name of label time. Ex: 'Label_Time'
        feat_time_col : str
            Column name of feature time. Ex: 'Feat_Time'
        label_col : str
            Column name of label. Ex: 'Return'

        Returns
        -------
        self : Label
            Label with final label data (pandas.DataFrame)
            Ex. columns names: ['Ticker', 'Label_Time', 'Feat_Time', 'Return']
        '''
        nominator = self.__get_mean_price(
            last_date=last_date_nominator,
            label_col='Label_Nominator',
            term=term_nominator,
            gap=gap)
        denominator = self.__get_mean_price(
            last_date=last_date_denominator,
            label_col='Label_Denominator',
            term=term_denominator,
            gap=gap)

        label_df = (
            nominator
            .merge(denominator, how='left', on=['Ticker', 'Label_Time'])

            # add lag prices columne (denominator column)
            .sort_values(['Ticker', 'Label_Time'], ascending=[True, False])
            .assign(Label_Denominator=lambda df: (
                df.groupby('Ticker')['Label_Denominator'].shift(-1)))

            # calculate return
            .assign(Label=lambda df: (
                df['Label_Nominator'] / df['Label_Denominator'] - 1))

            # add feature time column for later reference
            .assign(Feat_Time=lambda df: (
                df.groupby('Ticker')['Label_Time'].shift(-1)))
            .dropna()  # drop the last record, which is NA because of no denominator

            # adjusted with market return
            .assign(Label_Market=lambda df: df.groupby('Label_Time')['Label'].transform(stats.trim_mean, 0.1))
            .assign(Label_Normalized=lambda df: df['Label'] - df['Label_Market'])

            # clean data
            .loc[:, ['Ticker', 'Label_Time', 'Feat_Time', 'Label_Normalized']]
            .astype({'Label_Time': str, 'Feat_Time': str})
            .rename(columns={
                'Ticker': id_col,
                'Label_Time': label_time_col,
                'Feat_Time': feat_time_col,
                'Label_Normalized': label_col})
            .reset_index(drop=True)
        )
        self.label_df = label_df
        return self

    def export(self, path='label_six_months.csv'):
        self.label_df.to_csv(path, index=False)


if __name__ == '__main__':
    label = Label()
    label.get_tickers()
    label.get_raw()
    label.get_label()
    label.export()
