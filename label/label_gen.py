'''
This script is to generate label for stock trading.
'''
__author__ = 'Khanh Truong'
__date__ = '2021-08-22'

import os
from datetime import datetime

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy import stats


class Label:

    def __init__(self, tickers=None):
        self.tickers = tickers
        self.raw_df = None
        self.label_df = None

    def get_tickers(self, folder='../data/excelfull'):
        '''
        Get all tickers in a folder

        Parameters
        ----------
        folder : str
            Path to folder wanted to get tickers. Ex: '../data/excelfull'

        Returns
        -------
        self : Label
            Label with tickers (list of strings)
        '''
        file_names = pd.Series(os.listdir(folder))
        file_names = file_names.sort_values().str.split('_')
        tickers = [file_name[0] for file_name in file_names]
        tickers = [ticker for ticker in tickers if len(ticker) == 3]
        self.tickers = tickers
        return self

    def get_raw(self, folder='../data/excelfull'):
        '''
        Get raw price data

        Parameters
        ----------
        folder: list of strings
            List of tickers. Ex: ['TCB', 'CTG', 'GAS']

        Returns
        -------
        self : Label
            Label with raw of prices (pandas.DataFrame)
        '''
        paths = []
        for ticker in self.tickers:
            paths.append(f'{folder}/{ticker}_excelfull.csv')

        raw_df = []
        for path in paths:
            raw = pd.read_csv(path)
            raw_df.append(raw)
        raw_df = pd.concat(raw_df).reset_index(drop=True)
        self.raw_df = raw_df
        return self

    @staticmethod
    def get_last_dates(
        last_date,
        gap=3,
        periods=100,
        input_format='%Y%m%d',
        out_format='%Y%m%d'
    ):
        '''
        Get list of last dates of months in every periods month

        Parameters
        ----------
        last_date : str
            Input last date. Ex: '20210731'
        gap : int
            Gap in months between periods. Ex: 3
        periods : int
            Number of periods wanted to look back. Ex: 100
        input_format : str
            Format of input date. Ex: '%Y%m%d'
        output_format : str
            Format of output dates. Ex: '%Y%m%d'

        Returns
        -------
        list of str
            List of last dates. Ex: ['20210731', '20210430', '20210131', ...]
        '''
        last_date = datetime.strptime(str(last_date), input_format)
        dates = []
        for i in range(periods):
            date = last_date - relativedelta(months=i * gap)
            date = date.replace(day=28) + relativedelta(days=4)
            date = date - relativedelta(days=date.day)
            date = date.strftime(out_format)
            dates.append(date)
        return dates

    @staticmethod
    def group_dates(input_dates, last_date, term=1, gap=3):
        '''
        Get list of last dates of months in every periods month

        Parameters
        ----------
        input_dates : list or series of str
            Input dates. Ex: ['20210731', '20210730', '20210729', ...]
        last_date : str
            Last date that wanted to group to. Ex: '20210731'
        term : int
            Term in months from last_date. Ex: 1
            Term must not be greater than gap.
        gap : int
            Gap in months between periods. Ex: 3

        Returns
        -------
        list of str
            List of last dates. Ex: ['20210731', '20210731', '20210731', ...]
        '''
        # covert input dates to datetime64[D]
        input_dates = [datetime.strptime(str(input_date), '%Y%m%d') for input_date in input_dates]
        input_dates = [datetime.strftime(input_date, '%Y-%m-%d') for input_date in input_dates]
        input_dates = np.array(input_dates, dtype='datetime64[D]')

        # covert last dates to datetime64[D]
        last_dates = Label.get_last_dates(last_date, gap=gap)
        last_dates = [datetime.strptime(str(last_date), '%Y%m%d') for last_date in last_dates]
        last_dates = [datetime.strftime(last_date, '%Y-%m-%d') for last_date in last_dates]
        last_dates = np.array(last_dates, dtype='datetime64[D]')

        # calculate gap between last dates and input dates
        # the output is an 2D array whose shape is len(input_dates) x len(last_dates)
        day_from_last_to_input = np.subtract.outer(last_dates, input_dates)
        day_from_last_to_input = np.array(day_from_last_to_input, dtype='int')

        # replace any gap between 0 and 30 days by the corresponding last date
        # otherwise, replace by 0 (later, the array will be summed by axis=0)
        for i, last_date in enumerate(last_dates):
            condition = (0 <= day_from_last_to_input[i]) & (day_from_last_to_input[i] < term * 30)
            day_from_last_to_input[i][condition] = int(str(last_date).replace('-', ''))
            day_from_last_to_input[i][~condition] = 0

        # sum the array
        group_dates = np.sum(day_from_last_to_input, axis=0)
        group_dates = [str(group_date) if group_date != 0 else np.nan for group_date in group_dates]
        return group_dates

    def __get_mean_price(self, last_date, label_col, term=1, gap=3):
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
            self.raw_df
            # group dates
            .rename(columns={'<Ticker>': 'Ticker', '<DTYYYYMMDD>': 'Label_Time'})
            .assign(Label_Time=lambda df: (
                self.group_dates(
                    df['Label_Time'],
                    last_date=last_date,
                    term=term,
                    gap=gap)))
            # calculate mean closing prices
            .groupby(['Ticker', 'Label_Time'])
            .agg({'<Close>': 'mean'})
            .reset_index()
            .rename(columns={'<Close>': label_col})
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
