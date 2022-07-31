import pickle
import inflection
import math
import datetime
import pandas as pd
import numpy as np


class Pharmaceutical(object):
    def __init__(self):
        self.home_path = 'C:\\Users\\mathe\\OneDrive\\Documentos\\Pharmaceutical_Sales\\'
        self.competition_distance_scaler = pickle.load(
            open(self.home_path + 'parameter\\competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(
            open(self.home_path + 'parameter\\competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.home_path + 'parameter\\promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler = pickle.load(open(self.home_path + 'parameter\\year_scaler.pkl', 'rb'))
        self.store_type_scaler = pickle.load(open(self.home_path + 'parameter\\store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, df):
        # Rename columns - snake case
        cols_camelcase = list(df.columns)
        snake_case = lambda x: inflection.underscore(x)
        cols_snake_case = list(map(snake_case, cols_camelcase))

        df.columns = cols_snake_case
        df = df.drop(['sales','customers'], axis=1)

        # Converting variable 'date' to type datetime
        df['date'] = pd.to_datetime(df['date'])

        # Repalcing 0 with '0'
        df['state_holiday'] = df['state_holiday'].replace(0, '0')

        # Replacing NA with a very high distance == no near competition
        df['competition_distance'] = df['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        # Replacing NA with months == if diference between 'month store' and 'month open' equal zero, no competition open since month
        df['competition_open_since_month'] = df.apply(
            lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x[
                'competition_open_since_month'], axis=1)

        # Replacing NA with year == if diference between 'year store' and 'year open' equal zero, no competition open since year
        df['competition_open_since_year'] = df.apply(
            lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x[
                'competition_open_since_year'], axis=1)

        # Replacing NA with week == if diference between 'week store' and 'week since' equal zero, no promo participate since week
        df['promo2_since_week'] = df.apply(
            lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # Replacing NA with year == if diference between 'year store' and 'year since' equal zero, no promo participate since year
        df['promo2_since_year'] = df.apply(
            lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # Replacing NA with 0 = no promo
        df['promo_interval'] = df['promo_interval'].fillna(0)

        # Changing data type
        # competiton as int64
        df['competition_open_since_month'] = np.int64(df['competition_open_since_month'])
        df['competition_open_since_year'] = np.int64(df['competition_open_since_year'])

        # promo2 as int64
        df['promo2_since_week'] = np.int64(df['promo2_since_week'])
        df['promo2_since_year'] = np.int64(df['promo2_since_year'])

        return df

    def feature_engineering(self, df2):
        # Years
        df2['year'] = df2['date'].dt.year

        # Month
        df2['month'] = df2['date'].dt.month

        # Semester
        df2['semester'] = df2['month'].apply(lambda x: '1st_semester' if x < 7 else '2nd_semester')

        # Year and week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # day
        df2['day'] = df2['date'].dt.day

        # month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        # df['month_map'] = df['date'].dt.month.map(month_map)

        # Assortment names
        df2['assortment'] = df2['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # Holidays names
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # Months since competitor open
        df2['competition_since'] = df2.apply(
            lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'],
                                        day=1), axis=1)
        df2['competition_time_month'] = np.int64(
            ((df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days))

        # Weeks since promo is active
        df2['promo_since'] = (df2['promo2_since_year']).astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = np.int64(((df2['date'] - df2['promo_since']) / 7).apply(lambda x: x.days))

        # Only sales
        df2 = df2[(df2['open'] != 0) & (df2['sales'] > 0)]

        # Removing unnecessary variables
        cols_drop = ['customers', 'open', 'promo_interval']
        df2 = df2.drop(cols_drop, axis=1)

        return df2

    def data_preparation(self, df5):
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(
            df5[['competition_distance']].values)

        # competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(
            df5[['competition_time_month']].values)

        # promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)

        # year
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        # state holiday
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store type
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])

        # assortment
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # responde variable
        df5['sales'] = np.log1p(df5['sales'])

        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi / 7)))
        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi / 12)))
        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi / 30)))
        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance',
                         'competition_open_since_month','competition_open_since_year', 'promo2',
                         'promo2_since_week', 'promo2_since_year',
                         'competition_time_month', 'promo_time_week',
                         'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                         'week_of_year_sin', 'week_of_year_cos']

        return df5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        return original_data.to_json(orient='records', date_format='iso')
