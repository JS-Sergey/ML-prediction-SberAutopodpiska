import dill
import pandas as pd


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """function to remove unnecessary features in the end"""

    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'device_screen_resolution',
        'device_browser',
        'device_model',
        'geo_country',
        'geo_city'
    ]

    return df.drop(columns_to_drop, axis=1)


def datetime_converter(df: pd.DataFrame) -> pd.DataFrame:
    """time data type changing function"""

    df = df.copy()

    df.visit_date = pd.to_datetime(df.visit_date, utc=True)
    df.visit_time = pd.to_datetime(df.visit_time, utc=True)

    return df


def geo_data_filler(df: pd.DataFrame) -> pd.DataFrame:
    """geo-data processing function"""

    def replace_numeric_cities(city):
        """number substitution function in city names"""

        if str(city).isnumeric():
            return '(not set)'
        else:
            return city

    df = df.copy()

    df['geo_city'] = df['geo_city'].apply(replace_numeric_cities)

    # If the city is not specified, fill in its coords with the coords of the corresponding country.
    df.loc[df['geo_city'] == '(not set)', 'lat'] = df['lat_с']
    df.loc[df['geo_city'] == '(not set)', 'long'] = df['long_с']

    # If neither country nor city is specified, fill in coords - Russian and Moscow coords respectively.
    df.loc[(df['geo_country'] == '(not set)') & (df['geo_city'] == '(not set)'),
    ['lat', 'long', 'lat_с', 'long_с']] = [55.750446, 37.617494, 64.686314, 97.745306]

    # If neither country nor city is specified, fill in with 'Russia' and 'Moscow'.
    df.loc[(df['geo_country'] == '(not set)') & (df['geo_city'] == '(not set)'),
    ['geo_country', 'geo_city']] = ['Russia', 'Moscow']

    return df


def category_processor(df: pd.DataFrame) -> pd.DataFrame:
    """aggregation function and filling in category blanks"""

    # upload additional files with lists of "big" categories.
    with open('data/additional_files/rare_camp_lst.pkl', 'rb') as file:
        nrare_camp_lst = dill.load(file)

    with open('data/additional_files/rare_sourc_lst.pkl', 'rb') as file:
        nrare_sourc_lst = dill.load(file)

    with open('data/additional_files/rare_keyw_lst.pkl', 'rb') as file:
        nrare_keyw_lst = dill.load(file)

    df = df.copy()

    # If the value is not in the list of 'large' categories then replace with 'rare'.
    df['utm_campaign'] = df['utm_campaign'].apply(lambda x: x if x in nrare_camp_lst else 'rare')
    df['utm_source'] = df['utm_source'].apply(lambda x: x if x in nrare_sourc_lst else 'rare')
    df['utm_keyword'] = df['utm_keyword'].apply(lambda x: x if x in nrare_keyw_lst else 'rare')

    # Fill in the blanks with mod for 'utm_source' and 'other' for the rest
    df['utm_source'] = df['utm_source'].fillna('ZpYIoDJMcFzVoPFsHGJL')
    df['utm_keyword'] = df['utm_keyword'].fillna('other')
    df['utm_campaign'] = df['utm_campaign'].fillna('other')
    df['utm_adcontent'] = df['utm_adcontent'].fillna('other')

    return df


def device_os_nan_replacer(df: pd.DataFrame) -> pd.DataFrame:
    """function to fill in blanks in device OS and brand"""

    df = df.copy()

    windows_brws_list = [
        'Edge',
        'Internet Explorer'
    ]

    # if the brand is 'Apple' then fill in 'iOS', and if the browser is Microsoft then 'Windows'.
    df.loc[df['device_os'].isna() & (df['device_brand'] == 'Apple'), 'device_os'] = 'iOS'
    df.loc[df['device_os'].isna() & (df['device_browser'].isin(windows_brws_list)), 'device_os'] = 'Windows'
    # the rest is 'Android' (mode).
    df['device_os'] = df['device_os'].fillna('Android')
    # replace the brand blanks with 'other'.
    df['device_brand'] = df['device_brand'].fillna('other')

    return df


def convert_cat_dateypes(df: pd.DataFrame) -> pd.DataFrame:
    """categorical features type changing function"""

    df = df.copy()

    cols_to_convert = [
        'utm_source',
        'utm_medium',
        'utm_campaign',
        'utm_adcontent',
        'utm_keyword',
        'device_category',
        'device_os',
        'device_brand'
    ]

    df[cols_to_convert] = df[cols_to_convert].astype('category')

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """new feature creation"""

    def short_browser(x):
        """browser short name creation function"""

        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    df = df.copy()
    # List of organic traffic types.
    organic_lst = [
        'organic',
        'referral',
        '(none)'
    ]
    # List of types of traffic from advertising in social networks.
    social_add_lst = [
        'QxAxdyPLuQMEcrdZWdWb',
        'MvfHsxITijuriZxsqZqt',
        'ISrKoXQCxqqYvAZICvjs',
        'IZEXUFLARCUMynmHNBGo',
        'PlbkrSYoHuZBWfYjYnfw',
        'gVRrcxiDQubJiljoTbGm'
    ]
    # The short name of the browser
    df.loc[:, 'short_browser'] = df['device_browser'].apply(short_browser).astype('category')
    # Features for traffic type and advertising.
    df.loc[:, 'organic_traffic'] = df['utm_medium'].apply(lambda x: 1 if x in organic_lst else 0).astype('int64')
    df.loc[:, 'social_adds'] = df['utm_source'].apply(lambda x: 1 if x in social_add_lst else 0).astype('int64')
    # Geo-features like Moscow/no, SPB/no, Russia/no.
    df.loc[:, 'moscow'] = df['geo_city'].apply(lambda x: 1 if x == 'Moscow' else 0).astype('int64')
    df.loc[:, 'saint_petersburg'] = df['geo_city'].apply(lambda x: 1 if x == 'Saint Petersburg' else 0).astype('int64')
    df.loc[:, 'russia'] = df['geo_country'].apply(lambda x: 1 if x == 'Russia' else 0).astype('int64')
    # Date and time features
    df.loc[:, 'visit_month'] = df.visit_date.dt.month
    df.loc[:, 'visit_dayofmonth'] = df.visit_date.dt.day
    df.loc[:, 'visit_dayofweek'] = df.visit_date.dt.weekday
    df.loc[:, 'visit_hour'] = df.visit_time.dt.hour

    # Device screen features - width, height, and total number of pixels.
    df.loc[:, 'screen_width'] = df.device_screen_resolution.apply(lambda x: x.split("x")[0]).astype('int64')
    df.loc[:, 'screen_height'] = df.device_screen_resolution.apply(lambda x: x.split("x")[1]).astype('int64')
    df.loc[:, 'pixel_total'] = df['screen_height'] * df['screen_width']
    # Define high- and low-resolution steamers for the screens.
    q_25 = df['pixel_total'].quantile(0.25)
    q_75 = df['pixel_total'].quantile(0.75)
    # Screen categories.
    df.loc[:, 'pixel_total_cat'] = df['pixel_total'].apply(lambda x: 0.0 if x < q_25 else (1.0 if x > q_75 else 0.5))

    return df
