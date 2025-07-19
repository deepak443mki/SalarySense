import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean(path):
    df = pd.read_csv(path)
    # Replace '?' with NaN and drop missing
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df


def feature_engineer(df):
    # Create age bins
    df['age_bin'] = pd.cut(df['age'], bins=[17, 25, 35, 45, 55, 90],
                           labels=['18-25','26-35','36-45','46-55','56+'])
    # Gain/Loss flags
    df['has_capital_gain'] = (df['capital-gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital-loss'] > 0).astype(int)
    # Map education to educational-num
    edu_map = {'Preschool':1, '1st-4th':2, '5th-6th':3, '7th-8th':4,
               '9th':5, '10th':6, '11th':7, '12th':8, 'HS-grad':9,
               'Some-college':10, 'Assoc-voc':11, 'Assoc-acdm':12,
               'Bachelors':13, 'Masters':14, 'Prof-school':15, 'Doctorate':16}
    df['educational-num'] = df['education'].map(edu_map)
    return df


def split_data(df, target='income'):
    X = df.drop(columns=[target])
    y = (df[target] == '>50K').astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)