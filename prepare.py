import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prep_iris(iris_df):
    '''
    Takes in the iris_df, drops species id, and measurement id, then adds a dummy
    variable column. It then returns the iris_df cleaned as iris_df
    '''
    iris_df = iris_df.drop(columns = ['species_id', 'measurement_id'])
    iris_df = iris_df.rename(columns = {'species_name':'species'})
    #iris_df = pd.get_dummies(data = iris_df, columns = ['species'], drop_first=True)
    return iris_df

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

# # Remember to delte your titanic .csv if you've already run this, or else it will break 
def impute_mode(train, validate, test):
    '''
    impute mode for embark_town
    '''
    imputer = SimpleImputer(strategy='most_frequent', missing_values = None)
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test
    
def prep_titanic_data(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: train, test, split: three dataframes with the cleaning operations performed on them
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age', 'passenger_id'])
    train, test = train_test_split(df, test_size=0.2, random_state=1349, stratify=df.survived)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349, stratify=train.survived)

    train, validate, test = impute_mode(train, validate, test)
    train = pd.get_dummies(data = train, columns = ['sex', 'embark_town'], drop_first=[True,True])
    validate = pd.get_dummies(data = validate, columns = ['sex', 'embark_town'], drop_first=[True,True])
    test = pd.get_dummies(data = test, columns = ['sex', 'embark_town'], drop_first=[True,True])
    return train, validate, test