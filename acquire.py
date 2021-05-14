import pandas as pd
import env
import os
# must have env.py saved in same directory as script. ensure the env.py is in your .gitignore
def get_connection(db, user=env.username, host=env.host, password=env.password):
    '''
    This function makes a connection with and pulls from the CodeUp database. It 
    takes the database name as its argument, pulls other login info from env.py.
    Make sure you save this as a variable or it will print out your sensitive user
    info as plain text. 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
# these functions below will pull dbs as dataframes from the codeup database if it hasnt already
# if it has, it will write a cache. 
def get_titanic_data():
    '''
    This function pulls the titanic db into a dataframe if it doesnt exist, or it will cache 
    the titanic data into a .csv file. 
    '''
    filename = 'titanic.csv'
    if os.path.isfile(filename):
        titanic_df = pd.read_csv(filename, index_col=0)
        return titanic_df
    else:
        titanic_df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        titanic_df.to_csv(filename, mode = 'w+')
        return titanic_df
    
def get_iris_data():
    ''' This function pulls the iris db into a dataframe, or caches it as a .csv if it hasnt been already.
    '''
    filename = 'iris.csv'
    if os.path.isfile(filename):
        iris_df = pd.read_csv(filename, index_col=0)
        return iris_df
    else:
        iris_df = pd.read_sql('SELECT * FROM measurements JOIN species USING(species_id);', get_connection('iris_db'))
        iris_df.to_csv(filename)
        return iris_df