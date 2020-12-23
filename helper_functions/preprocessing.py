import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

class preprocessor:
    def __init__(self, df, y_var, cols_to_drop = [], numbers_to_encode = [], method_to_encode = 'onehot_encode', numbers_to_onehot_encode = [], test_frac = 0.2):
        ### Tests
        assert not numbers_to_onehot_encode, 'Please change "numbers_to_onehot_encode" to "numbers_to_encode", thx.'
        assert method_to_encode in ['onehot_encode', 'label_encode'], 'For method_to_encode, only "onehot_encode" or "label_encode" is accepted.'
        assert y_var, 'Please define a y_var!'

        ###Save some parameters to class:
        self.y = df[y_var].copy()
        self.test_frac = test_frac
        self.numbers_to_encode = numbers_to_encode
        self.cols_to_drop = cols_to_drop

        ### Drop Columns based on y_var and cols_to_drop
        print('Columns dropped to create X: ', cols_to_drop)
        self.X = df.drop(columns = [y_var])
        self.X = self.__drop_columns(self.X)

        ### Cast Dtypes:
        self.X = self.__cast_dtypes(self.X)

        # Fit method for encoding:
        self.label_method = method_to_encode
        if method_to_encode == 'onehot_encode':
            self.labler = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        else:
            self.labler = sklearn.preprocessing.LabelEncoder()

    def __drop_columns(self, df_):
        """
        This function drops some columns based on self.cols_to_drop
        df_: Dataframe to drop columns from
        """
        df = df_.copy()
        X = df.drop(columns = self.cols_to_drop)
        return X

    def __cast_dtypes(self, X_):
        """
        This function casts some numeric columns based on self.numbers_to_encode and adds a _ to not allow pandas to cast dtypes back to numeric.
        df_: Dataframe to cast columns in
        """
        X = X_.copy()
        if len(self.numbers_to_encode)>0:
            X = X.astype({k:str for k in self.numbers_to_encode}).copy() # Update to allow the user to 
            print("Succesfully casted Dtypes!\n",X.dtypes)
            # Dirty workaround because some pandas operations change the dtype back to numbers if possible :(
            for col in self.numbers_to_encode:
                X.loc[:,col] = X.loc[:,col] + "_"
        return X

    def __fillna(self, _df):
        df = _df.copy()
        for col in df:
            #get dtype for column
            dt = df[col].dtype
            #check if it is a number
            if is_object_dtype(dt):
                df[col] = df[col].fillna("No Entry")
            else:
                df[col] = df[col].fillna(-1)
        return df


    def __fit_df(self):
        """
        This function fits (one hot encoding) the categorical columns if they are of type object and
        function fits (standardscaler) the numerical columns if they are of numbers (based on select_dtypes).
        Takes:
        - _df: pandas.DataFrame
        - labler: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') or sklearn.preprocessing.LabelEncoder()
        returns:
        - encoded df as numpy.array
        """
        # Fit for onehot-encoder
        df_obj = self.X_train_.select_dtypes(include = [object])
        if self.label_method =='onehot_encode':
            self.labler.fit(df_obj)
        else:
            self.labler.fit(df_obj.values.flatten())
        
        # Fit for standard scaler:
        df_num = self.X_train_.select_dtypes(include = 'number')
        self.std = df_num.std(axis=0).fillna(1)
        self.mean = df_num.mean(axis=0).fillna(0)

    def __encode_transform_df(self, _df):
        """
        This function transforms the new df with the fitted encoder (one hot encoding).

        Takes:
        - _df: pandas.DataFrame
        - labler: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') or sklearn.preprocessing.LabelEncoder()

        returns:
        - transformed df as numpy.array
        """

        df = _df.copy()
        # Create onehot encoded labels.
        df_obj = df.select_dtypes(include = [object])
        df.drop(columns = df_obj.columns, inplace=True)
        if self.label_method == 'onehot_encode':
            encoded = np.concatenate((df.to_numpy(), self.labler.transform(df_obj).toarray()), axis = 1)
            df_labeled_col_names = self.labler.get_feature_names(df_obj.columns.to_list())
        else:
            for col in df_obj.columns:
                df_obj[col] = self.labler.transform(df_obj[col])
            encoded = np.concatenate((df.to_numpy(), df_obj.to_numpy()), axis = 1)
            df_labeled_col_names = df_obj.columns.to_list()

        col_names = []
        col_names.extend(df.columns.to_list())
        col_names.extend(df_labeled_col_names)
        return pd.DataFrame(encoded, columns = col_names).convert_dtypes()

    def __standardise_df(self, _df):
        """
        This function transforms the new df with the fitted encoder (one hot encoding).

        Takes:
        - _df: pandas.DataFrame
        - labler: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') or sklearn.preprocessing.LabelEncoder()

        returns:
        - transformed df as numpy.array
        """
        df = _df.copy()
        df_num = df.select_dtypes(include = 'number').copy()
        df.drop(columns = df_num.columns, inplace=True)
        df_num = ((df_num - self.mean) / (self.std)).copy()
        df = df.join(df_num)
        return df


    def split_X_y(self, test_frac = 0.2):
        self.X_test_ = self.X.sample(frac = test_frac, random_state = 42)
        self.X_train_ = self.X.drop(index = self.X_test_.index, axis = 0)
        self.y_train = self.y.iloc[self.X_train_.index]
        self.y_test = self.y.iloc[self.X_test_.index]

    def encode_sample(self, _sample, test_data = False):
        sample = _sample.copy()
        sample = self.__fillna(sample)
        if test_data:
            sample = self.__drop_columns(sample)
            sample = self.__cast_dtypes(sample)
        sample = self.__standardise_df(sample)
        sample = self.__encode_transform_df(sample)
        return sample

    def preprocess(self):
        self.split_X_y(test_frac = self.test_frac)
        self.X_train_ = self.__fillna(self.X_train_).copy()
        self.__fit_df()
        self.X_train_ = self.__standardise_df(self.X_train_).copy()
        self.X_train = self.__encode_transform_df(self.X_train_).copy()
        self.X_test = self.encode_sample(self.X_test_)
