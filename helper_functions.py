import sklearn
from pandas.api.types import is_object_dtype

class preprocessor:
    def __init__(self, df, y_var, cols_to_drop = []):
        self.y = df[y_var].copy()
        cols_to_drop.append(y_var)
        print('Columns dropped to create X: ', cols_to_drop)
        if len(cols_to_drop) == 1:
            self.X = df.drop(columns = [y_var])
        else:
            self.X = df.drop(columns = cols_to_drop)
        self.enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.scaler = sklearn.preprocessing.StandardScaler(with_std=False)
    

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
        - enc: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') object

        returns:
        - encoded df as numpy.array
        """
        # Fit for onehot-encoder
        df_obj = self.X_train_df.select_dtypes(include = [object])
        self.enc.fit(df_obj)
        
        # Fit for standard scaler:
        df_num = self.X_train_df.select_dtypes(include = 'number')
        self.std = df_num.std(axis=0).fillna(1)
        self.mean = df_num.mean(axis=0).fillna(0)

    def __encode_transform_df(self, _df):
        """
        This function transforms the new df with the fitted encoder (one hot encoding).

        Takes:
        - _df: pandas.DataFrame
        - enc: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') object

        returns:
        - transformed df as numpy.array
        """

        df = _df.copy()
        df_obj = df.select_dtypes(include = [object])
        df.drop(columns = df_obj.columns, inplace=True)
        return np.concatenate((df.to_numpy(), self.enc.transform(df_obj).toarray()), axis = 1)
        

    def __standardise_df(self, _df):
        """
        This function transforms the new df with the fitted encoder (one hot encoding).

        Takes:
        - _df: pandas.DataFrame
        - enc: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') object

        returns:
        - transformed df as numpy.array
        """
        df = _df.copy()
        df_num = df.select_dtypes(include = 'number')
        df.drop(columns = df_num.columns, inplace=True)
        df_num = ((df_num - self.mean) / (self.std)).copy()
        df = df.join(df_num)
        return df


    def split_X_y(self, test_frac = 0.2):
        self.X_test_df = self.X.sample(frac = test_frac)
        self.X_train_df = self.X.drop(index = self.X_test_df.index, axis = 0)
        self.y_train_df = self.y.iloc[self.X_train_df.index]
        self.y_test_df = self.y.iloc[self.X_test_df.index]

    def encode_sample(self, _sample):
        sample = _sample.copy()
        sample = self.__standardise_df(sample)
        sample = self.__fillna(sample)
        sample = self.__encode_transform_df(sample)
        return sample

    def preprocess(self):
        self.split_X_y()
        self.__fit_df()
        self.X_train_df = self.__fillna(self.X_train_df).copy()
        self.X_train_df = self.__standardise_df(self.X_train_df).copy()
        self.X_train = self.__encode_transform_df(self.X_train_df).copy()
        self.X_test = self.encode_sample(self.X_test_df)