############################################################################################
# utility.py
# 
# This file contains utility functions for the ER Wait Time notebooks, as well as
# constants used across the entire project's Notebook files.
#
# (c) Julie Leung, 2023 except for code snippets where credit is given to others
############################################################################################

# Library Imports

import pandas as pd
from IPython.display import display
import requests
import matplotlib.pyplot as plt
from matplotlib import colormaps
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Utility:
    ############################################################################################
    # Define constants
    ############################################################################################

    # Random Seed to be used in all models
    RANDOM_STATE_CONST = 42

    # Paths for base raw and fully cleaned data
    RAW_INFO_FILENAME = '../data/Alberta Hospital Info Nov32023.csv'
    CLEAN_INFO_FILENAME = '../data/cleandata/hospital_info_clean.csv'
    RAW_WAIT_FILENAME = '../data/Alberta Hospital Wait Time Data_Nov22_2023_raw.csv'
    CLEAN_WAIT_FILENAME = '../data/cleandata/df_final_er_wait_time_nov_22_2023.csv'

    # Not as importnant: paths to the intermediary datacleaning DFs that are not typically used in notebooks other than cleaning notebook
    RAW_WAIT_DATETIME_DROP_NULL_HOSPITALS_DF_FILENAME = '../data/cleandata/df_er_wait_time_nov_22_2023.csv'
    RAW_WAIT_INITIAL_MELTED_DF_MELTED_FILENAME = '../data/cleandata/df_melted_er_wait_time_nov_22_2023.csv'
    RAW_WAIT_DROPNULL_DATETIME_FEATURE_ENG_DF_CLEAN_FILENAME = '../data/cleandata/df_clean_er_wait_time_nov_22_2023.csv'
    
    # Endpoint for the Geocoding API
    GEOCODE_ENDPOINT = 'https://geocode.maps.co/'

    # dayperiod bins correspond to the following, per guidance from ER clinician-researcher Principal Investigator
    # night   : 00h00 to 07h59, inclusive
    # day     : 08h00 to 15h59, inclusive
    # evening : 16h00 to 23h59, inclusive
    DAYPERIOD_BINS_LABELS = {
        'labels' : ['night', 'day', 'evening'],
        'bins' : [-1, 7, 15, 23]
    }

    # Definition for the 'target' variable 'longwait'
    LONGWAIT_MINUTES = 400 # Chosen based on guidance from CAEP and CTAS guidelines

    # Column definitions for various data treatments for modelling experiments
    LOGIT_DROP_COLS = ['datetime', 'hospital', 'waittime', 'year', 'numdayofweek', 'services', 'cityarea', 'citypop', 'hosplat', 'hosplong']
    LOGIT_ONEHOT_COLS_A = ['dayofweek', 'dayperiod', 'id', 'city', 'citytype']
    LOGIT_ONEHOT_COLS_B = ['dayofweek', 'id', 'city', 'citytype']
    RFC_ONEHOT_COLS_ALL = ['dayofweek', 'weekofyear', 'hour', 'dayperiod', 'id', 'city', 'citytype']

    # Base output paths for saving models and output from notebooks
    MODELS_RELATIVE_PATH = '../output/models/'
    OUTPUT_RELATIVE_PATH = '../output/'
    CLEANDATA_RELATIVE_PATH = '../data/cleandata/'

    ############################################################################################
    # Define Masks for selecting subsets of data from the cleaned waittimes dataframe
    ############################################################################################
    MASK_EMERGENCY_SERVICES = lambda df: df['services'] == 'emergency'

    ############################################################################################
    # Static Utility Methods
    ############################################################################################

    @staticmethod
    def get_raw_hospital_info_dataframe():
        """
        Load the raw hospital "info" csv file into a dataframe, which is returned to the calling code.
        This is required in the notebook that cleans and standardizes the data and adds geolocations where missing.

        Parameters:
            None (uses path defined by Utility.RAW_INFO_FILENAME, above)

        Returns:
            Pandas dataframe with the raw hospital info data loaded

        Usage:
            df = Utility.get_raw_hospital_info_dataframe()
        """
        return pd.read_csv(Utility.RAW_INFO_FILENAME)


    @staticmethod
    def get_hospital_info_dataframe():
        """
        Load the hospital "info" csv file into a dataframe, which is returned to the calling code.

        Parameters:
            None (uses path defined by Utility.CLEAN_INFO_FILENAME, above)

        Returns:
            Pandas dataframe with the cleaned hospital info data loaded

        Usage:
            df = Utility.get_hospital_info_dataframe()
        """
        return pd.read_csv(Utility.CLEAN_INFO_FILENAME)
    
    
    @staticmethod
    def get_raw_waittimes_dataframe():
        """
        Load the raw wait time csv file (from the Google Sheet provided by data source) into a dataframe, which is returned to the calling code.
        This dataframe needs to undergo data cleaning and pd.melt() in order to be usable by the machine learning models.

        Parameters:
            None (uses path defined by Utility.RAW_WAIT_FILENAME, above)

        Returns:
            Pandas dataframe with the raw waittime data loaded

        Usage:
            df = Utility.get_raw_waittimes_dataframe()
        """
        return pd.read_csv(Utility.RAW_WAIT_FILENAME)
    

    @staticmethod
    def get_clean_waittimes_dataframe():
        """
        Load the cleaned csv file into a dataframe, which is returned to the calling code.

        Parameters:
            None (uses path defined by Utility.CLEAN_WAIT_FILENAME, above)

        Returns:
            Pandas dataframe with the clean waittime data loaded

        Usage:
            df = Utility.get_clean_waittimes_dataframe()
        """
        return pd.read_csv(Utility.CLEAN_WAIT_FILENAME)
    
    
    @staticmethod
    def setup_display_max_cols_rows():
        """
        Sets the Pandas option to have no restrictions on number of columns across, nor number of rows down, when displaying a dataframe.
        Useful when we do not want Pandas to hide display of the middle rows of a dataframe with ellipses,
        nor to truncate the right columns of a dataframe with numerous columsn that scroll to the right off the screen so they cannot be seen.

        Parameters:
            None

        Usage:
            df = Utility.setup_display_max_cols_rows()
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)


    @staticmethod
    def reset_display_max_cols_rows():
        """
        Resets the Pandas option to have default restrictions on number of columns across and number of rows down, when displaying a dataframe.
        
        Parameters:
            None

        Usage:
            df = Utility.reset_display_max_cols_rows()
        """
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')


    @staticmethod
    def get_geocode(address):
        """
        Calls the Geocode Endpoint (defined in Utility.GEOCODE_ENDPOINT constant at top of Utility class) with the passed-in address, and returns
        the latitude and longitude provided by the API, as a tuple.
        
        Parameters:
            address: (string):  The address of a place for which you want the latitude and longitude from the Geocode API

        Returns:
            (tuple):  (latitude,longitude) of the address as geocoded by the Geocode API

        Usage:
            (lat,long) = Utility.get_geocode('101 9 Ave SW, Calgary, AB T2P 1J9')
        """
        # Constants related to the endpoint and API
        geocode_endpoint = Utility.GEOCODE_ENDPOINT
        searchapi = 'search?q='
        header_dict = {
            'accept': 'application/json'
            }

        request_url = geocode_endpoint + searchapi + address
        #print(f"Inside get_geocode: request_url = {request_url}, header_dict = {header_dict}")  # DEBUG: Uncomment if required

        print(f"     Calling API: request_url = {request_url}")

        # Make the call, get response out
        #response = requests.get(request_url, headers=header_dict)
        response = requests.get(request_url)

        # Pass a tuple of (lat,lon) back to the caller
        #print(response.json())  # DEBUG: Uncomment if required
        if len(response.json()) > 0:
            lat = response.json()[0].get('lat',None)
            long = response.json()[0].get('lon',None)
            return (lat,long)
        else:
            print(f"          *** Inside get_geocode(): payload is empty!  Please check out address: {address}!")
            return (0,0)


    @staticmethod
    def make_datetime_column_on_raw_waittime_df(df):
        """
        Adds a column to the passed in dataframe, df, called 'datetime'.
        This is part of the raw waittime data cleaning/preparation workflow which manufactures a pd.datetime object made from
        the 'Date', 'Year', and 'Time' columns that are part of the raw waittimes dataset.
        Parameters:
            df: (Pandas Dataframe): The dataframe on which to add the 'datetime' column.
        Usage:
            Utility.make_datetime_column_on_raw_waittime_df(my_already_existing_dataframe)
        """
        df.insert(df.columns.get_loc('Time') + 1, 'datetime', pd.to_datetime(df['Year'].astype(str) + ' ' + df['Date'] + ' ' + df['Time'], format='%Y %d-%b %I:%M %p'))


    @staticmethod
    def add_longwait_feature(df):
        """
        Adds a column to the passed in dataframe, df, called 'longwait'.
        This is a feature engineering method that sets a '0' (int) if the corresponding 'waittime' column is less than than Utility.LONGWAIT_MINUTES, else sets a '1' (int)
        
        Parameters:
            df: (Pandas Dataframe): The dataframe on which to add the 'longwait' feature column.

        Usage:
            Utility.add_longwait_feature(my_already_existing_dataframe)
        """
        df['longwait'] = df['waittime'].apply(lambda x: 1 if x > Utility.LONGWAIT_MINUTES else 0)

    
    @staticmethod
    def add_dayperiod_feature(df):
        """
        Adds a column to the passed in dataframe, df, called 'dayperiod'.
        This is a feature engineering method that sets categorical value 'night', 'day' or 'evening' based on the 'hour' of the observation,
        per Utility.DAYPERIOD_BINS_LABELS defined above.
        
        Parameters:
            df: (Pandas Dataframe): The dataframe on which to add the 'dayperiod' feature column.

        Usage:
            Utility.add_dayperiod_feature(my_already_existing_dataframe)
        """
        # For DayPeriod, set up bins and labels
        labels = Utility.DAYPERIOD_BINS_LABELS['labels']
        bins = Utility.DAYPERIOD_BINS_LABELS['bins']
        df['dayperiod'] = pd.cut(df['hour'], bins=bins, labels=labels, include_lowest=True)


    @staticmethod
    def collapse_periurban_to_rural(df):
        """
        Replaces all cells in the 'citytype' column of the passed in dataframe, df, which is currently 'peri', with 'rural'.
        Effectively collapses rows associated to cities that are defined as 'peri-urban' to 'rural' for the purposes of modelling.
        
        Parameters:
            df: (Pandas Dataframe): The dataframe on which to have 'peri' citytypes changed to 'rural'.

        Usage:
            Utility.collapse_periurban_to_rural(my_already_existing_dataframe)
        """
        df['citytype'].replace('peri', 'rural', inplace=True)

    @staticmethod
    def drop_unwanted_columns(df, columns_to_drop):
        """
        Drops columns defined in passed in parameter 'columns_to_drop', from in passed in parameter df, if each of the columns exists in df.
        Is meant to be called with a list of columns defined in this Utility class, such as Utility.LOGIT_DROP_COLS
        
        Parameters:
            df:  (Pandas dataframe): The dataframe on which to drop columns.
            columns_to_drop:  (list):  The list of column names to drop from the dataframe.

        Usage:
            Utility.drop_unwanted_columns(my_already_existing_dataframe, list_of_columns_to_drop)
        """
        # Meant to be called with Utility.LOGIT_DROP_COLS from Notebook files
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)


    @staticmethod
    def sort_df_before_encoding_or_modeling(df):
        """
        Sorts the dataframe by the following columns, in this order:
        dayofweek, weekofyear, hour, dayperiod, id, city, citytype

        This ensures that any one-hot encoding done on the dataframe will be done in the same order and the same columns will be dropped on categorical variables.

        Parameters:
            df:  (Pandas dataframe): The dataframe on which to sort in a particular order.

        Returns:
            Pandas dataframe after sorting

        Usage:
            Utility.sort_df_before_encoding_or_modeling(my_already_existing_dataframe)
        """
        df_sorted = df.sort_values(['dayofweek', 'weekofyear', 'hour', 'dayperiod', 'id', 'city', 'citytype']).reset_index(drop=True)
        return df_sorted
    
    
    @staticmethod
    def one_hot_encode_categorical_columns(df, columns_to_onehot_encode, drop_first=True):
        """
        One-Hot encodes columns defined in passed-in parameter 'columns_to_onehot_encode', on the passed-in dataframe 'df'.
        Removes the first one-hot encoded column to prevent multicollinearity when training model using 'df'.
        Is meant to be called with a list of columns defined in this Utility class, such as Utility.LOGIT_ONEHOT_COLS_A or Utility.LOGIT_ONEHOT_COLS_B.
        
        Parameters:
            df:  (Pandas dataframe): The dataframe on which to one-hot encode columns.
            columns_to_onehot_encode:  (list):  The list of column names to one-hot encode.

        Returns:
            Pandas dataframe after the one-hot encoding

        Usage:
            Utility.one_hot_encode_categorical_columns(my_already_existing_dataframe, list_of_columns_to_onehot_encode)
        """
        # Meant to be called with Utility.LOGIT_ONEHOT_COLS_A or Utility.LOGIT_ONEHOT_COLS_B, from Notebook files
        df_dummied = pd.get_dummies(df, columns=columns_to_onehot_encode, drop_first=drop_first)
        return df_dummied
    
    
    @staticmethod
    def convert_booleans(df):
        """
        Converts all columns of dtype boolean, from True to 0 (int) or False to 1 (int).
        
        Parameters:
            None

        Usage:
            Utility.convert_booleans(my_already_existing_dataframe)
        """
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        df[bool_cols] = df[bool_cols].astype(int)


    @staticmethod
    def display_df_general_info(df):
        """
        Displays certain standard information about the passed-in dataframe, df.
        Displays:
         - df.head()
         - df.shape (along with a print statment preceeding this print)
         - df.columns (along with a print statment preceeding this print)
         - df.dtypes (along with a print statment preceeding this print)
        
        Parameters:
            None

        Usage:
            Utility.display_df_general_info(my_already_existing_dataframe)
        """
        display(df.head())
        print(f"\nShape of dataframe:\n{df.shape}")
        print(f"\nColumns of dataframe:\n{df.columns}")
        print(f"\ndtypes of dataframe:\n{df.dtypes}")


    @staticmethod
    def display_class_balance(y_train, y_test):
        """
        Invokes a Counter object on the passed-in dataframe Pandas Series y_train, y_test to obtain a count of total number of items
        and the class counts in percentage.
        Displays:
         - Total number of items in each of y_train, y_test
         - Percentage of 0's in each of y_train, y_test
         - Percentage of 1's in each of y_train, y_test
        
        Parameters:
            y_train: (pd.Series): Series of target variable for y_train (binary, as 0 or 1)
            y_test: (pd.Series): Series of target variable for y_test (binary, as 0 or 1)

        Usage:
            Utility.display_class_balance(y_train_series, y_test_series)
        """
        y_train_counter = Counter(y_train)
        y_train_total = y_train_counter[0] + y_train_counter[1]
        print (f"y_train_counter: {y_train_counter}, 0: {(y_train_counter[0]/y_train_total)*100}, 1: {(y_train_counter[1]/y_train_total)*100}")
        
        y_test_counter = Counter(y_test)
        y_test_total = y_test_counter[0] + y_test_counter[1]
        print (f"y_test_counter: {y_test_counter}, 0: {(y_test_counter[0]/y_test_total)*100}, 1: {(y_test_counter[1]/y_test_total)*100}")


    @staticmethod
    def print_num_stats(df):
        """
        Generates a nicely formatted, tab-aligned print out showing:
        - Column Name, Percentage of Null Values in that Column, Number of Null values in that column, and Total Number of Rows in the column
        
        Parameters:
            df:  (Pandas dataframe): The dataframe on which to generate null value statistics.

        Usage:
            Utility.print_num_stats(my_already_existing_dataframe)
        """
        print("Stats for Number of Nulls:\n")
        print("{:<40}: {:>20} {:>13} {:>13}".format('Column Name', 'Null Percentage', 'Num Nulls', 'Num Rows'))
        print("{:<40}: {:>20} {:>13} {:>13}".format('-----------', '---------------', '---------', '--------'))
        for col in df.columns:
            num_nulls = df[col].isnull().sum()
            num_rows = df[col].shape[0]
            null_percentage = (num_nulls / num_rows) * 100
            print("{:<40}: {:>20.2f}% {:>12} {:>12}".format(col, null_percentage, num_nulls, num_rows))

    
    @staticmethod
    def plot_roc_curve(fpr, tpr, plot_baseline=True):
        """
        Plots a ROC curve given the false positive rate (fpr)
        and true positive rate (tpr) of a model.

        Credit: Daniel Bourke
        https://github.com/mrdbourke/zero-to-mastery-ml/tree/master

        Parameters:
            fpr:  (Numpy 1D array): The numpy array with y_true values
            tpr:  (Numpy 1D array):  The probabilities of being classified as '1' (true), typically `model.predict_proba(X_test)[:,1]`
            plot_baseline:  (boolean, defaulted to True):  Whether or not to plot the "Guessing" diagonal line across the ROC Curve

        Usage:
            Utility.plot_roc_curve(fpr, tpr)
        """
        # Plot roc curve
        plt.plot(fpr, tpr, color="orange", label="ROC")
        
        # Plot line with no predictive power (baseline)
        if plot_baseline == True:
            plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")

        # Customize the plot
        plt.xlabel("False positive rate (fpr)")
        plt.ylabel("True positive rate (tpr)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.show()


    @staticmethod
    def evaluate_preds(y_true, y_preds):
        """
        Performs evaluation comparison on y_true labels vs. y_pred labels
        on a classification.

        Credit: Daniel Bourke (f1 macro and weighted scores and formatting added by Julie Leung)
        https://github.com/mrdbourke/zero-to-mastery-ml/tree/master

        Parameters:
            y_true:  (Numpy 1D array): The numpy array with y_true values
            y_preds:  (Numpy 1D array):  The numpy array with y_predicted values, which came from `model.predict(X_test)` as an example

        Returns:
            Dictionary with metric labels as keys, and associated metric scores as values

        Usage:
            Utility.evaluate_preds(y_test, y_pred)
        """
        accuracy = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds)
        recall = recall_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds)
        f1_macro = f1_score(y_true, y_preds, average='macro')
        f1_weighted = f1_score(y_true, y_preds, average='weighted')
        roc_auc = roc_auc_score(y_true, y_preds)
        metric_dict = {"accuracy": round(accuracy, 2),
                       "precision": round(precision, 2),
                       "recall": round(recall, 2),
                       "f1": round(f1, 2),
                       "f1_macro": round(f1_macro, 2),
                       "f1_weighted": round(f1_weighted, 2),
                       "roc_auc": round(roc_auc, 2)}

        # print(f"Accuracy: {accuracy * 100:.2f}%")
        # print(f"Precision: {precision:.2f}")
        # print(f"Recall: {recall:.2f}")
        # print(f"F1 score: {f1:.2f}")
        # print(f"F1 (macro) score: {f1_macro:.2f}")
        # print(f"F1 (weighted) score: {f1_weighted:.2f}")

        print("{:<20}: {:>10.2f}%".format("Accuracy", accuracy * 100))
        print("{:<20}: {:>10.2f}".format("Precision", precision))
        print("{:<20}: {:>10.2f}".format("Recall", recall))
        print("{:<20}: {:>10.2f}".format("F1 score", f1))
        print("{:<20}: {:>10.2f}".format("F1 (macro) score", f1_macro))
        print("{:<20}: {:>10.2f}".format("F1 (weighted) score", f1_weighted))
        print("{:<20}: {:>10.2f}".format("ROC-AUC", roc_auc))

        return metric_dict
    
    
    @staticmethod
    def print_confusion_matrix_with_labels(y_true, y_preds):
        """
        Generates, labels and prints out a confusion matrix using sklearn's confusion_matrix() function, forcing labels = [1,0]
        so the labels on the dataframe will be in the correct order.

        Parameters:
            y_true:  (Numpy 1D array): The numpy array with y_true values
            y_preds:  (Numpy 1D array):  The numpy array with y_predicted values, which came from `model.predict(X_test)` as an example

        Returns:
            A dataframe of the confusion matrix with correct labels for each cell.

        Usage:
            scores_dictionary = Utility.print_confusion_matrix_with_labels(y_true, y_pred)
        """
        cm = confusion_matrix(y_true, y_preds, labels=[1,0])
        cm_df = pd.DataFrame(
            cm,
            index=['Actual Positive', 'Actual Negative'],
            columns=['Predicted Positive', 'Predicted Negative'])
        
        display(cm_df)
        return cm_df
    

    @staticmethod
    def get_feature_importances_and_plot(model, X_train, modelname='Model', nlargest=10, bartype='barh', figsize=(5,5)):
        """
        Generates sorted featured importances pd.Series, and plots the top 'nlargest' feature importances in a horizontal bar chart.

        Parameters:
            model:  (Base Estimator): Model that has been fit on X_train, which holds .feature_importances_ attribute
            X_train:  (Numpy 2D array):  The numpy array with X_train values on which the model was fit,
             which came from train_test_split(), as an example
            modelname:  (string, defaulted to 'Model'):  The name of the model, used in the title of the plot
            nlargest:  (int, defaulted to 10):  The number of top feature importances to plot
            bartype:  (string, defaulted to 'barh'):  The type of bar chart to plot, 'barh' for horizontal, 'bar' for vertical

        Returns:
            A pd.Series of the feature importances, sorted from highest to lowest

        Usage:
            feat_importances = Utility.get_feature_importances_and_plot(mymodel, X_train, 'RFC using One-Hot Encoding')
        """
        # Create the Feature Importances series from the model's feature_importances_ and the column names on X_train
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

        # Sort from highest to lowest
        feat_importances.sort_values(ascending=False, inplace=True)

        # Plot the top 'nlargest' feature importances
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.title(f'Top {nlargest} Feature Importances on {modelname} Model')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Feature')
        viridis_cmap = colormaps.get_cmap('viridis')
        num_features = nlargest
        colors = [viridis_cmap(i / num_features) for i in range(num_features)]
        feat_importances.nlargest(nlargest)[::-1].plot(kind=bartype, figsize=figsize, color=colors);
        return feat_importances
    