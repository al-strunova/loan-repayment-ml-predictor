from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Description:
    -------
    Clean up data, create new features for every dataset and merge everything into one dataset.

    Parameters:
    - client_profile_data: pandas.core.frame.DataFrame, client profile data
    - history_data: pandas.core.frame.DataFrame, applications history data
    - bki_data: pandas.core.frame.DataFrame, BKI data
    - payments_data: pandas.core.frame.DataFrame, payments data
    """

    def __init__(self,
                 client_profile_data='../data/client_profile.csv',
                 history_data='../data/applications_history.csv',
                 bki_data='../data/bki.csv',
                 payments_data='../data/payments.csv',
                 percentile_lower=5,
                 percentile_upper=95):

        # Input data files
        self.client_profile_data = pd.read_csv(client_profile_data)
        self.history_data = pd.read_csv(history_data)
        self.bki_data = pd.read_csv(bki_data)
        self.payments_data = pd.read_csv(payments_data)

        # Percentile values for outlier handling
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper

        # Bounds for outlier handling
        self.annuity_right_bond = None
        self.annuity_left_bond = None
        self.credit_right_bond = None
        self.salary_right_bond = None
        self.credit_left_bond = None
        self.salary_left_bond = None

        # Miscellaneous attributes
        self.test_ids = None

        # Statistics attributes for various features
        self.stats_fam_status = None
        self.stats_region = None
        self.stats_gender = None
        self.stats_education = None

    def fit(self, X):
        """
        Description:
        -------
        Fit the transformer to the data and calculate various statistics.

        Parameters:
        - X: pandas.core.frame.DataFrame, input data
        - y: None, not used

        Returns:
        - self
        """
        # Merging data to create statistics
        merged_data = self.merge_tables_on_app_num(X, self.client_profile_data)

        # Generate statistics from merged_data
        self.set_groupby_profile_stats(merged_data)
        self.set_percentile_bonds(merged_data)

        # Clearing the temporary merged data
        del merged_data

        return self

    def transform(self, X):
        """
        Description:
        -------
        Transform the input data by creating new features and merging datasets.

        Parameters:
        - X: pandas.core.frame.DataFrame, input data
        - y: None, not used

        Returns:
        - X_transformed: pandas.core.frame.DataFrame, transformed data
        """

        # Create features based on the client's profile data
        X = self.create_client_profile_features(X)

        # Create features based on the applications' history data
        history = self.create_applications_history_features(self.history_data)

        # Create features based on the BKI data
        bki = self.create_bki_features(self.bki_data)

        # Create features based on the payments data
        payments = self.create_payments_features(self.payments_data)

        # Merge all the created datasets/tables into the main dataset X
        for table in [history, bki, payments]:
            X = self.merge_tables_on_app_num(X, table)

        # Store application numbers for any potential future use
        # (e.g., when making predictions on test data)
        self.test_ids = list(X["APPLICATION_NUMBER"])

        # Drop the APPLICATION_NUMBER column as it's no longer needed in the training data
        X.drop(columns=["APPLICATION_NUMBER"], inplace=True)

        return X

    def merge_tables_on_app_num(self, left_df, right_df):
        """
        Description:
        -------
            Merge any two tables based on APPLICATION_NUMBER column

        Parameters
        ----------
            left_df: pandas.core.frame.DataFrame
            right_df: pandas.core.frame.DataFrame

        Returns
        -------
            merged_df: pandas.core.frame.DataFrame

        """

        # Merge two dataframes on the "APPLICATION_NUMBER" column. Missing values from right_df
        # are filled in left_df with NaN.
        df = pd.merge(left_df, right_df, how="left", on="APPLICATION_NUMBER")

        # Replace positive infinity values with NaN
        df = df.replace(np.inf, np.nan)

        # Replace negative infinity values with NaN
        df = df.replace(-np.inf, np.nan)

        return df

    def get_agg_function(self, function_name: str):
        agg_functions = {
            'mean': np.nanmean,
            'median': np.nanmedian,
            'sum': np.nansum,
            'var': np.nanvar,
            'min': np.min,
            'max': np.max,
        }

        return agg_functions.get(function_name, function_name)

    def set_groupby_profile_stats(self, X):
        """
        Description:
        -------
            Calculate group statists such as TOTAL_SALARY and AMOUNT_CREDIT mean for
            EDUCATION_LEVEL, GENDER, FAMILY_STATUS and REGION_POPULATION on the train set

        Parameters
        ----------
            X: pandas.core.frame.DataFrame
        """

        # Aggregate specific columns (TOTAL_SALARY, AMOUNT_CREDIT) based on different categorical features
        # (EDUCATION_LEVEL, GENDER, REGION_POPULATION, FAMILY_STATUS). The aggregated statistics, which are
        # mean in this case, are computed for each group and stored for later use.
        # These aggregated statistics can be useful for creating group-specific features in the future.
        aggs = {"TOTAL_SALARY": ["mean"], "AMOUNT_CREDIT": ["mean"]}
        self.stats_education = self.create_numerical_aggs(X, groupby_id="EDUCATION_LEVEL", aggs=aggs,
                                                          prefix="GROUPBY_EDU_LEVEL_")
        self.stats_gender = self.create_numerical_aggs(X, groupby_id="GENDER", aggs=aggs, prefix="GROUPBY_GENDER_")
        self.stats_region = self.create_numerical_aggs(X, groupby_id="REGION_POPULATION", aggs=aggs,
                                                       prefix="GROUPBY_REGION_")
        self.stats_fam_status = self.create_numerical_aggs(X, groupby_id="FAMILY_STATUS", aggs=aggs,
                                                           prefix="GROUPBY_FAM_STATUS_")

    def compute_percentiles(self, data, column_name):
        """
        Compute lower and upper percentiles for a given column.

        Args:
        - data: pandas DataFrame, the input data.
        - column_name: str, the column for which percentiles need to be computed.

        Returns:
        - lower_bound: float, the lower percentile value.
        - upper_bound: float, the upper percentile value.
        """
        lower_bound = np.nanpercentile(data[column_name], q=self.percentile_lower)
        upper_bound = np.nanpercentile(data[column_name], q=self.percentile_upper)

        return lower_bound, upper_bound

    def set_percentile_bonds(self, X):
        """
        Description:
        -------
            Calculate left and right percentile bonds for
            TOTAL_SALARY, AMOUNT_CREDIT and AMOUNT_ANNUITY on the train set

        Parameters
        ----------
            X: pandas.core.frame.DataFrame
        """

        # Applying the utility function to compute percentiles
        self.salary_left_bond, self.salary_right_bond = self.compute_percentiles(X, 'TOTAL_SALARY')
        self.credit_left_bond, self.credit_right_bond = self.compute_percentiles(X, 'AMOUNT_CREDIT')
        self.annuity_left_bond, self.annuity_right_bond = self.compute_percentiles(X, 'AMOUNT_ANNUITY')

    def create_numerical_aggs(self, data: pd.DataFrame, groupby_id: str, aggs: dict, prefix: Optional[str] = None,
                              suffix: Optional[str] = None) -> pd.DataFrame:
        """
        Description:
        -------
            Create aggregations for numeric features

        Parameters
        ----------
            data: pandas.core.frame.DataFrame
            groupby_id: str
                The column by which to group the data
            aggs: dict
                Dictionary with feature's name and the list of aggr functions to perform
            prefix: str, optional, default = None
                Prefix which will be used to name a new created feature
            suffix: str, optional, default = None
                Suffix which will be used to name a new created feature

        Returns
        -------
            stats: pandas.core.frame.DataFrame

        """
        # Error handling
        if groupby_id not in data.columns:
            raise ValueError(f"'{groupby_id}' not found in dataframe columns")

        # Convert string keys in the aggs dictionary to actual functions
        # for feature, functions in aggs.items():
        # aggs[feature] = [self.get_agg_function(func) for func in functions]

        # Ensure prefix and suffix have valid default values
        prefix = prefix or ""
        suffix = suffix or ""

        # Group data by the specified column and compute the specified aggregates
        data_grouped = data.groupby(groupby_id)
        stats = data_grouped.agg(aggs)

        # Rename the columns of the aggregated data using the provided prefix and suffix
        stats.columns = [f"{prefix}{feature}_{stat}{suffix}".upper() for feature, stat in stats]

        # Reset index for the resulting dataframe
        stats = stats.reset_index()

        return stats

    def create_client_profile_features(self, X):
        """
        Description:
        -------
            Create new features for client_profile dataset

        Parameters
        ----------
            X: pandas.core.frame.DataFrame

        Returns
        -------
            X_transformed: pandas.core.frame.DataFrame
        """

        # Create a deep copy of the input DataFrame to avoid modifying the original
        X = X.copy()

        # Merges two tables based on application numbers
        X = self.merge_tables_on_app_num(X, self.client_profile_data)

        # FLAGGING MISSING VALUES AND OUTLIERS

        # Mark specific columns with missing values by creating new columns with the prefix 'MISSING_'
        # These features have a lot of missing values
        flag_missing_columns = ['OWN_CAR_AGE', 'EXTERNAL_SCORING_RATING_1', 'EXTERNAL_SCORING_RATING_3',
                                'AMT_REQ_CREDIT_BUREAU_MON']
        X[[f'MISSING_{col}' for col in flag_missing_columns]] = X[flag_missing_columns].isna().astype(int)

        # Flag DAYS_ON_LAST_JOB values higher than 350000 as missing
        X['MISSING_DAYS_ON_LAST_JOB'] = (X.DAYS_ON_LAST_JOB > 350000).astype('int')

        # Correct 'MISSING_OWN_CAR_AGE' value if flagged incorrectly
        X.loc[X['MISSING_OWN_CAR_AGE'] == 1, 'MISSING_OWN_CAR_AGE'] = 0

        # FLAG OUTLIERS IN CERTAIN COLUMNS

        # Flag outliers for 'TOTAL_SALARY', 'AMOUNT_CREDIT', 'AMOUNT_ANNUITY'
        X['OUTLIER_TOTAL_SALARY'] = (
                (X['TOTAL_SALARY'] < self.salary_left_bond) | (X['TOTAL_SALARY'] > self.salary_right_bond)).astype(
            'int')
        X['OUTLIER_AMOUNT_CREDIT'] = ((X['AMOUNT_CREDIT'] < self.credit_left_bond) | (
                X['AMOUNT_CREDIT'] > self.credit_right_bond)).astype(int)
        X['OUTLIER_AMOUNT_ANNUITY'] = ((X['AMOUNT_ANNUITY'] < self.annuity_left_bond) | (
                X['AMOUNT_ANNUITY'] > self.annuity_right_bond)).astype('int')

        # PROCESS NUMERIC FEATURES

        # Convert 'CHILDRENS' column into a categorical format
        X['CHILDREN_0'] = (X.CHILDRENS == 0).astype('int')
        X['CHILDREN_1_2'] = ((X['CHILDRENS'] >= 1) & (X['CHILDRENS'] <= 2)).astype('int')
        X['CHILDREN_3+'] = (X.CHILDRENS >= 3).astype('int')

        # Convert 'FAMILY_SIZE' column into a categorical format
        X['FAMILY_SIZE_0'] = (X.FAMILY_SIZE == 0).astype('int')
        X['FAMILY_SIZE_1'] = (X.FAMILY_SIZE == 1).astype('int')
        X['FAMILY_SIZE_2'] = (X.FAMILY_SIZE == 2).astype('int')
        X['FAMILY_SIZE_3+'] = (X.FAMILY_SIZE >= 3).astype('int')

        # Generate new EDUCATION_LEVEL metrics
        X = X.merge(self.stats_education, how="left", on="EDUCATION_LEVEL")
        X["RATIO_CREDIT_to_MEAN_CREDIT_BY_EDUCATION"] = X["AMOUNT_CREDIT"] / X["GROUPBY_EDU_LEVEL_AMOUNT_CREDIT_MEAN"]
        X["RATIO_SALARY_to_MEAN_SALARY_BY_EDUCATION"] = X["TOTAL_SALARY"] / X["GROUPBY_EDU_LEVEL_TOTAL_SALARY_MEAN"]
        X["DIFF_SALARY_and_MEAN_SALARY_BY_EDUCATION"] = X["TOTAL_SALARY"] - X["GROUPBY_EDU_LEVEL_TOTAL_SALARY_MEAN"]

        # Generate new GENDER metrics
        X = X.merge(self.stats_gender, how="left", on="GENDER")
        X["RATIO_CREDIT_to_MEAN_CREDIT_BY_GENDER"] = X["AMOUNT_CREDIT"] / X["GROUPBY_GENDER_AMOUNT_CREDIT_MEAN"]
        X["RATIO_SALARY_to_MEAN_SALARY_BY_GENDER"] = X["TOTAL_SALARY"] / X["GROUPBY_GENDER_TOTAL_SALARY_MEAN"]
        X["DIFF_SALARY_and_MEAN_SALARY_BY_GENDER"] = X["TOTAL_SALARY"] - X["GROUPBY_GENDER_TOTAL_SALARY_MEAN"]

        # Generate new REGION_POPULATION metrics
        X = X.merge(self.stats_region, how="left", on="REGION_POPULATION")
        X["RATIO_CREDIT_to_MEAN_CREDIT_BY_REGION"] = X["AMOUNT_CREDIT"] / X["GROUPBY_REGION_AMOUNT_CREDIT_MEAN"]
        X["RATIO_SALARY_to_MEAN_SALARY_BY_REGION"] = X["TOTAL_SALARY"] / X["GROUPBY_REGION_TOTAL_SALARY_MEAN"]
        X["DIFF_SALARY_and_MEAN_SALARY_BY_REGION"] = X["TOTAL_SALARY"] - X["GROUPBY_REGION_TOTAL_SALARY_MEAN"]

        # Generate new FAMILY_STATUS metrics
        X = X.merge(self.stats_fam_status, how="left", on="FAMILY_STATUS")
        X["RATIO_CREDIT_to_MEAN_CREDIT_BY_FAM_STATUS"] = X["AMOUNT_CREDIT"] / X["GROUPBY_FAM_STATUS_AMOUNT_CREDIT_MEAN"]
        X["RATIO_SALARY_to_MEAN_SALARY_BY_FAM_STATUS"] = X["TOTAL_SALARY"] / X["GROUPBY_FAM_STATUS_TOTAL_SALARY_MEAN"]
        X["DIFF_SALARY_and_MEAN_SALARY_BY_FAM_STATUS"] = X["TOTAL_SALARY"] - X["GROUPBY_FAM_STATUS_TOTAL_SALARY_MEAN"]

        # Generate financial metrics
        X['RATIO_CREDIT_to_ANNUITY'] = X['AMOUNT_CREDIT'] / X['AMOUNT_ANNUITY']
        X['RATIO_CREDIT_to_SALARY'] = X['AMOUNT_CREDIT'] / X['TOTAL_SALARY']
        X['RATIO_SALARY_TO_CREDIT'] = X['TOTAL_SALARY'] / X['AMOUNT_CREDIT']
        X['RATIO_ANNUITY_to_SALARY'] = X['AMOUNT_ANNUITY'] / X['TOTAL_SALARY']
        X['DIFF_SALARY_and_ANNUITY'] = X['TOTAL_SALARY'] - X['AMOUNT_ANNUITY']
        X["FLG_MORE_THAN_50PERCENT_FOR_CREDIT"] = np.where(X["RATIO_ANNUITY_to_SALARY"] > 0.5, 1, 0)
        X["FLG_MORE_THAN_30PERCENT_FOR_CREDIT"] = np.where(X["RATIO_ANNUITY_to_SALARY"] > 0.3, 1, 0)
        X["FLG_PHONE_and_EMAIL"] = np.where((X["FLAG_PHONE"] == 1) & (X["FLAG_EMAIL"] == 1), 1, 0)

        # Generate scoring metrics
        for function_name in ['mean', 'median', 'min', 'max', 'var']:
            feature_name = "EXTERNAL_SCORING_{}".format(function_name.upper())
            function = self.get_agg_function(function_name)
            X[feature_name] = X[
                ["EXTERNAL_SCORING_RATING_1", "EXTERNAL_SCORING_RATING_2", "EXTERNAL_SCORING_RATING_3"]
            ].apply(lambda row: function(row) if not row.isna().all() else np.nan, axis=1)

        X["EXTERNAL_SCORING_PROD"] = X["EXTERNAL_SCORING_RATING_1"] * X["EXTERNAL_SCORING_RATING_2"] * X[
            "EXTERNAL_SCORING_RATING_3"]
        X["EXTERNAL_SCORING_WEIGHTED"] = X["EXTERNAL_SCORING_RATING_1"] * 2 + X["EXTERNAL_SCORING_RATING_2"] * 1 + X[
            "EXTERNAL_SCORING_RATING_3"] * 3
        X["EXPECTED_TOTAL_LOSS_1"] = X["EXTERNAL_SCORING_RATING_1"] * X["AMOUNT_CREDIT"]
        X["EXPECTED_TOTAL_LOSS_2"] = X["EXTERNAL_SCORING_RATING_2"] * X["AMOUNT_CREDIT"]
        X["EXPECTED_TOTAL_LOSS_3"] = X["EXTERNAL_SCORING_RATING_3"] * X["AMOUNT_CREDIT"]
        X["EXPECTED_MONTHLY_LOSS_1"] = X["EXTERNAL_SCORING_RATING_1"] * X["AMOUNT_ANNUITY"]
        X["EXPECTED_MONTHLY_LOSS_2"] = X["EXTERNAL_SCORING_RATING_2"] * X["AMOUNT_ANNUITY"]
        X["EXPECTED_MONTHLY_LOSS_3"] = X["EXTERNAL_SCORING_RATING_3"] * X["AMOUNT_ANNUITY"]

        # Ratio with Age
        X["RATIO_ANNUITY_to_AGE"] = X["AMOUNT_ANNUITY"] / X["AGE"]
        X["RATIO_CREDIT_to_AGE"] = X["AMOUNT_CREDIT"] / X["AGE"]
        X["RATIO_SALARY_to_AGE"] = X["TOTAL_SALARY"] / X["AGE"]
        X["RATIO_AGE_to_SALARY"] = X["AGE"] / X["TOTAL_SALARY"]

        # Ratio with days_on_last_job
        X["RATIO_ANNUITY_to_DAYS_ON_LAST_JOB"] = X["AMOUNT_ANNUITY"] / X["DAYS_ON_LAST_JOB"]
        X["RATIO_CREDIT_to_DAYS_ON_LAST_JOB"] = X["AMOUNT_CREDIT"] / X["DAYS_ON_LAST_JOB"]
        X["RATIO_SALARY_to_DAYS_ON_LAST_JOB"] = X["TOTAL_SALARY"] / X["DAYS_ON_LAST_JOB"]
        X["RATIO_DAYS_ON_LAST_JOB_to_SALARY"] = X["DAYS_ON_LAST_JOB"] / X["TOTAL_SALARY"]
        X["RATIO_AGE_to_DAYS_ON_LAST_JOB"] = X["AGE"] / X["DAYS_ON_LAST_JOB"]
        X["RATIO_AGE_to_OWN_CAR_AGE"] = X["AGE"] / X["OWN_CAR_AGE"]

        # Ratio with FAMILY_SIZE
        X["RATIO_SALARY_TO_PER_FAMILY_SIZE"] = X["TOTAL_SALARY"] / X["FAMILY_SIZE"]

        # BKI metrics
        bki_flags = [flag for flag in X.columns if "AMT_REQ_CREDIT_BUREAU" in flag]
        X["BKI_REQUESTS_COUNT"] = X[bki_flags].sum(axis=1)
        X["BKI_KURTOSIS"] = X[bki_flags].kurtosis(axis=1)

        # Categorical metrics
        X.GENDER.replace('XNA', 'Missing', inplace=True)
        X.FAMILY_STATUS.replace('Unknown', 'Missing', inplace=True)

        X = X.drop(["CHILDRENS", "FAMILY_SIZE"], axis=1)

        return X

    def create_applications_history_features(self, X):
        """
        Description:
        -------
            Create new features for applications_history_data dataset

        Returns
        -------
            X_transformed: pandas.core.frame.DataFrame
        """

        # Create new features for previously refused applications
        aggs_refused = {
            'PREV_APPLICATION_NUMBER': ['count'],
            'AMT_APPLICATION': ['mean'],
            'DAYS_DECISION': ['mean']
        }
        mask_refused = X["NAME_CONTRACT_STATUS"] == "Refused"
        stats_refused = self.create_numerical_aggs(X[mask_refused], groupby_id="APPLICATION_NUMBER", aggs=aggs_refused,
                                                   prefix="PREV_REFUSED_")

        # Create new features for previously approved applications
        aggs_approved = {
            'PREV_APPLICATION_NUMBER': ['count'],
            'AMOUNT_CREDIT': ['sum', 'mean'],
        }
        mask_approved = X["NAME_CONTRACT_STATUS"] == "Approved"
        stats_approved = self.create_numerical_aggs(X[mask_approved], groupby_id="APPLICATION_NUMBER",
                                                    aggs=aggs_approved, prefix="PREV_APPROVED_")

        res = stats_refused.merge(stats_approved, how='outer', on='APPLICATION_NUMBER')

        return res

    def create_bki_features(self, X):
        """
        Description:
        -------
            Create new features for bki dataset

        Returns
        -------
            X_transformed: pandas.core.frame.DataFrame
        """

        # Create new features for active applications
        aggs_active = {
            'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean']
        }
        mask_active = X['CREDIT_ACTIVE'] == 'Active'
        stats_active = self.create_numerical_aggs(X[mask_active], groupby_id="APPLICATION_NUMBER", aggs=aggs_active,
                                                  prefix="BKI_ACTIVE_")

        # Create new features for closed applications
        aggs_closed = {
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean']
        }
        mask_closed = X['CREDIT_ACTIVE'] == 'Closed'
        stats_closed = self.create_numerical_aggs(X[mask_closed], groupby_id="APPLICATION_NUMBER",
                                                  aggs=aggs_closed, prefix="BKI_CLOSED_")

        res = stats_active.merge(stats_closed, how='outer', on='APPLICATION_NUMBER')

        return res

    def create_payments_features(self, X):
        """
        Description:
        -------
            Create new features for payment dataset

        Returns
        -------
            X_transformed: pandas.core.frame.DataFrame
        """

        X["RATIO_DAYS_PAYMENT_to_DAYS_INSTALMENT"] = X["DAYS_ENTRY_PAYMENT"] / X["DAYS_INSTALMENT"]
        X["RATIO_DAYS_INSTALMENT_to_DAYS_PAYMENT"] = X["DAYS_INSTALMENT"] / X["DAYS_ENTRY_PAYMENT"]
        X["DIFF_DAYS_PAYMENT_and_DAYS_INSTALMENT"] = X["DAYS_ENTRY_PAYMENT"] - X["DAYS_INSTALMENT"]
        X["RATIO_AMT_INSTALMENT_to_AMT_PAYMENT"] = X["AMT_INSTALMENT"] / X["AMT_PAYMENT"]
        X["RATIO_AMT_PAYMENT_to_AMT_INSTALMENT"] = X["AMT_PAYMENT"] / X["AMT_INSTALMENT"]
        X["DIFF_AMT_PAYMENT_and_AMT_INSTALMENT"] = X["AMT_PAYMENT"] - X["AMT_INSTALMENT"]
        X["RATIO_DAYS_PAYMENT_to_AMT_PAYMENT"] = X["DAYS_ENTRY_PAYMENT"] / X["AMT_PAYMENT"]
        X["RATIO_DAYS_INSTALMENT_to_AMT_INSTALMENT"] = X["DAYS_INSTALMENT"] / X["AMT_INSTALMENT"]

        aggs = {
            "AMT_PAYMENT": ["mean"],
            "AMT_INSTALMENT": ["mean"],
            "RATIO_DAYS_PAYMENT_to_DAYS_INSTALMENT": ["mean", "std"],
            "RATIO_DAYS_INSTALMENT_to_DAYS_PAYMENT": ["mean", "std"],
            "DIFF_DAYS_PAYMENT_and_DAYS_INSTALMENT": ["mean"],
            "RATIO_AMT_INSTALMENT_to_AMT_PAYMENT": ["mean", "std"],
            "DIFF_AMT_PAYMENT_and_AMT_INSTALMENT": ["mean"],
            "RATIO_DAYS_PAYMENT_to_AMT_PAYMENT": ["mean"],
            "RATIO_DAYS_INSTALMENT_to_AMT_INSTALMENT": ["mean", "std"],
            "RATIO_AMT_PAYMENT_to_AMT_INSTALMENT": ["mean", "std"]
        }

        res = self.create_numerical_aggs(
            X, groupby_id="APPLICATION_NUMBER", aggs=aggs, prefix="PAYMENT_STAT_"
        )

        return res
