import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

def compas(include_sex = True):
    #Import data and select the appropriate columns
    data = pd.read_csv('Compas_data/features.csv')
    data = data[[i for i in list(data.columns) if i != 'person_id' and i != 'screening_date'
                 and i != 'Risk of Failure to Appear_decile_score' and i != 'Risk of Failure to Appear_raw_score' and i != 'Risk of Recidivism_decile_score'
                 and i != 'Risk of Violence_decile_score' and i != 'Risk of Violence_raw_score' and i != 'Risk of Recidivism_raw_score' and i != 'recid_violent'
                 and i != 'raw_score' and i != 'decile_score' and i != 'filt1' and i != 'filt2' and i != 'filt3' and i != 'filt4'
                 and i != 'filt5' and i != 'filt6' and i != 'p_famviol_arrest']]

    data.loc[:, 'first_offense_date'] = pd.to_datetime(data['first_offense_date'], format='%Y-%m-%d').apply(
        datetime.toordinal)
    data.loc[:, 'current_offense_date'] = pd.to_datetime(data['current_offense_date'], format='%Y-%m-%d').apply(
        datetime.toordinal)
    data.loc[:, 'before_cutoff_date'] = pd.to_datetime(data['before_cutoff_date'], format='%Y-%m-%d').apply(
        datetime.toordinal)

    data.drop_duplicates(keep=False, inplace=True)

    y = data['recid']
    X = data[[i for i in list(data.columns) if i != 'recid']]

    # Split the data into train and test data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if(not include_sex):
        is_male_train = np.asarray(X_train['sex_male'])
        is_male_test = np.asarray(X_test['sex_male'])

        X_train = X_train[[i for i in list(X_train.columns) if i != 'sex_male']]
        X_test = X_test[[i for i in list(X_test.columns) if i != 'sex_male']]
        return X_train, X_test, y_train, y_test, is_male_train, is_male_test

    return X_train, X_test, y_train, y_test

def german_credit(include_sex = True):
    ## Import data
    column_names = ['credit_account_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'saving_account',
                    'employment_since', 'installment_rate', 'personal_status_sex', 'other_debtors',
                    'present_residence_since', 'property', 'age', 'other_installment', 'housing', 'existing_credits',
                    'job', 'liable_people', 'telephone', 'foreign_worker', 'Risk']
    data = pd.read_csv('German_Credit_data/german.data', sep=' ', names=column_names, index_col=False)  # Load the data

    # Make all ordinal categorical columns numerical:
    cat_variables = ['1', '3', '6', '7', '12', '17', '19', '20']
    numbers = ['0', '1', '2', '3', '4', '5']
    for i in cat_variables:
        for num, j in enumerate(numbers):
            label = "A" + i + j
            data = data.replace(label, num)

    # One-hot-encode the non-ordinal categorical columns:
    # Column 4: Purpose
    data = data.replace('A40', 'car_(new)')
    data = data.replace('A41', 'car_(used)')
    data = data.replace('A42', 'furniture/equipment')
    data = data.replace('A43', 'radio/television')
    data = data.replace('A44', 'domestic_appliances')
    data = data.replace('A45', 'repairs')
    data = data.replace('A46', 'education')
    data = data.replace('A47', 'vacation')
    data = data.replace('A48', 'retraining')
    data = data.replace('A49', 'business')
    data = data.replace('A410', 'others')

    data['purpose'] = pd.Categorical(data['purpose'])
    data_dummies = pd.get_dummies(data['purpose'], prefix='purpose')
    data = pd.concat([data, data_dummies], axis=1)
    data = data.drop('purpose', 1)

    # Column 9: Personal status and sex
    data = data.replace('A91', 'm_div/sep')
    data = data.replace('A92', 'f_div/sep/mar')
    data = data.replace('A93', 'm_single')
    data = data.replace('A94', 'm_mar/wid')
    data = data.replace('A95', 'f_single')

    data['personal_status_sex'] = pd.Categorical(data['personal_status_sex'])
    data_dummies = pd.get_dummies(data['personal_status_sex'])
    data = pd.concat([data, data_dummies], axis=1)
    data = data.drop('personal_status_sex', 1)

    # Column 10: Other debtors / guarantors
    data = data.replace('A101', 'none')
    data = data.replace('A102', 'co-applicant')
    data = data.replace('A103', 'guarantor')

    data['other_debtors'] = pd.Categorical(data['other_debtors'])
    data_dummies = pd.get_dummies(data['other_debtors'], prefix='debtor')
    data = pd.concat([data, data_dummies], axis=1)
    data = data.drop('other_debtors', 1)

    # Column 14: Other installment plans
    data = data.replace('A141', 'bank')
    data = data.replace('A142', 'stores')
    data = data.replace('A143', 'none')

    data['other_installment'] = pd.Categorical(data['other_installment'])
    data_dummies = pd.get_dummies(data['other_installment'], prefix='inst_plan')
    data = pd.concat([data, data_dummies], axis=1)
    data = data.drop('other_installment', 1)

    # Column 15: Housing
    data = data.replace('A151', 'rent')
    data = data.replace('A152', 'own')
    data = data.replace('A153', 'for_free')

    data['housing'] = pd.Categorical(data['housing'])
    data_dummies = pd.get_dummies(data['housing'], prefix='housing')
    data = pd.concat([data, data_dummies], axis=1)
    data = data.drop('housing', 1)

    # Change the values of column 19: Telephone
    data['telephone'] = data['telephone'].replace(1,0)
    data['telephone'] = data['telephone'].replace(2, 1)

    # Split the data into train and test data:
    y = data['Risk']
    X = data[[i for i in list(data.columns) if i != 'Risk']]

    # Change: good = '0', bad = '1'
    y = y.replace(1, 0)
    y = y.replace(2, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if not include_sex:
        is_male_train = np.asarray(X_train['f_div/sep/mar'])
        is_male_test = np.asarray(X_test['f_div/sep/mar'])

        is_male_train = (is_male_train == 0).astype(np.float64)
        is_male_test = (is_male_test == 0).astype(np.float64)

        X_train = X_train[[i for i in list(X_train.columns) if i != 'f_div/sep/mar' and i != 'm_div/sep' and i != 'm_single'
                           and i != 'f_single' and i != 'm_mar/wid']]
        X_test = X_test[[i for i in list(X_test.columns) if i != 'f_div/sep/mar' and i != 'm_div/sep' and i != 'm_single'
                           and i != 'f_single' and i != 'm_mar/wid']]
        return X_train, X_test, y_train, y_test, is_male_train, is_male_test

    return X_train, X_test, y_train, y_test
