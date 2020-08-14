import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb # use xgboost=1.0.2
import pickle

def read_excel(filePath):
    df = pd.read_excel(filePath, sheet_name='Sheet1_user_dt')
    df_1 = df.dropna()
    drop_colume = ['email',
                   'sn',
                   'username',
                   'reg_time',
                   'birthday',
                   'reg_type',
                   'reg_state',
                   'country',
                   'city',
                   'area',
                   'num_country',
                   'num_city',
                   'num_area',
                   'sell_type',
                   'sell_reason',
                   'sell_target']

    df_2 = df_1.drop(drop_colume, axis=1)
    return df_2

def split_data(df):
    train, test = train_test_split(df, test_size=0.3)
    target_factor = 'sc_day_month'
    drop_factor = ['used_day_month', 'used_freq_day', 'used_day_month',
                   'sc_times', 'no_sc_times', 'sc_days', 'no_sc_days', 'sc_freq_day']
    train_1 = train.drop(drop_factor, axis=1)
    test_1 = test.drop(drop_factor, axis=1)
    y_train = train_1[target_factor]
    x_train = train_1.drop(target_factor, axis=1)
    y_test = test_1[target_factor]
    x_test = test_1.drop(target_factor, axis=1)
    return x_train, y_train, x_test, y_test

def xgbmodel_fit(X_train, Y_train, learning_rate, n_estimators, max_depth, gamma):
    xgb_model = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                  gamma=gamma, subsample=0.6, objective='binary:logistic', nthread=4, scale_pos_weight=1)
    xgb_model.fit(X_train, Y_train, eval_metric='auc')
    return xgb_model
'''
def saveModel(model_name, model_fit):
    with open(str(model_name)+'.pickle', 'wb') as model:
        pickle.dump(model_fit, model)

def loadModel(model_name):
    with open(model_name+'.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
    return model
'''
def store_excel(good_learning_rate_default, good_n_estimators_default, good_max_depth_default, good_gamma_default):
    headers = ['learning_rate', 'n_estimators', 'max_depth', 'gamma']
    data = {
        headers[0]: [good_learning_rate_default],
        headers[1]: [good_n_estimators_default],
        headers[2]: [good_max_depth_default],
        headers[3]: [good_gamma_default],
    }

    df = pd.DataFrame(data)
    df.to_excel("weights.xlsx", header=headers, index=None)
    return "finished!"

def mainTask(df):
    learning_rate_default = 0.01
    n_estimators_default = 100
    max_depth_default = 3
    gamma_default = 0.05

    learning_rate_list = [i * 0.01 for i in range(1, 101, 2)]
    n_estimators_list = [i for i in range(100, 1100, 50)]
    max_depth_list = [i for i in range(3, 11, 1)]
    gamma_list = [i * 0.01 for i in range(5, 25, 5)]

    good_learning_rate_default = learning_rate_default
    good_n_estimators_default = n_estimators_default
    good_max_depth_default = max_depth_default
    good_gamma_default = gamma_default

    mean_error_rate_max = 1.0

    params = [learning_rate_list, n_estimators_list, max_depth_list, gamma_list]

    learning_rate = learning_rate_default
    n_estimators = n_estimators_default
    max_depth = max_depth_default
    gamma = gamma_default
    params_index = 0
    for param in params:
        mean_error_list = []
        error_rate_list = []
        if params_index == 0:
            learning_rate = learning_rate_default
            n_estimators = n_estimators_default
            max_depth = max_depth_default
            gamma = gamma_default
        elif params_index == 1:
            learning_rate = good_learning_rate_default
        elif params_index == 2:
            n_estimators = good_n_estimators_default
        else:
            gamma = good_gamma_default
        # print("param = ", param)

        for var in param:
            if params_index == 0:
                learning_rate = var

            elif params_index == 1:
                n_estimators = var

            elif params_index == 2:
                max_depth = var

            else:
                gamma = var

            print("%%%%%%%%%%%%%%%%%%")
            print("learning_rate = ", learning_rate)
            print("n_estimators = ", n_estimators)
            print("max_depth = ", max_depth)
            print("gamma = ", gamma)
            print("%%%%%%%%%%%%%%%%%%")

            # input("===")
            mean_error_list = []
            error_rate_list = []
            for i in range(10):
                x_train, y_train, x_test, y_test = split_data(df)
                # extract features
                headers = x_train.columns

                x_train, y_train, x_test, y_test = np.asarray(x_train), np.vstack((y_train)), np.asarray(
                    x_test), np.vstack((y_test))

                # categorize target value to [0, 1]
                criteria = 20
                y_train_cat = np.where(y_train > criteria, 1, 0)
                y_train_cat = np.vstack((y_train_cat))
                y_test_cat = np.where(y_test > criteria, 1, 0)

                model_fit = xgbmodel_fit(x_train, y_train_cat, learning_rate, n_estimators, max_depth, gamma)
                predictions = model_fit.predict(x_test)
                error_list = abs(predictions - y_test_cat.flatten())
                error_rate = sum(error_list) / len(error_list)
                error_rate_list.append(error_rate)
            print("===Round end===")

            mean_error_rate_list = np.mean(error_rate_list)
            if mean_error_rate_list < mean_error_rate_max:
                if params_index == 0:
                    # print("learning_rate = ", learning_rate)
                    good_learning_rate_default = learning_rate

                elif params_index == 1:
                    # print("n_estimators = ", n_estimators)
                    good_n_estimators_default = n_estimators

                elif params_index == 2:
                    # print("max_depth = ", max_depth)
                    good_max_depth_default = max_depth

                else:
                    # print("gamma = ", gamma)
                    good_gamma_default = gamma

                mean_error_rate_max = mean_error_rate_list

        params_index += 1

        print("%%%%%%%%%%%%%%%%%%")
        print("good_learning_rate_default = ", good_learning_rate_default)
        print("good_n_estimators_default = ", good_n_estimators_default)
        print("good_max_depth_default = ", good_max_depth_default)
        print("good_gamma_default = ", good_gamma_default)
        print("%%%%%%%%%%%%%%%%%%")

    return good_learning_rate_default, good_n_estimators_default, good_max_depth_default, good_gamma_default
