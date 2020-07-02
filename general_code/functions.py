import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics



def statistical_parity_difference(y_pred_a, y_pred_b):
    return y_pred_a.mean() - y_pred_b.mean()


def conditional_statistical_parity_difference(con_att, X_test, X_test_a, y_pred_a, X_test_b, y_pred_b):
    spd_list = []
    for att in con_att:
        for value in X_test[att].unique():
            try:
                y_pred_a_value = y_pred_a[X_test_a[att] == value]
                y_pred_b_value = y_pred_b[X_test_b[att] == value]
                if len(y_pred_a_value) != 0 and len(y_pred_b_value) != 0:
                    spd_value = statistical_parity_difference(y_pred_a_value, y_pred_b_value)
                    spd_list.append(spd_value)
            except KeyError as e1:
                print('KeyError occured')

    if len(spd_list) == 0:
        return 0.0
    else:
        return np.mean(spd_list)

def test_cm(cm):
    if np.size(cm)!= 4:
        return False
    else:
        [[tp_a, fp_a], [fn_a, tn_a]] = cm
        if tp_a == 0 and fn_a == 0:
            return False
        else:
            return True


def equal_opportunity_difference(y_pred_a, y_pred_b, y_test_a, y_test_b):
    cm_a = metrics.confusion_matrix(y_test_a, y_pred_a)
    cm_b = metrics.confusion_matrix(y_test_b, y_pred_b)
    if test_cm(cm_a) and test_cm(cm_b) :
        #False negative rate for group a
        [[tp_a, fp_a], [fn_a, tn_a]] = cm_a
        FNR_a = fn_a/(fn_a+tp_a)

        #False negative rate for group b
        [[tp_b, fp_b], [fn_b, tn_b]] = cm_b
        FNR_b = fn_b/(fn_b+tp_b)

        #Equal opportunity difference
        eod = FNR_a-FNR_b
        return eod
    else:
        return -1

def overall_accuracy_equality_difference(y_pred_a, y_pred_b, y_test_a, y_test_b):
    return metrics.accuracy_score(y_test_a, y_pred_a) - metrics.accuracy_score(y_test_b, y_pred_b)

def consistency(X_test, y_pred, indices):
    negative_bias = []
    positive_bias = []
    num_samples = np.shape(X_test)[0]
    consistency = 0.0
    for i in range(num_samples):
        difference = y_pred[i] - np.mean(y_pred[indices[i]])
        consistency += np.abs(difference)
        if difference > 0:
            negative_bias.append(X_test.iloc[i])
        elif difference < 0:
            positive_bias.append(X_test.iloc[i])
    consistency = 1.0 - consistency/num_samples
    return consistency, negative_bias, positive_bias

def euclidean_distance(l1, l2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(l1, l2)]))


def split_data(X_train, y_train, X_test, y_test, model, att, val1=1, val2=0):
    # Split the data on protected attribute
    X_train_a = X_train.loc[X_train[att] == val1]
    y_train_a = y_train.loc[X_train[att] == val1]

    X_train_b = X_train.loc[X_train[att] == val2]
    y_train_b = y_train.loc[X_train[att] == val2]

    X_test_a = X_test.loc[X_test[att] == val1]
    y_test_a = y_test.loc[X_test[att] == val1]

    X_test_b = X_test.loc[X_test[att] == val2]
    y_test_b = y_test.loc[X_test[att] == val2]

    if np.shape(X_test_a)[0] == 0:
        print('X_test does not have any samples in with ', att, ' is ', val1)
        return att

    if np.shape(X_test_b)[0] == 0:
        print('X_test does not have any samples in with ', att, ' is ', val2)

    # Predict
    y_pred_a = model.predict(X_test_a)
    y_pred_b = model.predict(X_test_b)

    return X_train_a, y_train_a, X_test_a, y_test_a, y_pred_a, X_train_b, y_train_b, X_test_b, y_test_b, y_pred_b

def bar_plot(att, val, x_label, size, save = False, title='no_title'):
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=size)
    ax.barh(np.arange(len(att)), val, align='center', height=0.8,)
    ax.set_yticks(np.arange(len(att)))
    ax.set_yticklabels(att)
    ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_xlabel(x_label)
    if(save):
        plt.tight_layout()
        plt.savefig(title + '.png')
    plt.show()