'''
Linear regression

Ata Algan

Main file for linear regression and model selection.
'''

import numpy as np
from sklearn.model_selection import train_test_split
import util


class DataSet(object):
    '''
    Class for representing a data set.
    '''

    def __init__(self, dir_path):
        '''
        Class for representing a dataset, performs train/test
        splitting.

        Inputs:
            dir_path: (string) path to the directory that contains the
              file
        '''

        parameters_dict = util.load_json_file(dir_path, "parameters.json")
        self.feature_idx = parameters_dict["feature_idx"]
        self.name = parameters_dict["name"]
        self.target_idx = parameters_dict["target_idx"]
        self.training_fraction = parameters_dict["training_fraction"]
        self.seed = parameters_dict["seed"]
        self.labels, self.data = util.load_numpy_array(dir_path, "data.csv")

        # do standardization before train_test_split
        if(parameters_dict["standardization"] == "yes"):
            self.data = self.standardize_features(self.data)

        self.training_data, self.testing_data = train_test_split(self.data,
            train_size=self.training_fraction, test_size=None,
            random_state=self.seed)

    # data standardization
    def standardize_features(self, data): 
        '''
        Standardize features to have mean 0.0 and standard deviation 1.0.
        Inputs:
          data (2D NumPy array of float/int): data to be standardized
        Returns (2D NumPy array of float/int): standardized data
        '''
        mu = data.mean(axis=0) 
        sigma = data.std(axis=0) 
        return (data - mu) / sigma

class Model(object):
    '''
    Class for representing a model.
    '''
 
    def __init__(self, dataset, feature_idx):
        '''
        Construct a data structure to hold the model.
        Inputs:
            dataset: an dataset instance
            feature_idx: a list of the feature indices for the columns (of the
              original data array) used in the model.
        '''
        self.dataset = dataset
        self.feature_idx = feature_idx
        self.target_idx = dataset.target_idx
        self.X_train = util.prepend_ones_column(dataset.training_data[:, feature_idx])
        self.y_train = dataset.training_data[:, dataset.target_idx]
        self.X_test = util.prepend_ones_column(dataset.testing_data[:, feature_idx])
        self.y_test = dataset.testing_data[:, dataset.target_idx]
        self.beta = util.linear_regression(self.X_train, self.y_train)
        self.R2 = self.calculate_R2(self.X_train, self.y_train)
 
    def __repr__(self):
        '''
        Format model as a string.
        '''
        str_repr = "{} ~ {} + ".format(self.dataset.labels[self.target_idx], self.beta[0])
        for i, idx in enumerate(self.feature_idx):
            str_repr += "{} * {}".format(self.beta[i + 1], self.dataset.labels[idx])
 
        return str_repr
    
    def calculate_R2(self, x, y_true):
        y_pred = util.apply_beta(self.beta, x)
        y_mean = np.mean(y_true)
        total_variance = np.sum((y_true - y_mean) ** 2)
        explained_variance = np.sum((y_true - y_pred) ** 2)
        R2 = 1 - explained_variance / total_variance

        return R2
    
def compute_single_var_models(dataset):
    '''
    Computes all the single-variable models for a dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        List of Model objects, each representing a single-variable model
    '''

    return [Model(dataset, [i]) for i in dataset.feature_idx]


def compute_all_vars_model(dataset):
    '''
    Computes a model that uses all the feature variables in the dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object that uses all the feature variables
    '''

    return Model(dataset, dataset.feature_idx)


def compute_best_pair(dataset):
    '''
    Find the bivariate model with the best R2 value

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object for the best bivariate model
    '''

    best_bimodel = Model(dataset, dataset.feature_idx[:2])
    best_R2 = best_bimodel.R2

    for i in dataset.feature_idx:
        for j in range(i + 1, len(dataset.feature_idx)):
            feature_pair = [dataset.feature_idx[i], dataset.feature_idx[j]]
            bimodel = Model(dataset, feature_pair)
            my_model = bimodel.R2
            if my_model > best_R2:
                best_R2 = my_model
                best_bimodel = bimodel

    return best_bimodel



def forward_selection(dataset):
    '''
    Given a dataset with P feature variables, uses forward selection to
    select models for every value of K between 1 and P.

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A list (of length P) of Model objects. The first element is the
        model where K=1, the second element is the model where K=2, and so on.
    '''
    models = []

    for i in range(1, len(dataset.feature_idx) + 1):
        selected_features = []
        best_model = None
        best_R2 = None
        for j in range(i):
            remaining_features = set(dataset.feature_idx) - set(selected_features)
            for feature in remaining_features:
                current_features = selected_features + [feature]
                current_model = Model(dataset, current_features)
                if best_R2 is None or current_model.R2 > best_R2:
                    best_R2 = current_model.R2
                    best_model = current_model

            selected_features.append(best_model.feature_idx[-1])

        models.append(best_model)

    return models
    
    
def validate_model(dataset, model):
    '''
    Given a dataset and a model trained on the training data,
    compute the R2 of applying that model to the testing data.

    Inputs:
        dataset: (DataSet object) a dataset
        model: (Model object) A model that must have been trained
           on the dataset's training data.

    Returns:
        (float) An R2 value
    '''
    
    return model.calculate_R2(model.X_test, model.y_test)

