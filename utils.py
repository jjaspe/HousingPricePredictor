from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def score_model(model, X, y):    
    predicted_home_prices = model.predict(X)
    return mean_absolute_error(y, predicted_home_prices)

def drop_na_cols(X, X_test):
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
    return X.drop(cols_with_missing, axis=1), X_test.drop(cols_with_missing, axis=1)    

def fix_na_columns(X, X_test):
    my_imputer = SimpleImputer()
    imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
    imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

    # Imputation removed column names; put them back
    imputed_X.columns = X.columns
    imputed_X_test.columns = X_test.columns
    return imputed_X, imputed_X_test

def drop_non_numerical_cols(X, X_test):
    return X.select_dtypes(exclude=['object']), X_test.select_dtypes(exclude=['object'])

def fix_non_numerical_cols(X, X_test):
    X_train = X.copy()
    X_valid = X_test.copy()
    # Categorical columns in the training data
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely ordinal encoded
    good_label_cols = [col for col in object_cols if 
                    set(X_valid[col]).issubset(set(X_train[col]))]
        
    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols)-set(good_label_cols))  

    # Drop categorical columns that will not be encoded
    X_train = X_train.drop(bad_label_cols, axis=1)
    X_valid = X_valid.drop(bad_label_cols, axis=1)

    # Apply one hot encoding to low cardinality columns
    low_cardinality_cols = [col for col in good_label_cols if X_train[col].nunique() < 10]
    high_cardinality_cols = list(set(good_label_cols)-set(low_cardinality_cols))
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_X_train = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_X_valid = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinality_cols]))
    OH_X_train.index = X_train.index
    OH_X_valid.index = X_valid.index
    # Remove and replace
    X_train = X_train.drop(low_cardinality_cols, axis=1)
    X_valid = X_valid.drop(low_cardinality_cols, axis=1)
    X_train = pd.concat([X_train, OH_X_train], axis=1)
    X_valid = pd.concat([X_valid, OH_X_valid], axis=1)

    # Apply ordinal encoder to others
    ordinal = OrdinalEncoder()
    X_train[high_cardinality_cols] = ordinal.fit_transform(X_train[high_cardinality_cols])
    X_valid[high_cardinality_cols] = ordinal.transform(X_valid[high_cardinality_cols])

    return X_train, X_valid

def train_and_score(model, train_X, train_y, valid_X, valid_y):
    model.fit(train_X, train_y)
    score = score_model(model, valid_X, valid_y)
    return score

def train_and_score_forest_model(estimators, depth, train_X, train_y, valid_X, valid_y):
    model = RandomForestRegressor(n_estimators=estimators, max_depth=depth, criterion='absolute_error', random_state=0)
    return train_and_score(model, train_X, train_y, valid_X, valid_y)

def get_cols_by_type(X_train_full):    
    categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

    ordinal_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() >= 10 and 
                    X_train_full[cname].dtype == "object"
                    ]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if 
                    X_train_full[cname].dtype in ['int64', 'float64']]
    return categorical_cols, numerical_cols, ordinal_cols