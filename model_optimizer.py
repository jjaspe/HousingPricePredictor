
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils import *
from pipelines import *

def search_random_forest_models(X_train, y_train, X_valid, y_valid):
    estimators = [300, 400, 500]
    depths = [8,9,10]
    scores = pd.DataFrame([[est, depth, train_and_score_forest_model(est, depth, X_train, y_train, X_valid, y_valid)] for est in estimators for depth in depths])
    sorted_scores = scores.sort_values(2)