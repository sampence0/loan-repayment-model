# Function definitions for data analysis and preprocessing

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


##########################################################################
##########################################################################

def count_na(df):
    # Count NA values in features
    print(f'Amount of NA\'s in each column: ')
    print('-'*50)
    count = 0
    for col in df:
        print(f'{count}. {col}: {df[col].isna().sum()}')
        count += 1 
    print('-'*50, '\n')
    return f'Amount of Columns: {count}'



##########################################################################
##########################################################################

def get_rows_with_nans(df, columns):
    """
    Pass df and list of columns to be conditionalized.
    """
    return df[df[columns].isnull().any(axis=1)]



##########################################################################
##########################################################################

def missing_docs(df, columns):
    df['missingdocs'] = df[columns].apply(lambda x: x == 1).sum(axis=1)
    df = df.drop(columns=columns)
    return df

##########################################################################
##########################################################################

def correlation_analysis(df, n):
    """
    Returns the top n columns by magnitude of correlation to TARGET.
    """
    corrs_dict = {}
    for col in df.columns:
        if col != 'TARGET':  # Avoid calculating correlation with the target itself
            correlation = df['TARGET'].corr(df[col])
            corrs_dict[col] = correlation
    
    # Sort correlations in descending order by magnitude of correlation to target
    sorted_corrs = sorted(corrs_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_half_corrs = sorted_corrs[:n]
    
    # Print the top n correlations
    for col, corr in top_half_corrs:
        print(f'Correlation between {col} and TARGET')
        print(col, ":", corr)
        print('-'*50)
    
    return "Correlation Analysis Complete."

##########################################################################
##########################################################################

def weight_of_evidence_encode(df, column_name, label):
    """
    Pass train df, name of column to be encoded, and target column.
    """
    
    pos = df.groupby(column_name)[label].mean()
    neg = 1 - pos
    weight_of_evidence = np.log(pos / neg)
    
    woe_dict = weight_of_evidence.to_dict()
    woe_encoded = df[feature].map(woe_dict)
    
    return woe_encoded

##########################################################################
##########################################################################

def parameters_and_importances(df, target_col):
    """
    Pass preprocessed training data and target column separately!!!
    """
    print('Beginning Randomized Search Algorithm...')
          
    X = df
    y = target_col
    
    hyperparams = {
        'n_estimators': [50, 100, 150, 200, 400],
        'max_depth': [None, 10, 25, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=81)
    
    rf_params = RandomizedSearchCV(estimator=rf, param_distributions=hyperparams,
                                  n_iter=50, cv=3, random_state=81, n_jobs=-1)
    
    rf_params.fit(X, y)
    best = rf_params.best_estimator_
    importances = dict(zip(X.columns, best.feature_importances_))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Optimal Hyperparameters: {rf_params.best_params_}")
    scores = cross_val_score(best, X, y, cv=cv, n_jobs=-1)
    print(f"Cross-validation Scores: {scores}")
    print(f"Mean CV Score: {np.mean(scores):.4f}")
    
    y_pred = best.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    
    return sorted_importances, rf_params.best_params_


