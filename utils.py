import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

def load_data():
  df = pd.read_csv('data/train.csv')
  return df

def preprocess_data(df: pd.DataFrame):
  df = df.drop(['PassengerId', 'Name'], axis=1)
  categorical_features = ['Pclass', 'Sex', 'Cabin', 'Embarked']
  df['Cabin'] = df['Cabin'].apply(
      lambda s: s.split(' ')[0] if pd.notnull(s) else np.nan)
  df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else np.nan)
  df['Room'] = df['Cabin'].apply(lambda s: int(
      s[1:]) if pd.notnull(s) and s[1:] != '' else np.nan)
  df = df.drop('Cabin', axis=1)
  df['Ticket'] = df['Ticket'].map(lambda s: s.split(' ')[-1])
  df['Ticket'] = df['Ticket'].apply(lambda s: s if s.isnumeric() else np.nan)
  df['Ticket'] = pd.to_numeric(df['Ticket'])
  categorical_features.remove('Cabin')
  categorical_features.append('Deck')
  df['Pclass'] = df['Pclass'].astype(str)
  dummies = pd.get_dummies(df[categorical_features])
  df = df.drop(categorical_features, axis=1)
  df = pd.concat([df, dummies], axis=1)
  return df

def split_data(df: pd.DataFrame):
  train_val, test = train_test_split(df, test_size=0.1)
  train, val = train_test_split(train_val, test_size=0.1/0.9)
  X_train = train.drop('Survived', axis=1)
  y_train = train['Survived']
  X_val = val.drop('Survived', axis=1)
  y_val = val['Survived']
  X_test = test.drop('Survived', axis=1)
  y_test = test['Survived']
  return X_train, y_train, X_val, y_val, X_test, y_test

def tune_xgb_model(xgb_model, X_train, y_train, X_val, y_val, default_params, target_param, values, eval_metric, plot=False, overwrite_params=False):
  best_score = -1
  best_value = values[0]
  scores = []
  for i in range(len(values)):
    params = default_params.copy()
    params[target_param] = values[i]
    xgb_model.set_params(**params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    score = xgb_model.evals_result()['validation_0'][eval_metric][-1]
    scores.append(score)
    if score < best_score or best_score == -1:
      best_score = score
      best_value = values[i]
  if plot:
    x_axis = values
    fig, ax = plt.subplots()
    ax.plot(x_axis, scores)
    ax.legend()
    plt.ylabel('Scores')
    plt.xlabel('Values')
    plt.title(f'XGBoost {target_param} Tuning')
    plt.show()
  if overwrite_params:
    default_params[target_param] = best_value
  return best_score, best_value, scores
  
def eval_xgb_model(xgb_model, X_train, y_train, X_val, y_val, verbose=False):
    train_accuracy = metrics.accuracy_score(xgb_model.predict(X_train), y_train)
    validation_accuracy = metrics.accuracy_score(xgb_model.predict(X_val), y_val)
    if (verbose):
        print(f'Train accuracy:      {train_accuracy:.2%}')
        print(f'Validation accuracy: {validation_accuracy:.2%}')
    return train_accuracy, validation_accuracy