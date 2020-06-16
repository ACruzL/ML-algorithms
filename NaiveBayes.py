import pandas as pd
import numpy as numpy
from pprint import pprint
from sklearn import naive_bayes as nb


def naivebayes(df):
    #Getting features and outcome names
    features = df.columns[:-1]
    outcomes = df.columns[-1]
    possible_outcomes = df[outcomes].unique()

    #Calculate probability of each outcome
    outcomes_count = df[outcomes].value_counts()
    outcomes_count /= outcomes_count.sum()
    outcomes_probabilities = outcomes_count.to_dict()


    #Calculate P(xi | yj) for each xi and yj
    frequency_dict = {}
    for feature in features:
        aux_df = df.groupby([feature, outcomes]).size().unstack().fillna(0)
        aux_df /= aux_df.sum()

        frequency_dict[feature] = aux_df

    #Calculate the predictions of the dataset
    predicts = [{} for _ in range(len(df))]
    for index, row in df.iterrows():
        result = None
        for outcome in possible_outcomes:
            prod = 1
            for feature in features:
                prod *= frequency_dict[feature].loc[row[feature], outcome]
            prod *= outcomes_probabilities[outcome]
            predicts[index][outcome] = prod

    predicts = [max(dic, key=dic.get) for dic in predicts]
    pprint(predicts == df[outcomes])
    

df = pd.read_csv('datasets/playgolf.csv')

naivebayes(df)

