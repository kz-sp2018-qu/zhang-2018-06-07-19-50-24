import pandas as pd
from sklearn.externals import joblib
import numpy as np
import glob


# GIVEN: {"Y":"snp","Algorithm":"MLPRegressor","X"={"nyse":[736],"dax":[123}}
# RETURNS : [list of values]
def selectModel(inp):
    # print ("Inp inside selectModel : ",inp)
    classificationModels=['LogisticRegression', 'NeuralNetwork',
                          'RandomForestClassifier']

    best_probs=1
    best_scores=0
    response ={}
    for algo in classificationModels:
        model = readModel(algo)
        test_predictors = constructdf(inp)
        probs, scores  = makePrediction(test_predictors, model[algo])
        if scores > best_scores:
            best_scores = scores
        if algo =='LogisticRegression':
            best_probs= probs
        response[algo+"_CreditScore"] = str(scores[0])
        # print("Predicted credit score : ", scores[0])
    response["DefaultProbability"] = str(best_probs[0])
    return response, best_probs, best_scores


def readModel(algo):
    model=joblib.load("../output/{0}.pkl".format(algo))
    # print("Model read from pickle file : ",model)
    return model

def constructdf(inp):
    df=createEmptyDataFrame()
    df = addData(df.copy(),inp)
    return df.copy()

def createEmptyDataFrame():
    features = ['fico_range_low', 'fico_range_high', 'inq_last_6mths',
                'home_ownership', 'annual_inc', 'loan_amnt']

    df = pd.DataFrame(columns = features)
    return df.copy()

def addData(df, inp):
    df = df.append(inp, ignore_index=True)
    return df.copy()

def makePrediction(test_predictors, model):
    # predicted = model.predict(test_predictors.as_matrix())
    probs, scores = cal_credit(model, test_predictors)
    return probs, scores

# Calculate credit score
def calculate_score(log_odds):
    # 300 baseline + (40 points equals double risk) * odds
    return 300 + (63 / np.log(2)) * 2.0*(-log_odds)

def cal_credit(model, test_predictors):
    probs = model.predict_proba(test_predictors.as_matrix())[:,1]
    # print ("Probability that customer will default : ", probs)
    log_probs = model.predict_log_proba(test_predictors.as_matrix())[:,1]
    scores = calculate_score(log_probs)
    # print ("Credit Score : ", scores)
    return probs, scores
