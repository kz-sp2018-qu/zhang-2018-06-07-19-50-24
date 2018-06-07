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
        means=dict(pd.read_pickle('../output/means.pkl'))
        stds=dict(pd.read_pickle('../output/stds.pkl'))
        test_predictors = constructdf(inp)
        # means = constructdf(means)
        # stds = constructdf((stds))
        print ("test_predictors dataframe : ",test_predictors)
        # test_predictors= (test_predictors - means)/stds

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
                'home_ownership_num', 'annual_inc', 'loan_amnt']

    df = pd.DataFrame(columns = features)
    return df.copy()

def addData(df, inp):
    df = df.append(inp, ignore_index=True)
    return df.copy()

def computedifference(test_predictors, means):
    test_predictors['loan_amnt'][0] -= means['loan_amnt']
    test_predictors['fico_range_high'][0] -= means['fico_range_high']
    test_predictors['inq_last_6mths'][0] -= means['inq_last_6mths']
    test_predictors['home_ownership_num'][0] -= means['home_ownership_num']
    test_predictors['fico_range_low'][0] -= means['fico_range_low']
    test_predictors['annual_inc'][0] -= means['annual_inc']
    print ("Done subtracting")
    return test_predictors.copy()


def computestd(test_predictors, std):
    test_predictors['loan_amnt'][0] = test_predictors['loan_amnt']/std[
        'loan_amnt']
    test_predictors['fico_range_high'][0] = test_predictors[
                                             'fico_range_high']/std['fico_range_high']
    test_predictors['home_ownership_num'][0] /= std['home_ownership_num']
    test_predictors['inq_last_6mths'][0] = test_predictors[
                                             'inq_last_6mths']/std['inq_last_6mths']
    test_predictors['fico_range_low'][0] = test_predictors[
                                             'fico_range_low']/std['fico_range_low']

    test_predictors['annual_inc'][0] /= std['annual_inc']
    return test_predictors.copy()

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
    print ("scores : ", scores)
    return probs, scores
