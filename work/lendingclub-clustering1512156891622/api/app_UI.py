from flask import Flask, render_template, request, \
    json, url_for
import os
import joblib
# from searchSpark import *
import pandas as pd
from werkzeug.utils import secure_filename
import pygal
import numpy as np
from service.selectModel import *


app = Flask(__name__)
port = int(os.getenv('PORT', 5000))

@app.route("/",methods=['GET','POST'])
# , MaxFICOScore = 700, InqLast6mths = 10,
#          HomeOwnership= "RENT", AnnualIncome = 70000, LoanAmnt = 10000
def main():
    print ("insside main")
    if request.method == 'POST':
        print ("Inside Main as Post")
        MinFICOScore = request.form['MinFICOScore']
        MaxFICOScore = request.form['MaxFICOScore']
        InqLast6mths = request.form['InqLast6mths']
        print (type(request.form['HomeOwnership']), request.form[
            'HomeOwnership'])
        if request.form['HomeOwnership'] == 'RENT':
            HomeOwnership = 1
        elif request.form['HomeOwnership'] == 'OWN':
            HomeOwnership = 2
        else:
            HomeOwnership = 3

        AnnualIncome = request.form['AnnualIncome']
        LoanAmnt = request.form['LoanAmnt']

        inp={}
        inp['fico_range_low'] = float(MinFICOScore)
        inp['fico_range_high'] = float(MaxFICOScore)
        inp['inq_last_6mths'] = float(InqLast6mths)
        inp['home_ownership_num'] = float(HomeOwnership)
        inp['annual_inc'] = float(AnnualIncome)
        inp['loan_amnt'] = float(LoanAmnt)

        print ("Received Input : ",inp)
        results, probs, scores = selectModel(inp)
        print (results)

        print (MinFICOScore, MaxFICOScore, InqLast6mths, HomeOwnership,
               AnnualIncome, LoanAmnt)

        return render_template('index.html', DefaultProbability=results[
            'DefaultProbability'], RFS=results[
            'RandomForestClassifier_CreditScore'],
                               LRS=results['LogisticRegression_CreditScore'],
                               NNS=results['NeuralNetwork_CreditScore'])

    return render_template('index.html', DefaultProbability='None', RFS=None,
                           LRS=None, NNS=None)



@app.route('/showSignUp')
def showSignUp():

    # show the form, it wasn't submitted
    return render_template('showSignUp.html')

@app.route('/graph/')
def graph(chartID = 'chart_ID', chart_type = 'line', chart_height = 500):
    dataset=[[1408395614.0, 430.2], [1408395614.0, 431.13 ], [1408395617.0,
                                                              431.354]]
    chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
    probs=joblib.load("../output/defaultProbs.pkl")
    print (probs)
    series = [{"name": 'Label1', "data": list(probs)}]
    title = {"text": 'My Title'}
    xAxis = {"categories": ['xAxis Data1', 'xAxis Data2', 'xAxis Data3']}
    yAxis = {"title": {"text": 'yAxis Label'}}
    return render_template('index.html', chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)



# run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True) # deploy with debug=Falses



