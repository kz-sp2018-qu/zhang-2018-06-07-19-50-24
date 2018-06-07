from service.selectModel import *
import pandas as pd
import pickle as pkl

df = pd.read_csv("../data/LoanStats3a.csv", skiprows=1)
print ("DataFrame size : ",df.shape)
# print (df.head())

df = df.dropna(axis=1, thresh=2)

df['year_issued'] = df.issue_d.apply(lambda x: int(x.split("-")[0]))
df_term = df[df.year_issued < 2013]


def convert(value):
    if value == 'RENT':
        return 1
    if value == 'OWN':
        return 2
    if value == 'MORTGAGE':
        return 3


bad_indicators = [
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
    "Charged Off"
]

df_term['is_bad'] = df_term.loan_status.apply(
    lambda x: 1 if x in bad_indicators else 0)
df_term['home_ownership_num'] = df_term.home_ownership.apply(convert)
features = ['fico_range_low', 'fico_range_high', 'inq_last_6mths',
            'home_ownership_num', 'annual_inc', 'loan_amnt', 'is_bad']
df_term = df_term[features]

df_term = df_term.fillna(method='backfill')

df_term = df_term.loc[:,df_term.columns!='is_bad']

features = ['fico_range_low', 'fico_range_high', 'inq_last_6mths',
            'home_ownership', 'annual_inc', 'loan_amnt']

df = pd.DataFrame(columns = features)
defaultProbabilities = []

for i in range(len(df_term)):
    temp={}
    temp['fico_range_low']=df_term.iloc[i,:][0]
    temp['fico_range_high'] = df_term.iloc[i,:][1]
    temp['inq_last_6mths'] = df_term.iloc[i,:][2]
    temp['home_ownership'] = df_term.iloc[i,:][3]
    temp['annual_inc'] = df_term.iloc[i,:][4]
    temp['loan_amnt'] = df_term.iloc[i,:][5]
    # print (temp)
    # df = df.append(temp, ignore_index=True)
    # print (temp)
    response, probs, scores = selectModel(temp)
    # print (response['DefaultProbability'])
    defaultProbabilities.append(response['DefaultProbability'])

with open("../output/defaultProbs.pkl", 'wb') as handle:
    pkl.dump(defaultProbabilities, handle, protocol=pkl.HIGHEST_PROTOCOL)
# results, probs, scores = selectModel(inp)