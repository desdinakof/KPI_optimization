data['Saving accounts'] = data['Saving accounts'].map({"little":0,"moderate":1,"quite rich":2 ,"rich":3 });
data['Saving accounts'] = data['Saving accounts'].fillna(data['Saving accounts'].dropna().mean())

data['Checking account'] = data['Checking account'].map({"little":0,"moderate":1,"rich":2 });
data['Checking account'] = data['Checking account'].fillna(data['Checking account'].dropna().mean())

data['Sex'] = data['Sex'].map({"male":0,"female":1}).astype(float);

data['Housing'] = data['Housing'].map({"own":0,"free":1,"rent":2}).astype(float);

data['Purpose'] = data['Purpose'].map({'radio/TV':0, 'education':1, 'furniture/equipment':2, 'car':3, 'business':4,
       'domestic appliances':5, 'repairs':6, 'vacation/others':7}).astype(float);

data['Risk'] = data['Risk'].map({"bad" : 0, "good" : 1, });



submission_df = data.copy()

submission_df.to_csv('german_data_risky_mean.csv', index=False)


data['begintime'] = data['begintime'].astype(float)

data['begintime']=data['begintime'].replace(to_replace="-", value=".")