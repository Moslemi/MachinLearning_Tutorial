import pandas as pd
import quandl,math
import numpy as np
#sklearn will scale the renge of data
from sklearn import preprocessing,svm, model_selection
from sklearn.linear_model import LinearRegression
# download data from Quandl
quandl.ApiConfig.api_key = '_KGNP_-WwzLzMTVJhdJn'

df = quandl.get('CHRIS/MGEX_IH1')

# Categorising the column based on what we need in df
df = df[['Open','High', 'Low','Last','Volume',]]

# creating new data form based on the apeard data
df['HL_PCT'] = ((df['High'] - df['Low'])/df['Last']*100.0)
df['PCT_CHANGE'] = ((df['Last'] - df['Open'])/df['Open']*100.0)

# having all the calculated column in one dataframe
df = df[['Last','HL_PCT','PCT_CHANGE','Volume']]

# consider one of the coloumn as forcasr_col
forcast_col = 'Last'

# rather than getting ride of data we can consider a value(like -99999) instead of "NaN" values to have better calculation
df.fillna(-99999,inplace=True)

# int(math.ceil) will round the number to a nearest intiger
forcast_out = int(math.ceil(0.01*len(df)))

# creating lable
# give a range to the data which should be shown in the figure (column lable) (here we ignore the top 1%)
df['label'] = df[forcast_col].shift(-forcast_out)

# ????
df.dropna(inplace=True)
# features will show as "X" and labels will show as "Y"

X = np.array(df.drop("label",1))  # we drop the fist colmen which is the name of lable in label column
Y = np.array(df["label"])

X = preprocessing.scale(X)
#X = X[:forcast_out+1]            #shifting scale
df.dropna(inplace =True)
Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.2)


# use different model to predict
clf = LinearRegression()

#clf = svm.SVR(kernel = 'poly')    # default is linear
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test,Y_test)

print(len(X),len(Y))
print(accuracy)






#print(df.head())
