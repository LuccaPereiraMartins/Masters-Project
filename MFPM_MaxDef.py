import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Input Parameters

GO_E, GO_phi, TF_c, TF_E, TF_phi, max_def = 83800, 43.5, 7.7, 12400, 31.1, 20.51

inputs = [GO_E, GO_phi, TF_c, TF_E, TF_phi, max_def]
inputs_as_strings = ['GO_E', 'GO_phi', 'TF_c', 'TF_E', 'TF_phi', 'max_def']

iterations = 50

def sample_gaussian(iterations):
    
    np.random.seed(1)

    GO_E_gs = [np.random.normal(GO_E,0.05*GO_E) for x in range(iterations)]
    GO_phi_gs = [np.random.normal(GO_phi,0.05*GO_phi) for x in range(iterations)]
    TF_c_gs = [np.random.normal(TF_c,0.05*TF_c) for x in range(iterations)]
    TF_E_gs = [np.random.normal(TF_E,0.05*TF_E) for x in range(iterations)]
    TF_phi_gs = [np.random.normal(TF_phi,0.05*TF_phi) for x in range(iterations)]
    max_def_gs = [np.random.normal(max_def,0.05*max_def) for x in range(iterations)]
    inputs_gs = [GO_E_gs,GO_phi_gs,TF_c_gs,TF_E_gs,TF_phi_gs,max_def_gs]
    return inputs_gs

def run(iterations):
    inputs_gs = sample_gaussian(iterations)
    experimental_df = pd.DataFrame(inputs_gs,index = inputs_as_strings).transpose()
    #experimental_csv = experimental_df.to_csv(path_or_buf='Experimental.csv')


    # Import low-fidelity data

    low_df = pd.read_csv('Dataset.csv')
    low_inputs,low_output = low_df.drop(labels=['Phase_3C_deflection','Max_BM','Max_anchor'],axis=1), 1000*low_df[['Phase_3C_deflection']]
    low_output.columns=['max_def']

    # Import high-fidelity data

    data_db = pd.DataFrame(inputs_gs).transpose()
    data_db.columns = inputs_as_strings
    high_inputs = data_db[['GO_E', 'GO_phi', 'TF_c', 'TF_E', 'TF_phi']]
    high_output = data_db[['max_def']]

    # Now perform multi-fidelity learning (MFPM)

    xl, xh = normalize(low_inputs,axis=1), normalize(high_inputs,axis=1)
    yl, yh = normalize(low_output, norm='l2',axis=0), normalize(high_output, norm='l2',axis=0)
    yl, yh = np.ravel(yl), np.ravel(yh)

    xltrain, xltest, yltrain, yltest = train_test_split(xl,yl,
                                                            test_size=0.3,
                                                            random_state=1)

    xhtrain, xhtest, yhtrain, yhtest = train_test_split(xh,yh,
                                                            test_size=0.3,
                                                            random_state=1)

    # Stack test datasets
    xtest = np.vstack((np.array(xltest),np.array(xhtest)))
    ytest = np.hstack((yltest,yhtest))

    mlow = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(20,15,10),           
                        activation='logistic',          
                        solver = 'adam',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-4,
                        max_iter=10000).fit(xltrain,yltrain)

    mhighonly = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(20,15,10),           
                        activation='logistic',          
                        solver = 'adam',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-4,
                        max_iter=10000).fit(xhtrain,yhtrain)

    ystar = MLPRegressor.predict(mlow,xhtrain)
    ytemp = MLPRegressor.predict(mlow,xtest)
    yhtemp = MLPRegressor.predict(mlow,xhtest)

    yhstar = MLPRegressor.predict(mhighonly,xhtest)

    ystar = normalize(ystar.reshape(-1,1), axis=0)

    xhtrain = np.hstack((xhtrain,ystar.reshape(len(ystar),1)))
    xtest = np.hstack((xtest,ytemp.reshape(len(ytemp),1)))
    xhtest = np.hstack((xhtest,yhtemp.reshape(len(yhtemp),1)))

    mhigh = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(20,15,10),           
                        activation='logistic',          
                        solver = 'adam',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-4,
                        max_iter=10000).fit(xhtrain,yhtrain)

    ytempstar = MLPRegressor.predict(mhigh,xtest)
    yhtempstar = MLPRegressor.predict(mhigh,xhtest)

    MSE_ml = mean_squared_error(ytemp,ytest)
    MSE_mh = mean_squared_error(ytempstar,ytest)
    MSE_ml_high = mean_squared_error(yhtemp,yhtest)
    MSE_mh_high = mean_squared_error(yhtempstar,yhtest)
    MSE_mhonly = mean_squared_error(yhstar,yhtest)

    MSE = [MSE_ml,MSE_mh,MSE_ml_high,MSE_mh_high,MSE_mhonly]
    return MSE

xvals = range(10,50)
MSE = [run(i) for i in xvals]
MSE_df = pd.DataFrame(MSE)

MSE_low = MSE_df[2]
MSE_multi = MSE_df[3]
MSE_high = MSE_df[4]


MSE_it30 = MSE[:][30-10]
#print(MSE_it30)

print('high-fidelity test dataset on ml MSE: ', MSE_it30[-3])
print('high-fidelity test dataset on mh MSE: ', MSE_it30[-1])
print('high-fidelity test dataset on mf MSE: ', MSE_it30[-2])


plt.plot(xvals,MSE_low)
plt.plot(xvals,MSE_high)
plt.plot(xvals,MSE_multi)
plt.xlabel('Iterations of High-Fidelity Data')
plt.ylabel('Model MSE')
plt.legend(['Low-fidelity Model','High-fidelity Model','Multi-fidelity Model'])
plt.show()

# Clearly there is significant improvement when test dataset consists only of high-fidelity, or 'true', data
# There is also sufficient improvement from mhonly against mhigh when evaluating xhtest
# Increases signficantly as iterations , or hence test dataset size, increases, as expected
# Interesting to see how low ml MSE is on mixed test dataset, why is this the case?
# Currently, value of def_max has no impact on MSE, is this because normalizing focus on the relative relationship
# rather than the absolute values? Makes no sense if not normalized.
# Does the graph even make any sense if there is no lower bound of iteration?

# high-fidelity test dataset on ml MSE:  0.022459845981991532
# high-fidelity test dataset on mh MSE:  0.0007850708109554271
# high-fidelity test dataset on mf MSE:  0.00026572891574103613

# The MFPM approach seems to work when predicting the max deflection. Extend this to deflection wrt to depth
# possibly in the form of depth profiles