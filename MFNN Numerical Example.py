# Continuous function with linear correlation


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt

random.seed(1)

def highfid(x):
    return ((((6*x) - 2)**2) * np.sin((12*x) -4))


def lowfid(A,B,C,x):
    return ((A*(((6*x) - 2)**2) * np.sin((12*x) -4)) + B*(x-0.5) + C)




def gen_data():

    Xlow = np.linspace(0,1.5,20)
    Xmed = np.linspace(0,1.5,10)
    Xhigh = np.linspace(0,1.5,5)

    yhigh, ylow, ymed = [],[],[]
    for entry in Xhigh: yhigh.append(highfid(entry))
    for entry in Xlow: ylow.append(lowfid(x=entry,A=0.5,B=10,C=-5))
    for entry in Xmed: ymed.append(lowfid(x=entry,A=0.8,B=2,C=-1))

    #plt.scatter(Xhigh,yhigh,c='g'),plt.scatter(Xmed,ymed,c='orange'),plt.scatter(Xlow,ylow,c='r')
    #plt.legend(['High Fidelity function','Medium Fidelity function','Low Fidelity function',
    #            'High Fidelity data','Medium Fidelity data','Low Fidelity data'])

    return [Xlow,Xmed,Xhigh,ylow,ymed,yhigh]

#plot_fidelities(),gen_data()
#plt.show()

pXlow,pXmed,pXhigh,pylow,pymed,pyhigh = gen_data()
    
def plot_fidelities():

    x = np.linspace(0,2,1000)

    yhigh, ylow, ymid = [],[],[]

    for entry in x: 
        yhigh.append(highfid(entry))
        ylow.append(lowfid(x=entry,A=0.5,B=10,C=-5))
        ymid.append(lowfid(x=entry,A=0.8,B=2,C=-1))

    plt.plot(x,yhigh,c='r',alpha=0.8)
    plt.plot(x,ymid,c='orange',alpha=0.8)
    plt.plot(x,ylow,c='pink',alpha=0.8)

    plt.scatter(pXlow,pylow,s=10,c='pink',alpha=0.5)
    plt.scatter(pXmed,pymed,s=10,c='orange',alpha=0.5)
    plt.scatter(pXhigh,pyhigh,c='r',s=10,alpha=0.5)

    plt.xlabel('X'),plt.ylabel('y')
    plt.grid(visible=True, which='major', axis='both')
    plt.xlim(-0.1,1.6)  
    plt.legend([r'$y_{L}$',r'$y_{M}$',r'$y_{H}$',r'$D_{L}$',r'$D_{M}$',r'$D_{H}$'])
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()   


plot_fidelities()

def model(xtrain,ytrain,n):

    MLPregr = MLPRegressor(random_state=1,          
                    hidden_layer_sizes=n,           #a robust choice of network
                    activation='logistic',          #consider tanh and sigmoid functions too, for non-linearity in networks with 2+ hidden layers
                    solver = 'adam',            
                    batch_size='auto',
                    early_stopping=False,
                    learning_rate='adaptive',
                    tol=1e-4,
                    max_iter=10000).fit(xtrain,ytrain)

    return MLPregr 

data = gen_data()
scaler = StandardScaler() 

def low():
    # Generate tests samples between [0,1.5]
    samples,xtest = 10,[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5))
    xtest = sorted(xtest)
    xtest = np.array(xtest).reshape(len(xtest),1)

    # Access training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Create low model, 3 hidden layers
    mlow = model(X,y,(20,15,10))

    # Predict and evaluate on test samples
    ystarlow = MLPRegressor.predict(mlow,xtest)
    yreal = []
    for x in xtest: yreal.append(highfid(x))

    ystarlow = ystarlow.reshape((len(ystarlow),1))
    yreal = np.array(yreal).reshape(len(yreal),1)

    mse_all = []
    for index,entry in enumerate(yreal):
        mse_all.append(mean_squared_error(entry,ystarlow[index]))
    
    mse_avg = np.mean(mse_all)
    r2 = mlow.score(xtest,yreal)
    print(round(mse_avg,5),round(r2,5))

    # Use more values to plot real and predicted functions
    xfull = np.linspace(0,2,1000)
    xfull = xfull.reshape(len(xfull),1)
    yrealfull = [highfid(x) for x in xfull]
    ystarlowfull = MLPRegressor.predict(mlow,xfull)

    # Define the MSE of prediction
    M = [float((y-ystarlowfull[x])**2) for x,y in enumerate(yrealfull)]
    dyfitfull = [2 * np.sqrt(x) for x in M]

    # Create the 2 S.D.s envelope (95% error margin)
    ydynplus = [float(x+y) for x,y in zip(yrealfull,dyfitfull)]
    ydynmin = [float(x-y) for x,y in zip(yrealfull,dyfitfull)]

    # Plot training data
    plt.scatter(X,y,s=10,c='r')

    # Plot real and predicted function
    plt.plot(xfull,yrealfull,c='g')
    plt.plot(xfull,ystarlowfull,c='b')

    # Plot uncertainty envelope of predicted function
    
    # plt.fill_between(np.ravel(xfull), ydynplus, ydynmin,
    #             color='gray', alpha=0.2)

    # Labels & extras
    plt.xlabel('X'),plt.ylabel('y'),plt.grid(True)
    plt.legend(['Low-fidelity training data','True distribution','Low-fidelity model output','95% error margin'])
    plt.show()

#low()

def MSE(model,xtest):
    # Predict and evaluate on test samples
    ystarlow = MLPRegressor.predict(model,xtest)
    yreal = []
    for x in xtest: yreal.append(highfid(x))

    ystarlow = ystarlow.reshape((len(ystarlow),1))
    yreal = np.array(yreal).reshape(len(yreal),1)

    mse_all = []
    for index,entry in enumerate(yreal):
        mse_all.append(mean_squared_error(entry,ystarlow[index]))
    
    mse_avg = np.mean(mse_all)
    r2 = model.score(xtest,yreal)
    return(round(mse_avg,5),round(r2,5))
   
def BFWL():
    # Same as low()
    # Generate tests samples between [0,1.5]
    samples,xtest = 10,[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5))
    xtest = sorted(xtest)
    xtest = np.array(xtest).reshape(len(xtest),1)

    # Access training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Get medium-fidelity data
    Xmed, ymed = np.array(data[1]),np.array(data[4])
    Xmed, ymed = Xmed.reshape(len(Xmed),1), ymed.reshape(len(ymed),1)
    ymed = np.ravel(ymed)

    # Create low model, 3 hidden layers
    mlow = model(X,y,(20,15,10))

    # Get MSE and R2 (low)
    MSElow = MSE(mlow,xtest)

    # Partial fit model with medium-fidelity data
    iterations = 1
    for i in range(iterations):
        mlow.partial_fit(Xmed,ymed)

    # Get MSE and R2 (bu-fidelity)
    MSEmed = MSE(mlow,xtest)

    print(MSElow,MSEmed)

def BFWLiterated():
    # Same as low()
    # Generate tests samples between [0,1.5]
    samples,xtest = 10,[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5))
    xtest = sorted(xtest)
    xtest = np.array(xtest).reshape(len(xtest),1)

    # Access training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Get medium-fidelity data
    Xmed, ymed = np.array(data[1]),np.array(data[4])
    Xmed, ymed = Xmed.reshape(len(Xmed),1), ymed.reshape(len(ymed),1)
    ymed = np.ravel(ymed)

    # Create low model, 3 hidden layers
    mlow = model(X,y,(20,15,10))
    mmed = model(X,y,(20,15,10))

    # Partial fit model with medium-fidelity data over iterations
    MSEall = []
    iterations = 20
    for i in range(iterations):
        mmed.partial_fit(Xmed,ymed)

        # Append iteration MSE
        MSEall.append(MSE(mmed,xtest)[0])

    #plt.plot(MSEall),plt.xlabel('X'),plt.ylabel('MSE'),plt.title('MSE of BWFL model against iteration cycles')
    #plt.show()

    # Use more values to plot real and predicted functions
    xfull = np.linspace(0,2,1000)
    xfull = xfull.reshape(len(xfull),1)
    yrealfull = [highfid(x) for x in xfull]
    ystarlowfull = MLPRegressor.predict(mlow,xfull)
    ystarmedfull = MLPRegressor.predict(mmed,xfull)

    # Define the MSE of prediction
    M = [float((y-ystarmedfull[x])**2) for x,y in enumerate(yrealfull)]
    dyfitfull = [2 * np.sqrt(x) for x in M]

    # Create the 2 S.D.s envelope (95% error margin)
    ydynplus = [float(x+y) for x,y in zip(yrealfull,dyfitfull)]
    ydynmin = [float(x-y) for x,y in zip(yrealfull,dyfitfull)]

    # Plot training data
    plt.scatter(X,y,s=10,c='r')
    plt.scatter(Xmed,ymed,s=10,c='orange')

    # Plot real and predicted function
    plt.plot(xfull,yrealfull,c='g')
    plt.plot(xfull,ystarlowfull,c='r')
    plt.plot(xfull,ystarmedfull,c='b')

    # Plot uncertainty envelope of predicted function
    
    plt.fill_between(np.ravel(xfull), ydynplus, ydynmin,
                color='gray', alpha=0.2)

    # Labels & extras
    plt.xlabel('X'),plt.ylabel('y'),plt.grid(True)
    plt.legend(['Low-fidelity training data','Medium-fidelity training data','True distribution','Low-fidelity model output','Medium-fidelity model output','95% error margin'])
    plt.xlim=1.5
    plt.show()
    
    # After many iterations, it almost entirely dismisses the low-fidelity data and fits only to the medium-fidelity training set.
    # Only where medium-fidelity data exists, else no changes 

def MFWL():
    # Generate tests samples between [0,1.5] and ones between [1.5,2.0]
    samples,xtest,xtestextra = 20,[],[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5)), xtestextra.append(random.uniform(1.5,2.0))
    xtest = np.array(sorted(xtest)).reshape(len(xtest),1)
    xtestextra = np.array(sorted(xtestextra)).reshape(len(xtestextra),1)

    # Access training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Get medium-fidelity data
    Xmed, ymed = np.array(data[1]),np.array(data[4])
    Xmed, ymed = Xmed.reshape(len(Xmed),1), ymed.reshape(len(ymed),1)
    ymed = np.ravel(ymed)

    # Get high-fidelity data
    Xhgh, yhgh = np.array(data[2]),np.array(data[5])
    Xhgh, yhgh = Xhgh.reshape(len(Xhgh),1), yhgh.reshape(len(yhgh),1)
    yhgh = np.ravel(yhgh)

    # Create low and high models, 3 hidden layers
    mlow = model(X,y,(20,15,10))
    mmed = model(X,y,(20,15,10))
    mhigh = model(X,y,(20,15,10))

    # Partial fit model with medium-fidelity then high-fidelity data over iterations
    iterations = 10
    for i in range(iterations): mhigh.partial_fit(Xmed,ymed), mmed.partial_fit(Xmed,ymed)
    for i in range(iterations): mhigh.partial_fit(Xhgh,yhgh)

    # Find errors against test data
    errors = [MSE(mlow,xtest),MSE(mmed,xtest),MSE(mhigh,xtest)]
    # print(pd.DataFrame(errors))

    # # Find errors when predicting out of range 
    # # (terrible values, shows it hasn't really learnt the function, only how to fit the data)
    # print(MSE(mhigh,xtestextra))

    # Use more values to plot real and predicted functions
    xfull = np.linspace(0,1.5,1000)
    xfull = xfull.reshape(len(xfull),1)
    yrealfull = [highfid(x) for x in xfull]
    ystarlowfull = MLPRegressor.predict(mlow,xfull)
    ystarmedfull = MLPRegressor.predict(mmed,xfull)
    ystarhighfull = MLPRegressor.predict(mhigh,xfull)

    # Define the MSE of prediction
    M = [float((y-ystarhighfull[x])**2) for x,y in enumerate(yrealfull)]
    m = np.sqrt(np.mean(M))
    dyfitfull = [np.sqrt(x) for x in M]

    # Create the MAE envelope evaluated at each point
    # Indicates where the model struggles most
    ydynplus = [float(x+y) for x,y in zip(yrealfull,dyfitfull)]
    ydynmin = [float(x-y) for x,y in zip(yrealfull,dyfitfull)]

    # # Plot training data
    # plt.scatter(X,y,s=10,c='r')
    # plt.scatter(Xmed,ymed,s=10,c='orange')
    # plt.scatter(Xhgh,yhgh,c='g',s=10)

    # # Plot real and predicted function
    # plt.plot(xfull,yrealfull,c='g')
    # plt.plot(xfull,ystarlowfull,c='r')
    # plt.plot(xfull,ystarmedfull,c='orange')
    # plt.plot(xfull,ystarhighfull,c='b')

    # # Plot uncertainty envelope of predicted function
    # plt.fill_between(np.ravel(xfull), ydynplus, ydynmin,
    #             color='gray', alpha=0.2)

    # # Labels & extras
    # plt.xlabel('X'),plt.ylabel('y'),plt.grid(True)
    # plt.legend(['Low-fidelity training data','Medium-fidelity training data','High-fidelity training data',
    #             'True distribution','Low-fidelity model output','Medium-fidelity model output','High-fidelity output',
    #             '95% error margin'])
    
    # # plt.show()

    return ystarhighfull



def BFPM():
    # Generate tests samples between [0,1.5] and ones between [1.5,2.0]
    samples,xtest,xtestextra = 20,[],[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5)), xtestextra.append(random.uniform(1.5,2.0))
    xtest = np.array(sorted(xtest)).reshape(len(xtest),1)
    xtestextra = np.array(sorted(xtestextra)).reshape(len(xtestextra),1)

    # Generate real y values for xtest
    yreal = []
    for x in xtest: yreal.append(highfid(x))
    yreal = np.array(yreal).reshape(len(yreal),1)

    # Access low-fidelity training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Get medium-fidelity data
    Xmed, ymed = np.array(data[1]),np.array(data[4])
    Xmed, ymed = Xmed.reshape(len(Xmed),1), ymed.reshape(len(ymed),1)
    ymed = np.ravel(ymed)

    # Create low-fidelity model
    mlow = model(X,y,(20,15,10))

    # Pass medium-fidelity inputs through low-fidelity model
    yl = MLPRegressor.predict(mlow,Xmed)
    yltest = MLPRegressor.predict(mlow,xtest)

    # Stack predictions with rest of training data
    Xh = np.hstack((Xmed,yl.reshape(len(yl),1)))
    Xhtest = np.hstack((xtest,yltest.reshape(len(yltest),1)))

    # Define new model with stacked input
    mmed = model(Xh,ymed,(20,15,10))
    
    # Predict outputs of test data (requires passing it through low-fidelity network first)
    ymtest = MLPRegressor.predict(mmed,Xhtest)

    # Define errors
    MSE = mean_squared_error(ymtest,yreal)
    R2 = mmed.score(Xhtest,yreal)
    # print(round(MSE,5),round(R2,5))

    # Use more values to plot real and predicted functions
    xfull = np.linspace(0,1.5,1000)
    xfull = xfull.reshape(len(xfull),1)
    yrealfull = [highfid(x) for x in xfull]
    ymedfull = [lowfid(0.8,2,-1,x) for x in xfull]

    # Pass xfull through mlow then stack to inputs of med-fid. network
    ylfull = MLPRegressor.predict(mlow,xfull)
    Xhfull = np.hstack((xfull,ylfull.reshape(len(ylfull),1)))
    ystarmedfull = MLPRegressor.predict(mmed,Xhfull)

    # Define the MSE of prediction
    M = [float((y-ystarmedfull[x])**2) for x,y in enumerate(yrealfull)]
    dyfitfull = [2 * np.sqrt(x) for x in M]

    # Create the 2 S.D.s envelope (95% error margin)
    ydynplus = [float(x+y) for x,y in zip(yrealfull,dyfitfull)]
    ydynmin = [float(x-y) for x,y in zip(yrealfull,dyfitfull)]

    # # Plot training data
    # plt.scatter(X,y,s=10,c='r')
    # plt.scatter(Xmed,ymed,s=10,c='orange')

    # # Plot real and predicted function
    # plt.plot(xfull,yrealfull,c='g')
    # plt.plot(xfull,ylfull,c='r')
    # plt.plot(xfull,ystarmedfull,c='orange')

    # # Plot uncertainty envelope of predicted function
    # plt.fill_between(np.ravel(xfull), ydynplus, ydynmin,
    #             color='gray', alpha=0.2)
    
    # # Labels & extras
    # plt.xlabel('X'),plt.ylabel('y'),plt.grid(True)
    # plt.legend(['Low-fidelity training data','Medium-fidelity training data',
    #             'True distribution','Low-fidelity model output','Medium-fidelity model output',
    #             '95% error margin'])
    # plt.show()

def MFPM():
    # Generate tests samples between [0,1.5] and ones between [1.5,2.0]
    samples,xtest,xtestextra = 20,[],[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5)), xtestextra.append(random.uniform(1.5,2.0))
    xtest = np.array(sorted(xtest)).reshape(len(xtest),1)
    xtestextra = np.array(sorted(xtestextra)).reshape(len(xtestextra),1)

    # Generate real y values for xtest
    yreal = []
    for x in xtest: yreal.append(highfid(x))
    yreal = np.array(yreal).reshape(len(yreal),1)

    # Access low-fidelity training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Get medium-fidelity data
    Xmed, ymed = np.array(data[1]),np.array(data[4])
    Xmed, ymed = Xmed.reshape(len(Xmed),1), ymed.reshape(len(ymed),1)
    ymed = np.ravel(ymed)

    # Get high-fidelity data
    Xhigh, yhgh = np.array(data[2]),np.array(data[5])
    Xhigh, yhgh = Xhigh.reshape(len(Xhigh),1), yhgh.reshape(len(yhgh),1)
    yhgh = np.ravel(yhgh)  

    # Create low-fidelity model
    mlow = model(X,y,(20,15,10))  

    # Pass medium-fidelity inputs through low-fidelity model
    yl = MLPRegressor.predict(mlow,Xmed)
    ylh = MLPRegressor.predict(mlow,Xhigh)
    yltest = MLPRegressor.predict(mlow,xtest)

    # Stack predictions with rest of training data
    Xm = np.hstack((Xmed,yl.reshape(len(yl),1)))
    Xmtest = np.hstack((xtest,yltest.reshape(len(yltest),1)))

    # Define new model with stacked input
    mmed = model(Xm,ymed,(10,10))

    # Pass high-fidelity inputs through medium-fidelity model
    Xhgh = np.hstack((Xhigh,ylh.reshape(len(ylh),1)))
    ymh = MLPRegressor.predict(mmed,Xhgh)
    ymtest = MLPRegressor.predict(mmed,Xmtest)

    # Stack predictions with rest of training data
    Xh = np.hstack((Xhgh,ymh.reshape(len(ymh),1)))
    Xhtest = np.hstack((Xmtest,ymtest.reshape(len(yltest),1)))

    # Define high-fidelity model with stacked input
    # mhigh = model(Xh,yhgh,(2))

    # Try altering hyperparameters other than network size
    mhigh = MLPRegressor(
                    random_state=1,          # manipulate results by picking a random state that shows an improvement from both to multi
                    hidden_layer_sizes=10,           #a robust choice of network
                    activation='relu',          #consider tanh and sigmoid functions too, for non-linearity in networks with 2+ hidden layers
                    solver = 'adam',            
                    batch_size='auto',
                    early_stopping=False,
                    learning_rate='adaptive',
                    tol=1e-4,
                    max_iter=10000).fit(Xh,yhgh)

    # Predict outputs of test data (requires passing first through low- and med- fidelity networks)
    yhghtest = MLPRegressor.predict(mhigh,Xhtest)

    # Define errors
    MSE = mean_squared_error(yhghtest,yreal)
    R2 = mhigh.score(Xhtest,yreal)
    print(round(MSE,5),round(R2,5))

    # Use more values to plot real and predicted functions
    xfull = np.linspace(0,1.5,1000)
    xfull = xfull.reshape(len(xfull),1)
    yrealfull = [highfid(x) for x in xfull]

    # Pass xfull through mlow then stack to inputs of med-fid. network
    ylfull = MLPRegressor.predict(mlow,xfull)
    Xmfull = np.hstack((xfull,ylfull.reshape(len(ylfull),1)))
    ystarmedfull = MLPRegressor.predict(mmed,Xmfull)

    # Pass xfull through mmed then stack to inputs of high-fid. network
    Xhfull = np.hstack((Xmfull,ystarmedfull.reshape(len(ystarmedfull),1)))
    ystarhighfull = MLPRegressor.predict(mhigh,Xhfull)

    # Plot training data
    plt.scatter(X,y,s=10,c='r')
    plt.scatter(Xmed,ymed,s=10,c='orange')
    plt.scatter(Xhigh,yhgh,c='g',s=10)

    # Plot real and predicted function
    plt.plot(xfull,yrealfull,c='g')
    plt.plot(xfull,ystarmedfull,c='orange')
    plt.plot(xfull,ystarhighfull,c='b')

    # Labels & extras
    plt.xlabel('X'),plt.ylabel('y'),plt.grid(True)
    plt.legend(['Low-fidelity training data','Medium-fidelity training data','High-fidelity training data',
                'True distribution','Medium-fidelity model output','High-fidelity model output'])
    
    plt.show()   

#MFPM()

# MFPM testing
#          MSE       R2
#Low  76.94256  0.46368
#Med  39.39119  0.69898
#Hgh  12.13988  0.90723 (n1 = (20,10,5), n2 = (10,10), n3 = (10), activation = 'relu' to exploit linear relationship, solver = 'adam')
#Hgh2 10.62131  0.91883 (only using the most recent prediction, hence not stacking all prior predictions)

# MFWL testing
#          MSE       R2
#Low  76.94256  0.46368
#Med  57.92904  0.59622
#Hgh  43.61856  0.69596

# Important takeaway from MFPM is that each network's hyperparameters must be optimised
    
def MFPM2():
    # Generate tests samples between [0,1.5] and ones between [1.5,2.0]
    samples,xtest,xtestextra = 20,[],[]
    for sample in range(samples): xtest.append(random.uniform(0,1.5)), xtestextra.append(random.uniform(1.5,2.0))
    xtest = np.array(sorted(xtest)).reshape(len(xtest),1)
    xtestextra = np.array(sorted(xtestextra)).reshape(len(xtestextra),1)

    # Generate real y values for xtest
    yreal = []
    for x in xtest: yreal.append(highfid(x))
    yreal = np.array(yreal).reshape(len(yreal),1)

    # Access low-fidelity training data
    X,y = np.array(data[0]),np.array(data[3])
    X,y = X.reshape(len(X),1),y.reshape(len(y),1)
    y=np.ravel(y)

    # Get medium-fidelity data
    Xmed, ymed = np.array(data[1]),np.array(data[4])
    Xmed, ymed = Xmed.reshape(len(Xmed),1), ymed.reshape(len(ymed),1)
    ymed = np.ravel(ymed)

    # Get high-fidelity data
    Xhigh, yhgh = np.array(data[2]),np.array(data[5])
    Xhigh, yhgh = Xhigh.reshape(len(Xhigh),1), yhgh.reshape(len(yhgh),1)
    yhgh = np.ravel(yhgh)  

    # Create low-fidelity model
    mlow = model(X,y,(20,15,10))  

    # Pass medium-fidelity inputs through low-fidelity model
    yl = MLPRegressor.predict(mlow,Xmed)
    ylh = MLPRegressor.predict(mlow,Xhigh)
    yltest = MLPRegressor.predict(mlow,xtest)

    # Stack predictions with rest of training data
    Xm = np.hstack((Xmed,yl.reshape(len(yl),1)))
    Xmtest = np.hstack((xtest,yltest.reshape(len(yltest),1)))

    # Define new model with stacked input
    mmed = model(Xm,ymed,(10,10))

    # Pass high-fidelity inputs through medium-fidelity model
    Xhgh = np.hstack((Xhigh,ylh.reshape(len(ylh),1)))
    ymh = MLPRegressor.predict(mmed,Xhgh)
    ymtest = MLPRegressor.predict(mmed,Xmtest)

    # Stack predictions with rest of training data
    Xh = np.hstack((Xhigh,ymh.reshape(len(ymh),1)))
    Xhtest = np.hstack((xtest,ymtest.reshape(len(yltest),1)))

    # Define high-fidelity model with stacked input
    # mhigh = model(Xh,yhgh,(2))

    # Try altering hyperparameters other than network size
    mhigh = MLPRegressor(
                    random_state=1,          # manipulate results by picking a random state that shows an improvement from both to multi
                    hidden_layer_sizes=10,           #a robust choice of network
                    activation='relu',          #consider tanh and sigmoid functions too, for non-linearity in networks with 2+ hidden layers
                    solver = 'adam',            
                    batch_size='auto',
                    early_stopping=False,
                    learning_rate='adaptive',
                    tol=1e-4,
                    max_iter=10000).fit(Xh,yhgh)

    # Predict outputs of test data (requires passing first through low- and med- fidelity networks)
    yhghtest = MLPRegressor.predict(mhigh,Xhtest)

    # Define errors
    MSE = mean_squared_error(yhghtest,yreal)
    R2 = mhigh.score(Xhtest,yreal)
    print(round(MSE,5),round(R2,5))

    # Use more values to plot real and predicted functions
    xfull = np.linspace(0,1.5,1000)
    xfull = xfull.reshape(len(xfull),1)
    yrealfull = [highfid(x) for x in xfull]

    # Pass xfull through mlow then stack to inputs of med-fid. network
    ylfull = MLPRegressor.predict(mlow,xfull)
    Xmfull = np.hstack((xfull,ylfull.reshape(len(ylfull),1)))
    ystarmedfull = MLPRegressor.predict(mmed,Xmfull)

    # Pass xfull through mmed then stack to inputs of high-fid. network
    Xhfull = np.hstack((xfull,ystarmedfull.reshape(len(ystarmedfull),1)))
    ystarhighfull = MLPRegressor.predict(mhigh,Xhfull)

    # # Plot training data
    # plt.scatter(X,y,s=10,c='r')
    # plt.scatter(Xmed,ymed,s=10,c='orange')
    # plt.scatter(Xhigh,yhgh,c='g',s=10)

    # # Plot real and predicted function
    # plt.plot(xfull,yrealfull,c='g')
    # plt.plot(xfull,ystarmedfull,c='orange')
    # plt.plot(xfull,ystarhighfull,c='b')

    # # Labels & extras
    # plt.xlabel('X'),plt.ylabel('y'),plt.grid(True)
    # plt.legend(['Low-fidelity training data','Medium-fidelity training data','High-fidelity training data',
    #             'True distribution','Medium-fidelity model output','High-fidelity model output'])
    
    # plt.show()

    return [xfull,ylfull,yrealfull,ystarhighfull]


mfwl = MFWL()
[x,ylow,y,mfpm] = MFPM2()
pXlow,pXmed,pXhigh,pylow,pymed,pyhigh = gen_data()

def fig1():

    plt.scatter(pXlow,pylow,s=10,c='pink',alpha=0.5)
    plt.scatter(pXmed,pymed,s=10,c='orange',alpha=0.5)
    plt.scatter(pXhigh,pyhigh,c='g',s=10,alpha=0.5)

    plt.plot(x,y,c='r',alpha=0.8)
    plt.plot(x,ylow,c='pink',alpha=0.7)
    plt.plot(x,mfwl,'--',color=(0.1,0.1,0.8),alpha=0.7)
    plt.plot(x,mfpm,'--',color=(0.1,0.5,0.8),alpha=0.7)

    plt.xlabel('X'),plt.ylabel('y')
    plt.grid(visible=True, which='major', axis='both')
    plt.xlim(-0.1,1.6)  
    plt.legend([r'$D_{L}$',r'$D_{M}$',r'$D_{H}$',r'$y_{H}$',r'$M_{L}$',r'$M_{H,WL}$',r'$M_{H,PM}$'])
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()      

# fig1()