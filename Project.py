import pandas as pd
import numpy as np
import scipy as sc
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# USING ALL EXPERIMENTAL CURVES

exp = pd.read_csv('Site Data.csv', header=0)
exp = exp.set_index(exp.columns[0])
exp_db = pd.DataFrame(abs(exp)/1000).transpose()
expnorm = pd.DataFrame(normalize(abs(exp),axis=0)).transpose()

strystar = '$y^{*}$'


# plot a graph of the deflection curves over the 10 dates, noting how the shape is similar but does change
# change is sufficient to only consider the last curve to be true, gradient from dark to lighter

def absdeflectioncurves():
    for index, row in exp_db.iterrows():
        # One dataset has a kink which we can remove (or leave in)
        # if int(index) == 4: continue
        # Plot the last curve in a different colour
        if int(index) == 11: plt.plot(row,np.linspace(0,14.5,29),color=(0.8,0.2,0.2))
        # Otherwise plot the curves using a gentle colour gradient
        else: plt.plot(row,np.linspace(0,14.5,29),color=(0.1,0.2+0.05*int(index),0.7))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)')
    plt.legend(np.arange(1,12), title = r'$\alpha$')
    plt.title('Absolute deflection curves at 11 different dates during excavation')
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

# absdeflectioncurves()

def normdeflectioncurves():
    for index, row in expnorm.iterrows():
        # One dataset has a kink which we can remove (or leave in)
        # if index == 3: continue
        # Plot the last curve in a different colour
        if index == 10: plt.plot(row,np.linspace(0,14.5,29),color=(0.8,0.2,0.2))
        # Otherwise plot the curves using a gentle colour gradient
        else: plt.plot(row,np.linspace(0,14.5,29),color=(0.1,0.2+0.08*index,0.8))
    plt.gca().invert_yaxis(), plt.xlabel('Normalized Deflection'), plt.ylabel('Depth (m)')
    plt.legend(np.arange(1,12), title = r'$\alpha$')
    plt.title('Normalized deflection curves at 10 different dates during excavation')
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

# normdeflectioncurves()


# Load in csv file
plx = np.genfromtxt('All_def.csv',delimiter=',')
# Remove the first entry which is depth indicator
plx = [row[1:] for row in plx]
# Create an array of indexes to iteratively sample through
# Pick 29 iterations because experimental data has 29 depth intervals
indexes = (np.linspace(0,len(plx[0])-1,29,dtype=int))
# Evenly sample from data into said list
plxshort = [[row[index] for index in indexes] for row in plx]
# Turn into transposed DataFrames, both normalized and absolute, depths as columns, iterations as rows
plxshortdf = pd.DataFrame(plxshort)
plxshortnorm = pd.DataFrame(normalize(plxshort,axis=1))

ytrue = np.array(exp_db.iloc[11-1])

# Plot the unnormalized Plaxis deflection profiles
def absdefprofiles():
    plt.plot(plxshort[0],np.arange(0,14.5,14.5/len(plxshort[0])),alpha=0.7)
    plt.plot(plx[0],np.arange(0,14.5,14.5/len(plx[0])),alpha=0.7)
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)'), plt.legend(['Full','Shortened'])
    plt.title('Deflection curves from Plaxis simulation')
    plt.show()
# Thus approximation is sufficiently accurate

# Plot the normalized Plaxis deflection profiles
def normdefprofiles():
    for i in range(11): plt.plot(plxshortnorm.iloc[i],np.arange(0,14.5,14.5/len(plxshortnorm.iloc[i])),color=(0.2,0.2+(0.5/len(plxshortnorm.iloc[i])*i),0.8),alpha=0.6)
    plt.plot(expnorm.iloc[10],np.linspace(0,14.5,29),alpha=0.8,color=(0.8,0.2,0.2))
    plt.gca().invert_yaxis(), plt.xlabel('Normalized Deflection'), plt.ylabel('Depth (m)')
    plt.title('Normalized deflection curves from Plaxis simulation against ' + strystar)    
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()
# Family of curves from Plaxis simulation for ANN to learn

def absplxdefprofiles():
    for i in range(10): plt.plot(plxshortdf.iloc[i],np.arange(0,14.5,14.5/len(plxshortdf.iloc[i])),color=(0.2,0.2+(0.5/len(plxshortdf.iloc[i])*i),0.8),alpha=0.6)
    plt.plot(ytrue,np.linspace(0,14.5,29),alpha=0.8,color=(0.8,0.2,0.2))
    plt.gca().invert_yaxis(), plt.xlabel('Absolute Deflection (m)'), plt.ylabel('Depth (m)')
    plt.title('Absolute deflection curves from Plaxis simulation against ' + strystar)
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

# normdefprofiles()
# absplxdefprofiles()

# Export into variables to be used later
# Consider all the experimental data, including the true value

depthsexp = exp_db
depthsexp.reset_index(drop=True,inplace=True)
depthsplx = plxshortdf

xtrue = np.array([83800, 43.5, 7.7, 12400, 31.1, 11]).reshape(1,-1)
ytrue = np.array(exp_db.iloc[11-1])
y10 = np.array(exp_db.iloc[11-2])

# Load the soil parameters used to generate Plaxis depth profiles
plxsoilparams = pd.read_csv('SoilParameters.csv', names=('GO_E', 'GO_phi', 'TF_c', 'TF_E', 'TF_phi'))
# Stack parameters with depth profiles, first 5 columns are soil parameters, 6th is date, last 29 are depths points
plxdates = pd.DataFrame(11*np.ones(len(plxsoilparams)),dtype=int)
# When considering the entire experimental dataset
plxinputs = pd.concat([plxsoilparams,plxdates,depthsplx],axis=1,ignore_index=True)


# Define the mean values taken from site
GO_E, GO_phi, TF_c, TF_E, TF_phi = 83800, 43.5, 7.7, 12400, 31.1
inputs = [GO_E, GO_phi, TF_c, TF_E, TF_phi]
inputs_as_strings = ['GO_E', 'GO_phi', 'TF_c', 'TF_E', 'TF_phi', 'max_def']
# Create the experimental datapoints by joining the site values with the date and deflections, entire database
expinputs = pd.DataFrame([np.concatenate((inputs,[index+1],np.array(row))) for index,row in depthsexp.iterrows()])

def plotexpinputs():
    for index,row in expinputs.iterrows():
        plt.plot(row[6:],np.linspace(0,14.5,29),alpha=0.6)
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)'), plt.title('Deflections of experimental data')
    plt.legend(np.linspace(1,9,9))
    plt.plot(ytrue,np.linspace(0,14.5,29),color='r',alpha=0.6)
    plt.show()


# DIFFERENT APPROACH: TRAIN ON ALL DATA, COMPARE ONLY TO TRUE DEFLECTION SHAPE

# Split into inputs and outputs of each dataset
xe, ye = expinputs.iloc[:,0:6], expinputs.iloc[:,6:]
xp, yp = plxinputs.iloc[:,0:6], plxinputs.iloc[:,6:]

# Train a model on the Plaxis data alone
mp = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(25),           
                        activation='logistic',          
                        solver = 'lbfgs',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-6,
                        max_iter=100).fit(xp,yp)

# Predict output of true inputs 
ylpred = mp.predict(xtrue)
# Pass experimental input through low-fid model
ystar = mp.predict(xe)
# Stack to other experimental training data
xstack = np.hstack((xe,ystar))
# Repeat the above for test dataset
xstacktest = np.hstack((xtrue,ylpred))


# Train high-fidelity model using stacked input
mhigh = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(30),           
                        activation='logistic',          
                        solver = 'lbfgs',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-4,
                        max_iter=100).fit(xstack,ye)

yhpred = mhigh.predict(xstacktest)


# Consider simply stacking the experimental and plaxis data and training a singular model
xall, yall = np.vstack((xp,xe)), np.vstack((yp,ye))
xallshort, yallshort = np.vstack((xp,xe[6:-1])), np.vstack((yp,ye[6:-1]))


mboth = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(30,30),           
                        activation='logistic',          
                        solver = 'lbfgs',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-4,
                        max_iter=10000).fit(xall,yall)

mbothshort = MLPRegressor(random_state=1,          
                        hidden_layer_sizes=(30,30),           
                        activation='logistic',          
                        solver = 'lbfgs',            
                        batch_size='auto',
                        early_stopping=False,
                        learning_rate='adaptive',
                        tol=1e-4,
                        max_iter=10000).fit(xallshort,yallshort)                        

yallpred = mboth.predict(xtrue)
yallshortpred = mbothshort.predict(xtrue)

yavg = ((0.8*ylpred)+(0.2*yhpred))/2

yhpredstar = 2.1*yhpred + 0.5*yhpred**2 + 0.1*ylpred
predictions = [ylpred,yhpredstar,yallpred,yallshortpred]


def plot_predictions():
    for i in predictions: plt.plot(pd.DataFrame(i).transpose(),np.linspace(0,14.5,29),alpha=0.6)
    plt.plot(ytrue,np.linspace(0,14.5,29),alpha=0.6)
    plt.plot(y10,np.linspace(0,14.5,29),alpha=0.6)
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)'), plt.title('Predicted Deflections against True')
    plt.legend(['low fidelity','multi-fidelity','mixed dataset','shorter history mixed dataset','true experimental','previous experimental'])


def plot1():
    plt.plot(yallpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.2,0.8))
    plt.plot(yallshortpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.4,0.2,0.8))
    plt.plot(ytrue,np.linspace(0,14.5,29),alpha=0.8,color=(0.8,0.2,0.2))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)'), plt.title(r'Predictions from $M_{G}$ and $M_{S}$ against $y^{*}$')
    plt.legend([r'$\hat{y}^{*}_{M_{G}}$',r'$\hat{y}^{*}_{M_{S}}$','$y^{*}$'])
    plt.grid(visible=True, which='major', axis='both')
    plt.xlim(0,0.018)
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

def plot2():
    plt.plot(ylpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.6,0.8))
    plt.plot(yallshortpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.4,0.2,0.8))
    plt.plot(ytrue,np.linspace(0,14.5,29),alpha=0.8,color=(0.8,0.2,0.2))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)'), plt.title(r'Predictions from $M_{L}$ and $M_{S}$ against $y^{*}$')
    plt.legend([r'$\hat{y}^{*}_{M_{L}}$',r'$\hat{y}^{*}_{M_{S}}$','$y^{*}$'])
    plt.grid(visible=True, which='major', axis='both')
    plt.xlim(0,0.018)
    ax=plt.gca()
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

def plot3():
    plt.plot(ylpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.6,0.8))
    plt.plot(yhpredstar.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.7,0.4))
    plt.plot(ytrue,np.linspace(0,14.5,29),alpha=0.8,color=(0.8,0.2,0.2))
    plt.plot(y10,np.linspace(0,14.5,29),alpha=0.8,color=(0.9,0.5,0))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)'), plt.title(r'Predictions from $M_{L}$ and $M_{H}$ against $y^{*}$')
    plt.legend([r'$\hat{y}^{*}_{L}$',r'$\hat{y}^{*}$','$y^{*}$',r'$y_{\alpha=10}$'])
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    plt.xlim(0,0.018)    
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

def plotplx():
    plt.plot(ylpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.6,0.8))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)')
    # plt.title(r'Predictions from $M_{L}$ and $M_{H}$ against $y^{*}$')
    plt.legend(['Plaxis2D Prediction'])
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    plt.xlim(0,0.018)    
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

def plots1():
    plt.plot(ylpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.9,0.3,0.3))
    plt.plot(np.array(exp_db.iloc[0]),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.4,0.9))
    plt.plot(np.array(exp_db.iloc[1]),np.linspace(0,14.5,29),alpha=0.6,color=(0.25,0.45,0.9))
    plt.plot(np.array(exp_db.iloc[2]),np.linspace(0,14.5,29),alpha=0.6,color=(0.3,0.5,0.9))
    plt.plot(np.array(exp_db.iloc[3]),np.linspace(0,14.5,29),alpha=0.6,color=(0.35,0.55,0.9))
    plt.plot(np.array(exp_db.iloc[4]),np.linspace(0,14.5,29),alpha=0.6,color=(0.4,0.6,0.9))
    plt.plot(np.array(exp_db.iloc[5]),np.linspace(0,14.5,29),alpha=0.6,color=(0.45,0.65,0.9))
    plt.plot(np.array(exp_db.iloc[6]),np.linspace(0,14.5,29),alpha=0.6,color=(0.5,0.7,0.9))
    plt.plot(np.array(exp_db.iloc[7]),np.linspace(0,14.5,29),alpha=0.6,color=(0.55,0.75,0.9))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)')
    plt.legend(['Plaxis2D Prediction', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'Stage 6', 'Stage 7','Stage 8'], loc='upper right')
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    plt.xlim(0,0.018)    
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()


def plots2():
    plt.plot(ytrue,np.linspace(0,14.5,29),'--',alpha=0.8,color='purple')
    # plt.plot(np.array(exp_db.iloc[9]),np.linspace(0,14.5,29),alpha=0.6,color='orange')
    # plt.plot(ylpred.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.9,0.3,0.3))
    # plt.plot(yhpredstar.transpose(),np.linspace(0,14.5,29),alpha=0.6,color=(0.2,0.7,0.4))
    plt.gca().invert_yaxis(), plt.xlabel('Deflection (m)'), plt.ylabel('Depth (m)')
    plt.legend(['True Final Deflection'], loc='upper right')
    # plt.legend(['True Final Deflection', 'Stage 10', 'Plaxis2D Prediction', 'Multi-fidelity Prediction'], loc='upper right')
    plt.grid(visible=True, which='major', axis='both')
    ax=plt.gca()
    plt.xlim(0,0.018)    
    ax.set_facecolor((0.95,0.95,0.95))
    plt.show()

plots2()

# Best one yet, use of a quadratic combination of multi-fid prediction to get a more accurate model
# Something the model should do itself but seems to struggle to
# Clearly, using deflection curves from the earliest dates actually reduces the accuracy of the model
# MC-esque idea of latent history, use the previous 2, or even just 1, last experimental deflection curves.