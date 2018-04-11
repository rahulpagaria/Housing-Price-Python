
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd 
import os
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D


# In[61]:


os.chdir("/Users/rahulpagaria/Downloads/")

hp = pd.read_csv('hpdemo.csv', dtype=float) # creating dataframe containing 1405 rows of house price data of London in 1990

print (hp.isnull().values.any())

import matplotlib.pyplot as plt

#%matplotlib inline  
#plt.boxplot(hp.fl_area)


# In[62]:


minimum_price = np.min(hp.price)
maximum_price = np.max(hp.price)
mean_price = np.mean(hp.price)
median_price = np.median(hp.price)
std_price = np.std(hp.price)
first_quartile = np.percentile(hp.price, 25)
third_quartile = np.percentile(hp.price, 75)
inter_quartile = third_quartile - first_quartile

# Show the calculated statistics
print ("Statistics for London housing dataset:\n")
print ("Minimum price: ${:,.2f}".format(minimum_price))
print ("Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))
print ("First quartile of prices: ${:,.2f}".format(first_quartile))
print ("Second quartile of prices: ${:,.2f}".format(third_quartile))
print ("Interquartile (IQR) of prices: ${:,.2f}".format(inter_quartile))


# In[63]:


from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
x_scaler.fit(hp[['east','north','fl_area']])
X = x_scaler.transform(hp[['east','north','fl_area']])
print(X[:5,:]) # to print first 5 rows and all columns (easting, northing and fl_Area)


# In[64]:


from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor as NN 

pipe = Pipeline([('zscores', StandardScaler()),('NNreg', NN(n_neighbors=6, weights='uniform', p=2))])


# In[65]:


print (pipe)
price = hp['price']/1000.00
pipe.fit(hp[['east','north','fl_area']],price)
print(pipe.predict([[523800.0,179750.0,55.0]]))


# In[66]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
mae = make_scorer(mean_absolute_error, greater_is_better=False)

pipe = Pipeline([('zscores',StandardScaler()),('NNreg',NN())])
opt_nn2 = GridSearchCV(
                        estimator = pipe,
                        scoring = mae,param_grid = {
                                                   'NNreg__n_neighbors':range(1,35),
                                                   'NNreg__weights':['uniform','distance'],
                                                   'NNreg__p':[1,2]
                                                   }
                      )
opt_nn2.fit(hp[['east','north','fl_area']],price)
print(opt_nn2.predict([[523800.0, 179750.0, 55.0]]))


# In[67]:


east_mesh, north_mesh = np.meshgrid(np.linspace(505000, 555800, 100),np.linspace(158400, 199900, 100))
fl_mesh = np.zeros_like(east_mesh)
fl_mesh[:,:] = np.mean(hp['fl_area'])
print(east_mesh.shape)
print(north_mesh.shape)


# In[68]:


grid_predictor_vars = np.array([east_mesh.ravel(),north_mesh.ravel(), fl_mesh.ravel()]).T
hp_pred = opt_nn2.predict(grid_predictor_vars)
hp_mesh = hp_pred.reshape(east_mesh.shape)


# In[69]:


def print_summary2(opt_pipe_object):
    params = opt_pipe_object.best_estimator_.get_params()
    score = - opt_pipe_object.best_score_
    print ("Nearest neighbours: %8d" % params['NNreg__n_neighbors'])
    print ("Minkowski p : %8d" % params['NNreg__p'])
    print ("Weighting : %8s" % params['NNreg__weights'])
    print ("MAE Score : %8.2f" % score)
    return

print("Summary using pipeline:")
print_summary2(opt_nn2)

#print (hp[])
print(hp.loc[hp['east'] == 523800.0])
print(opt_nn2.predict([[523800.0, 179700.0, 40.0]]))



# In[70]:


fig = pl.figure()
ax = Axes3D(fig)
# Plot the surface.# Add a color bar which maps values to colors.
ax.plot_surface(east_mesh, north_mesh, hp_mesh, rstride=1,cstride=1, cmap='YlOrBr', lw=0.01)
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
ax.set_zlabel('Price at Mean Floor Area')
pl.show()


# In[71]:


# Function surf3d to plot the predicted values
def surf3d(pipe_model,fl_area,cscheme):  # Input args pipe object, fl_area integer & color scheme string
    east_mesh, north_mesh = np.meshgrid(
    np.linspace(505000,555800,100),
    np.linspace(158400,199900,100)) # Create a structural background grid
    fl_mesh = np.zeros_like(east_mesh) 
    fl_mesh[:,:] = fl_area
    grid_predictor_vars = np.array([east_mesh.ravel(),
    north_mesh.ravel(),fl_mesh.ravel()]).T        # Convert to single array using ravel & transpose
    hp_pred = pipe_model.predict(grid_predictor_vars)
    hp_mesh = hp_pred.reshape(east_mesh.shape)
    fig = pl.figure()
    ax = Axes3D(fig)
    ax.plot_surface(east_mesh, north_mesh, hp_mesh, # Axes define
    rstride=1, cstride=1, cmap=cscheme,lw=0.01)
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    zl = 'Price at Mean Floor Area',fl_area
    ax.set_zlabel(zl)
    return


# In[72]:


surf3d(opt_nn2, 75.0,"CMRmap_r")
pl.show()


# In[73]:


surf3d(opt_nn2, 125.0,"Greens")
pl.show()


# In[74]:


surf3d(opt_nn2, round(np.mean(hp.fl_area),2),"Spectral")
pl.show()


# In[75]:


from sklearn.decomposition import PCA

pipe1 = Pipeline([('zscores',StandardScaler()),('prcomp',PCA()),('NNreg',NN())])

opt_nn3 = GridSearchCV(
                        estimator = pipe1,
                        scoring = mae,
                        param_grid = {
                                     'NNreg__n_neighbors':range(1,35),
                                     'NNreg__weights':['uniform','distance'],
                                     'NNreg__p':[1,2],
                                     'prcomp__n_components':[1,2,3]
                                     }
                       )

opt_nn3.fit(hp[['east','north','fl_area']],price)

print(opt_nn3.best_estimator_.get_params()['prcomp__n_components'])
print(opt_nn3.best_score_)

