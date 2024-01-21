import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02

    if isinstance(X,np.ndarray):
        x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
        y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    elif isinstance(X,pd.DataFrame):
        x_min, x_max = X.iloc[:,0].min() - 10*h, X.iloc[:,0].max() + 10*h
        y_min, y_max = X.iloc[:,1].min() - 10*h, X.iloc[:,1].max() + 10*h
    else:
        raise TypeError("incorrect type for X parameter")
        
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    if isinstance(X,np.ndarray): 
        plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');
    else:
        plt.scatter(X.iloc[:,0], X.iloc[:,1], c=Y, cmap=cmap, edgecolors='k');

