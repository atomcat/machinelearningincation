
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

class SMOModel:
     """
     Container object for the model used for sequential minimal optimization
     """
    def __init__(self, X, y, C, kernel, alphas, b, errors):
        self.X = X
        self.y = y
        self.C = C
        self.kernel = kernel
        self.alphas = alphas
        self.b = b
        self.errors = errors    #error cache
        self._obj = []          #record of objective function value
        self.m = len(self.X)    #store size of training set
    
    def linear_kernel(x, y, b=1):
        """
        returns the linear combiantion of array `x` and `y` with the optional bias term `b`
        """
        return x @ y.T + b  #note the @ operator for matrix multiplication
    
    def gaussian_kernel(x, y, sigma=1):
        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(-np.linalog.norm(x - y) / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
            result = np.exp(-np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(-np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
        return result
    def objective_function(alphas, target, kernel, X_train):
        """
        Arguments:
            alphas -- vector of Lagrange multipliers
            target -- vector of class labels(-1, or 1) for training data
            kernel -- kernel function
            X_train -- training data for model
        Return:
            SVM objective function base in the input model define by upper arguments
        """
        return np.sum(alpha) - 0.5 * np.sum(target * target * kernel(X_train, X_train) * alphas * alphas)

    def desicion_function(alphas, target, kernel, X_train, x_test, b):
        """Applies the SVM decision function to the input feature vector in x_test"""
        result = (alphas * target) @ kernel(X_train) - b
    
    def plot_decison_boundary(model, ax, resolution=100, colors=('b', 'k', 'r')):
        """
        Plots the model's decision boundary on the input axes object.
        Range of decison boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grad`, `ax)
        """
        #Generate coordinate grid of shape[resolution x resolution]
        #and evaluate the model over the entire space
        xrange = np.linspace(model.X[:,0].min(), model.X[:, 0].max(), resolution)
        yrange = np.linspace(model.X[:,1].min(), model.X[:, 1].max(), resolution)
        grid = [[decision_function(model.alphas, model.y, 
                                   model.kernel, model.X,
                                   np.array([xr, yr]), model.b) for yr in yrange] for xr in xrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))
        #Plot decision contours using grid and make a scatter plot of training data
        ax.contour(xrang, yrang, grid, (-1, 0, 1), linewidths=(1, 1, 1),
                   linestyles=['--', '-', '--'], colors=colors)
        
        #plot support vectors (non-zero alphas) as circled pioints(linewidth > 0)
        ax.scatter(model.X[:, 0], model.X[:1], c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.5)
        mask = model.alphas != 0.0
        ax.scatter(model.X[:,0][mask], model.X[:,1][mask], c=model.y[mask], cmap=plt.cm.vridis)
        
        return grid, ax
    def take_step(i, j, model):
        #skip if chosen the same two alphas
        if i == j:
            reutnr 0, model
        
        alpha1 = model.alpha[i]
        alpha2 = model.alpha[j]
        y1 = model.y[i]
        y2 = model.y[j]
        E1 = model.errors[i]
        E2 = model.errors[j]
        s = y1 * y2
        
        if (y1 != y2):  #same sign
            L = max(0, alpah2 - alpha1)
            H = min(model.C, model.C + alpha2 - alpha1)
        elif (y1 == y2):
            L = max(0, alpha2 + alpah1 - model.C)
            H = min(model.C, alpha2 + alpha1)
        if (L == H):
            return 0, model
        
        #compute kernel and 2nd derivate eta
        k11 = model.kernel(model.X[i], model.X[i])
        k12 = model.kernel(model.X[i], model.X[j])
        k22 = model.kernel(model.X[j], model.X[j])
        eta = 2 * k12 - k11 - k22
        #compute new alpah2 (a2) if eta is negtive
        if (eta < 0):
            a2 = alpha2 - y2 * (E1 - E2) / eta
            #clip a2 base on bounds L and H
            if L < a2 < H:
                a2 = a2
            elif (a2 <= L):
                a2 = L
            elif (a2 >= H):
                a2 = H
        #if eta is non-negtive, move new a2 to bound with greater objective function value
        else:
            alphas_adj = model.alphas.copy()
            alphas_adj[j] = L
            #objective function output with a2 = L
            Lobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
            alphas_adj[j] = H
            Hobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
            if Lobj > (Hobj + eps):
                a2 = L
            elif Lobj < (Hobj - eps):
                a2 = H
            else:
                a2 = alpha2
        #push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (model.C - le-8):
            a2 = model.C
        
        #if examples can't be optimized within epsilon, skip the pair
        if (np.abs(a2 - alpha2) < eps * (a2 + alpha2 + eps)):
            return 0, model
        
        #calculate new alpha 1(a1)
        a1 = alpha1 + s * (alpha2 - a2)
        
        #update threshold b to reflect newly calclated alphas, calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12  + model.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22  + model.b
        
        #set new threshold based on if a1 or a2 is bound by L and/ or H
        if 0 < a1 < C:
            b_new = b1
        elif 0 < a2 < C:
            b_new = b2
        else:
            b_new = (b1 + b2) * 0.5
        
        #update model object with new alphas and threshod
        model.alphas[i] = a1
        model.alphas[j] = a2
        
        #update error cache
        #error cache for optimized alphas is set to 0 if they'er unbound
        for index, alph in zip([i, j], [a1, a2]):
            if 0.0 < alph < model.C:
                model.errors[index] = 0.0
        
        # set non-optimized errors based on equation 12.11 in Patt's book
        non_opt = [n for n in range(model.m) if (n != i and n ！= j)]
        model.erros[non_opt] = model.erros[non_opt] +                                y1 * (a1 - alpha1) * model.kernel(model.X[i], model.X[non_opt]) +                                y2 * (a2 - alpha2) * model.kernel(model.X[j], model.X[non_opt]) + model.b - b_new
        model.b = b_new
        return 1, model
    
    def examine_example(i2, model):
        y2 = model.y[x2]
        alpha2 = model.alphas[i2]
        E2 = model.errors[i2]
        r2 = E2 * y2
        
        #proceed if error is within specified tolerance (tol)
        if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alpha2 > 0)):
            #use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.erros)
            elif model.erros[i2] <= 0:
                i1 = np.argmax(model.errors)
            
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
            
            #loop through non-zero and non-C alphas, starting at a random point
            for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                              np.random.choice(np.arange(model.m))):
                step_result, model = take_step(i1, i2, model)
                if step_result:
                    return 1, model
            #loop through all alphas, starting at a random point
            for i1 in np.roll(np.arange(model.m),np.random.choice(np.arange(model.m))):
                step_result, model = take_step(i1, i2, model)
                if step_result:
                    return 1, model
        return 0, model

    def train(model):
        numChanged = 0
        examineAll = 1
        while (numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                for i in range(model.alphas.shape[0]):
                    examine_result, model = examine_example(i, model)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
                        model._obj.append(obj_result)
            else:
                for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                    examine_result, model = examine_example(i, model)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
                        model._obj.append(obj_result)
            if examineAll == 1:
                examineAll = 0
            elif examineAll == 0:
                examineAll = 1
        return model
            

