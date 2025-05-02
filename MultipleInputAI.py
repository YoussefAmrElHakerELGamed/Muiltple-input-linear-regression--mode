import numpy as np                                                                          #numpy for faster arrays
import pandas as pd                                                                         #pandas for the data
import matplotlib.pyplot as plt                                                             #matplotlib for visualization
from mpl_toolkits.mplot3d import Axes3D                                                     #axes3D for 3d plots
import scipy.stats as stats                                                                 #scipy for the anova
from sklearn.linear_model import LinearRegression as LR                                     #sklearn linear model for the linear regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


class Multiple_input_linear_reg_model:

    def __init__(self, data: pd.DataFrame, x :list[object], y: str, test_size, random_state):                        #class constractor
        self.model = LR()                                                                   #init the model
        self.Xdata :pd.DataFrame = data                                                     #loading the dataframe 
        self.test_size = test_size                                                          #test data size
        self.random_state = random_state                                                    #how much random noise to add to the data

        self.x :pd.DataFrame = data[x]                                                      #selecting the inputs
        self.x_train = None                                                                 #model train x
        self.x_test = None                                                                  #model test x
        self.y :pd.Series = data[y]                                                         #selecting the y
        self.y_train = None                                                                 #model train y
        self.y_test = None                                                                  #model test y
        self.y_pred = None                                                                  #model y predict (both train and test)
        self.y_train_pred = None                                                            #model y predict assigned after training
        self.y_test_pred  = None                                                            #model y predict assigned after testing
        self.n = None                                                                       #number of rows
        self.p = None                                                                       #number of columns
        self.MSE = None                                                                     #mean squares error assigned after training
        self.test_mse = None                                                                #mean squares error assigned after testing
        self.variance_cov_matrix = None                                                     #Variance-Covariance Matrix assigned after training



    def fit(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(                                        #split the data into test and train
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )

        self.n, self.p = self.x_train.shape                                                                             #assigning the n,p from the inputs shape
        self.model.fit(self.x_train, self.y_train)                                                                      #training the model on the training data

        self.y_train_pred = self.model.predict(self.x_train)                                                            #predicting the y


        e = self.y_train - self.y_train_pred                                                                            #the errors left over after prediction
        self.MSE = np.sum(e**2) / (self.n - self.p - 1)                                                                 #MSE whare n−p−1 is the degrees of freedom we lost p+1 from the training (w0 + w1X1 + w2X2 ....)
                                                                                                                        #MSE measures the average squared prediction error 
                                                                                                                        #need it for 1-Hypothesis tests 2-Confidence intervals

        x_with_intercept = np.column_stack((np.ones(self.n), self.x_train))                                             #adding the intercept column
        XTX_inv = np.linalg.inv(x_with_intercept.T @ x_with_intercept)                                                  #XTX (information matrix) tells us how the variable relate
                                                                                                                        #XTX inverse tells us how unsure we are about coefficients (from the training)
        self.variance_cov_matrix = self.MSE * XTX_inv                                                                   #tells us the uncertainty in each W


        self.y_test_pred = self.model.predict(self.x_test)                                                              #predicting y of the test set
        self.test_mse = mean_squared_error(self.y_test, self.y_test_pred)                                               #getting MSE of the model in test phase

        self.y_pred = pd.concat([pd.Series(self.y_test_pred), pd.Series(self.y_train_pred)] ,ignore_index=True)         #adding both the train y predict and the test y predict

        print("Train MSE:", self.MSE)                                                                                   #printing the evalution matrics
        print("Train accurcy: ", f'{round(r2_score(self.y_train, self.y_train_pred), 4) * 100} %')
        print("Test MSE:", self.test_mse)
        print("Test accurcy: ", f'{round(r2_score(self.y_test, self.y_test_pred), 4) * 100} %')
       
       
    def pridect(self, X : pd.DataFrame):
        return self.model.predict(X)                                                        #just applying the learned Ws

    def plot(self):
        if self.x.shape[1] != 2:
            raise ValueError('3D plot only supports exactly 2 features')                    #only works for 2 inputs (we have 3 axis that why it called 3D :) 2 for the inputs and 1 for the predicted) 
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')                                          #add the 3D plot 

        x1 = self.x.iloc[:,0]                                                               #get the first input (first column)
        x2 = self.x.iloc[:,1]                                                               #get the second input

        ax.scatter(x1, x2, self.y, color='orange', label='Actual')                          #draw the real data
        ax.scatter(x1, x2, self.y_pred, color='blue', label='Predicted')                    #draw the predicted data

        ax.set_xlabel(f'{self.x.columns[0]}')                                                                 #adding labels
        ax.set_ylabel(f'{self.x.columns[1]}')
        ax.set_zlabel(f'{self.y.name}')
        plt.legend()                                                                        #add the legend for more inforamtion
        plt.show()                                                                          #show the plot

    def anova(self):
        SST = np.sum((self.y - np.mean(self.y))**2)                                         #Total sum of squares, it's measures the total variability in y (how much the y varies around its mean)
        SSR = np.sum((self.y_pred - np.mean(self.y))**2)                                    #Regression Sum of Squares, it's measures the variability in the predicted y by the model
        SSE = np.sum((self.y - self.y_pred)**2)                                             #Error Sum of Squares, it's measures the leftover variability (the unexplained variability left from the last 2 errors calculations)
                                                                                            #SST = SSR + SSE

        #degrees of freedom
        df_reg = self.p                                                                     #degrees of freedom for regression = number of predictors (Each predictor gets its own W coefficient.)
        df_error = self.n - self.p -1                                                       #degrees of freedom for error = what's left over (we lose 1 degree because we estimate the intercept W0)
                                                                                            #we lose p degrees because we estimate p slopes (w1, w2, ....)
        df_total = self.n -1                                                                #degrees of freedom total = total number of independent pieces of information in y.
                                                                                            #Because to compute variance (like SST), we need to know the mean first. Estimating the mean uses up one degree of freedom.

        MSR = SSR/ df_reg                                                                   #Mean Square Regression
        MSE = SSE/ df_error                                                                 #Mean Square Error

        F_stat = MSR/MSE                                                                    #H₀: All W = 0 (no relationship between predictors and y)
                                                                                            #If F is large and the p-value is small, we reject the null — the model is useful.
        p_value = 1- stats.f.cdf(F_stat, df_reg, df_error)                                  
        return {
            'SSR': SSR,
            'SSE': SSE,
            'SST': SST,
            'F': F_stat,
            'p-value': p_value
        }

    def hypothesis_test(self, alpha = 0.05):
        beta = np.concatenate(([self.model.intercept_], self.model.coef_))                  #combineing the model’s intercept (the constant term) and the model’s slope coefficients (for each input) into one array. is the y-intercept of the fitted regression line.
        se = np.sqrt(np.diag(self.variance_cov_matrix))                                     #Extracting the standard errors (SE) for each coefficient from the variance_cov_matrix where the diagonal numbers are the variances of each coefficient estimate.

        t_state = beta/se                                                                   #Computeing the t-statistics for each coefficient. t-statistics measures how many standard errors away from zero the coefficient is.
        p_value = 2* (1- stats.t.cdf(np.abs(t_state), df=self.n - self.p -1))               #Computeign two-tailed p-values for each t-statistic.
                                                                                            #stats.t.cdf() gives the cumulative probability up to the absolute value of the t-stat.
                                                                                            #Subtracting from 1 gives the upper tail probability.
                                                                                            #Multiplying by 2 covers both tails of the distribution (positive and negative).
        significance = p_value < alpha                                                      #Flags which coefficients are statistically significant at the given alpha level (default 0.05).
                                                                                            #If p_value < alpha, then we reject the null hypothesis that **the coefficient is zero.**
        return {
            'alpha': alpha,
            'coefficients': beta,
            't_statistics': t_state,
            'p_values': p_value,
            'significant': significance
        }

    def interval_estimation(self, alpha=0.05):
        beta = np.concatenate(([self.model.intercept_], self.model.coef_))                  #beta is the vector of all model parameters (intercept + coefficients)
        se = np.sqrt(np.diag(self.variance_cov_matrix))                                     #Calculate Standard Errors (SE). it takes the square roots of the diagonal elements of the variance-covariance matrix, giving the standard errors for each parameter estimate
        t_value = stats.t.ppf(1 - alpha/2, df=self.n - self.p - 1)                          #t_value is the critical value from the t-distribution for the desired confidence level (default 95%, since alpha=0.05), considering the degrees of freedom (df = n - p - 1 like we explained earlier)

        lower_bounds = beta - t_value * se                                                  #For each parameter, it subtracts and adds (t_value * se) to beta to form the confidence intervals.
        upper_bounds = beta + t_value * se

        return np.column_stack((lower_bounds, upper_bounds))