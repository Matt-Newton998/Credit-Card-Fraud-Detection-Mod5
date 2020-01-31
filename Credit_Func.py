### Credit Card Fraud Detection - Mod05_Project - Functions File ###



# Libraries Imports:
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve


# Data Exploration Functions:

def plot_all(df, num_of_plots_per_row = 6, target = None):
    
    import matplotlib.pyplot as plt
    from textwrap import wrap
    
    num_of_cols = df.shape[1]
    num_of_sets = num_of_cols // (num_of_plots_per_row + 1)
    k = 0
    for r, s in enumerate(range(num_of_sets)):
        color = 'blue' if (r % 2) == 0 else 'red'
        x1 = k
        x2 = k + num_of_plots_per_row
        k = k + num_of_plots_per_row + 1
        r = r * 2
        dfplot = df.iloc[:, x1:x2]
        cols_to_plot = list(dfplot.columns)

        fig, axes = plt.subplots(nrows=2, ncols=len(cols_to_plot), figsize=(18,4))
        for n, xcol in enumerate(cols_to_plot):
            axes[0,n].set_title("\n".join(wrap(xcol, 25)), fontsize=9)
            axes[0,n].scatter(dfplot[xcol], df[target], color=color, s=2)
            axes[0,n].set_ylabel(target, fontsize=8)
            axes[1,n].hist(dfplot[xcol], color=color)       
        plt.show()

        
def show_corr_to_target(corr, target):
    df_target = corr[target]
    var_corrs = []
    i=0
    for var in df_target:
        if var == True:
            var_corrs.append(df_target.index[i])
        i+=1
    var_corrs.remove(target)
    return var_corrs
        
### Matt Model Reporting Functions:

# Get a Dataframe from the GridSearchCV Results
def GS_Output_DataFrame(gs):
    opt = pd.DataFrame(gs.cv_results_)
    opt.set_index(['rank_test_score'])
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    return opt
    
# Print Accuracy and heatmap
def acc(y_val,prediction):
    cm = confusion_matrix(y_val, prediction)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print ('Recall:', recall)
    print ('Precision:', precision)
    print ('\n clasification report:\n', classification_report(y_val,prediction))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, prediction))
    print('\n Accuracy Percentage  is : {}%'.format(accuracy_score(y_val,prediction) * 100))
    ax = sns.heatmap([precision,recall],linewidths= 0.5,cmap='PuRd', annot=True)    
    
    
    
### Dan Model Reporting Functions:


def scores(model,X_train,X_val,y_train,y_val):
    
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train,train_prob)
    val = roc_auc_score(y_val,val_prob)
    print('train:',round(train,2),'test:',round(val,2))
    
def annot(fpr,tpr,thr):
    k=0
    for i,j in zip(fpr,tpr):
        if k %50 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1
        

def roc_plot(model,X_train,y_train,X_val,y_val):
    train_prob = model.predict(X_train)
    val_prob = model.predict(X_val)
    plt.figure(figsize=(7,7))
    for data in [[y_train, train_prob],[y_val, val_prob]]:
        fpr, tpr, threshold = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    plt.legend(['train','val'])
    plt.show()
    
def opt_plots(opt_model):
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_test_score')*100)
    plt.title('ROC_AUC - Validation')
#     return opt