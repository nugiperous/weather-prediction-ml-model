# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 20:08:19 2022

@author: agarc
"""

import pandas as pd
from sklearn import tree
import pydotplus 
import collections 
from tqdm import tqdm
from sklearn import metrics

def create_prior_day_data_df(df, feature_names):
    for feature in feature_names:
        for idx in tqdm(df.index):
            if idx == 0:
                df.loc[idx, 'PREV_'+feature] = 0
            elif idx > 0:
                df.loc[idx, 'PREV_'+feature] = df.loc[idx-1, feature] 
    return df

def get_mse(model, x, y):
    model_pred_test = model.predict(x)
    sum_ = 0
    idx_pred = 0
    for idx_y in y.index:
        a = (y.loc[idx_y, 'NEXTDAYPRECIPAMT'] - model_pred_test[idx_pred])**2
        sum_ = sum_ + a
        idx_pred = idx_pred+1
    mse = (1/len(model_pred_test)*sum_)
    return mse

def get_true_positive(decision_tree, x_test, y_test):
    """
    Returns
    -------
    true_positive_value : float64
        Results from the total correctly predicited divided by total predictions.
    matrix_df : DataFrame
        Predicted Labels are Columns and True Labels are rows. 

    """
    test_pred_decision_tree = decision_tree.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, test_pred_decision_tree)
    #turn this into a dataframe
    matrix_df = pd.DataFrame(confusion_matrix)
    test_set_correctly_pred = matrix_df[0][0] + matrix_df[1][1]
    true_positive = test_set_correctly_pred / len(x_test)
    return true_positive, matrix_df

def replace_missing_values_avg(df, column_name, avg_value):
    """
    This function will take in a data frame and replace a missing value with
    the average.
    """
    missing_values_bool = df[column_name].isna()
    for idx in range(len(missing_values_bool)):
        if missing_values_bool[idx] == True:
            df.loc[idx, column_name] = avg_value
            print(f"Value Replaced for {column_name} at {idx}")
        elif missing_values_bool[idx] == False:
            pass

# for a two-class tree, call this function like this: 
# writegraphtofile(clf, ('F', 'T'), dirname+graphfilename) 
def writegraphtofile(clf, feature_labels, classnames, pathname): 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_labels, 
                                    class_names=classnames, 
                                    filled=True, rounded=True, 
                                    special_characters=True) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    colors = ('lightblue', 'green') 
    edges = collections.defaultdict(list) 
    for edge in graph.get_edge_list(): 
        edges[edge.get_source()].append(int(edge.get_destination())) 
    for edge in edges: 
        edges[edge].sort() 
        for i in range(2): 
            dest = graph.get_node(str(edges[edge][i]))[0] 
            dest.set_fillcolor(colors[i])
    graph.write_png(pathname)

class Weather_Data_CSV:
    def __init__(self, csv_path):
        self.data_raw = pd.read_csv(csv_path)
        self.df_prep
        
    def get_unique_column_values(df):
        """
        Identifying Unique Values of each Column in DF
        Output is a Dictionary of each Column
        """
        headers_unique = {}
        for label in tqdm(df.columns):
            headers_unique[label] = df[label].unique()
        #pbar.close()
        return headers_unique

    def prep_data(df, df_out, headers_unique):    
        """
        Extract Values for Elements and insert into df_prep
        """
        index_ = 0 
        for date in tqdm(headers_unique['DATE']):
            date_idx = df['DATE'] == date
            df_by_date = df[date_idx]
            df_out.loc[index_, 'DATE'] = date 
            for idx in df_by_date['ELEMENT'].index:
                df_out.loc[index_, df_by_date['ELEMENT'][idx]] = df_by_date['VALUE1'][idx]
            index_ = index_+1
            
    def addCol(self, label):
        pass


class StatsReport:
    def __init__(self):
        self.statsdf = pd.DataFrame()
        self.statsdf['stat'] = ['cardinality', 'mean', 'median', 'n_at_median', 'mode', 'n_at_mode', 'stddev', 'min', 'max', 'nzero', 'nmissing']
        pass
    
    def addCol(self, label, data):
        self.statsdf[label] = [self.cardinality_(data), self.mean_(data), 
                               self.median_(data), self.n_at_median(data), 
                               self.mode_(data), self.n_at_mode(data), 
                               self.std_(data), self.min_(data), 
                               self.max_(data), self.nzero_(data), 
                               self.nmissing_(data)]
        
    def to_string(self):
        return self.statsdf.to_string()
    
    def cardinality_(self, d):
        try:
            return d.nunique()
        except:
            return "N/A"
        
    def mean_(self, d):
        try:
            return d.mean()
        except:
            return "N/A"

    def median_(self, d):
        try:
            return d.median()
        except:
            return "N/A"

    def n_at_median(self, d):
        try:
            n = d == d.median()
            return n.sum()
        except:
            return "N/A"     
        
    def mode_(self, d):
        try:
            return int(d.mode())
        except:
            return "N/A"      
        
    def n_at_mode(self, d):
        try:
            n = d == int(d.mode())
            return n.sum()
        except:
            return "N/A"
        
    def std_(self, d):
        try:
            return d.std()
        except:
            return "N/A"     
        
    def min_(self, d):
        try:
            return d.min()
        except:
            return "N/A"       
        
    def max_(self, d):
        try:
            return d.max()
        except:
            return "N/A"
        
    def nzero_(self, d):
        try:
            n = d == 0
            return n.sum()
        except:
            return "N/A"      
        
    def nmissing_(self, d):
        try:
            n = d.isna()
            return n.sum()
        except:
            return "N/A"