
# coding: utf-8

# ### Importing necessary libraries

# In[ ]:

import pandas as pd
import numpy as np
import gc


# In[ ]:

# Setting working directory
# Change to your working dirctory

path = '/home/kelsh/MLProjects/Instacart/Data/'


# ### Loading data files

# In[ ]:

print('loading files, this can take long time ...')

aisles = pd.read_csv(path + 'aisles.csv')
dep = pd.read_csv(path + 'departments.csv')
ord_prd_pr = pd.read_csv(path + 'order_products__prior.csv')
ord_prd_tr = pd.read_csv(path + 'order_products__train.csv')
orders = pd.read_csv(path + 'orders.csv')
products = pd.read_csv(path + 'products.csv')


# ### Assigning variables types, removing uncessary float varliabs to integers

# In[ ]:

# Changing some columns types to int for less memory usage

aisles['aisle'] = aisles['aisle'].astype('category')
dep['department'] = dep['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')

orders.order_dow = orders.order_dow.astype(np.int8)
orders.order_hour_of_day = orders.order_hour_of_day.astype(np.int8)
orders.order_number = orders.order_number.astype(np.int16)
orders.order_id = orders.order_id.astype(np.int32)
orders.user_id = orders.user_id.astype(np.int32)
orders.days_since_prior_order = orders.days_since_prior_order.astype(np.float32)

products.aisle_id = products.aisle_id.astype(np.int8)
products.department_id = products.department_id.astype(np.int8)
products.product_id = products.product_id.astype(np.int32)

ord_prd_tr.reordered = ord_prd_tr.reordered.astype(np.int8)
ord_prd_tr.add_to_cart_order = ord_prd_tr.add_to_cart_order.astype(np.int16)

ord_prd_pr.order_id = ord_prd_pr.order_id.astype(np.int32)
ord_prd_pr.add_to_cart_order = ord_prd_pr.add_to_cart_order.astype(np.int16)
ord_prd_pr.reordered = ord_prd_pr.reordered.astype(np.int8)
ord_prd_pr.product_id = ord_prd_pr.product_id.astype(np.int32)


# ### Reshaping Products features and adding other features about reorders

# In[ ]:

print('Reshaping Products data ...')

frames = [products, dep, aisles] # to be merged together

products = pd.concat(frames, axis = 1)
products.drop(['department_id'], axis = 1, inplace = True)
products.drop(['aisle_id'], axis = 1, inplace = True)

ord_prd_tr = ord_prd_tr.merge(orders[['user_id','order_id']], left_on = 'order_id', right_on = 'order_id', how = 'inner')

orders_products = orders.merge(ord_prd_pr, how = 'inner', on = 'order_id')

# Memory cleaning
del ord_prd_pr
gc.collect()

# sorting orders and products to get the rank or the reorder times for the product
prdss = orders_products.sort_values(['user_id', 'order_number', 'product_id'], ascending=True)

prdss = prds.assign(product_time = prds.groupby(['user_id', 'product_id']).cumcount()+1)

# getting products ordered first and second times to calculate probability later
sub1 = prdss[prdss['product_time'] == 1].groupby('product_id').size().to_frame('prod_first_orders')
sub2 = prdss[prdss['product_time'] == 2].groupby('product_id').size().to_frame('prod_second_orders')

sub1['prod_orders'] = prdss.groupby('product_id')['product_id'].size()
sub1['prod_reorders'] = prdss.groupby('product_id')['reordered'].sum()
sub2 = sub2.reset_index().merge(sub1.reset_index())

# calculating reorder probability and ratio for the product
sub2['prod_reorder_probability'] = sub2['prod_second_orders']/sub2['prod_first_orders']
sub2['prod_reorder_times'] = 1 + sub2['prod_reorders']/sub2['prod_first_orders']
sub2['prod_reorder_ratio'] = sub2['prod_reorders']/sub2['prod_orders']
prd = sub2[['product_id', 'prod_orders','prod_reorder_probability', 'prod_reorder_times', 'prod_reorder_ratio']]

# Memory cleaning
del sub1, sub2
gc.collect()


# ### Reshaping Users features and adding other features about reorders

# In[ ]:

print('Reshaping Users data ...')

# extracting prior information (features) by user
users = orders[orders['eval_set'] == 'prior'].groupby(['user_id'])['order_number'].max().to_frame('user_orders')
users['user_period'] = orders[orders['eval_set'] == 'prior'].groupby(['user_id'])['days_since_prior_order'].sum()
users['user_mean_days_since_prior'] = orders[orders['eval_set'] == 'prior'].groupby(['user_id'])['days_since_prior_order'].mean()

# merging features about users and orders into one dataset
us = orders_products.groupby('user_id').size().to_frame('user_total_products')
us['eq_1'] = orders_products[orders_products['reordered'] == 1].groupby('user_id')['product_id'].size()
us['gt_1'] = orders_products[orders_products['order_number'] > 1].groupby('user_id')['product_id'].size()

# reorder ratio for the user (grouped by product)
us['user_reorder_ratio'] = us['eq_1'] / us['gt_1']
us.drop(['eq_1', 'gt_1'], axis = 1, inplace = True)
us['user_distinct_products'] = orders_products.groupby(['user_id'])['product_id'].nunique()

# the average basket size of the user
users = users.reset_index().merge(us.reset_index())
users['user_average_basket'] = users['user_total_products'] / users['user_orders']

us = orders[orders['eval_set'] != 'prior']
us = us[['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]

users = users.merge(us)

# Memory cleaning
del us
gc.collect()


# ### Merging all data (users, orders, products) into one dataset

# In[ ]:

# merging orders and products and grouping by user and product and calculating features for the user/product combination
data = orders_products.groupby(['user_id', 'product_id']).size().to_frame('up_orders')
data['up_first_order'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].min()
data['up_last_order'] = orders_products.groupby(['user_id', 'product_id'])['order_number'].max()
data['up_average_cart_position'] = orders_products.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
data = data.reset_index()

# Memory cleaning
del orders_products, orders
gc.collect()

#merging previous data with users
data = data.merge(prd, on = 'product_id')
data = data.merge(users, on = 'user_id')

#user/product combination features about the particular order
data['up_order_rate'] = data['up_orders'] / data['user_orders']
data['up_orders_since_last_order'] = data['user_orders'] - data['up_last_order']
data['up_order_rate_since_first_order'] = data['up_orders'] / (data['user_orders'] - data['up_first_order'] + 1)
data = data.merge(ord_prd_tr[['user_id', 'product_id', 'reordered']], how = 'left', on = ['user_id', 'product_id'])


# ### Now we have our dataset engineered into one dataframe 'data' that combines both train and test sets, has features about users, products, orders and user-product

# ### Next, get the train and test data

# In[ ]:

print('Preparing Train and Test sets ...')

# filter by eval_set = train, replace Nan with zeros (not reordered) and dropping the id's columns
# because they are not paart of the training features.
train = data[data['eval_set'] == 'train']
train['reordered'].fillna(0, inplace=True)
train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis = 1, inplace = True)

# filter by eval_set = test, replace Nan with zeros (not reordered) and
test = data[data['eval_set'] == 'test']
test['reordered'].fillna(0, inplace=True)
test.drop(['eval_set', 'user_id', 'reordered'], axis = 1, inplace = True)

# Saving train and test sets to files for easier loading to model, without having to re-prepare the data.
train.to_csv('my_train.csv', header = True, index = False)
test.to_csv('my_test.csv', header = True, index = False)

# Memory cleaning
del data, train, test
gc.collect()


# ### Finished preparing the data
# ## -------------------------------------------------------------------------------------------------------------

# ### Starting Model preparation

# ### Importing libraries

# In[ ]:

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# #### I will be testing two models: XGBOOST and LightGBM - both are gradient boosting trees. LightGBM has couple of advantages over XGBOOST:
# * LightGBM uses less memory.
# * Fast training efficiency.
# * Deals with large scale of data (more data can fit to GPU).
#
# Howvere it is down to really how each model will perform with data. XGBOOST has a good solid reputation.
#

# ### Loading my train and test files

# In[ ]:

print('loading files ...')
train = pd.read_csv('my_train.csv')
test = pd.read_csv('my_test.csv')

############ If using slow CPU and/or RAM, sample a 40% of the training data and uncomment the following
#print('sampling train data ...')
#train = train.sample(frac=0.4)

# Splitting the training set to train and validation set. Validation set
X_train, X_eval, y_train, y_eval = train_test_split(train[train.columns.difference(['reordered'])], train['reordered'], test_size=0.33, random_state=7)

# memory cleaning
del train
gc.collect()


# ### Creating LightGBM model
#
# #### I use GPU for training the model. If no GPU present, this option can be disabled in the 'params' dictionary.
# #### These parameters are bit greedy, with a regular laptop GPU takes ~ 3 minutes to train. on CPU this will be significantly longer.

# In[ ]:

print('formatting to lgbm format ...')
lgb_train = lgb.Dataset(X_train, label=y_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_iterations' : 1000,
    'max_bin' : 100,
    'num_leaves': 512,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'min_data_in_leaf' : 200,
    'device' : 'gpu',
    'learning_rate' : 0.1,
    'gpu_use_dp' : True,
}


print('training LightGBM model ...')
lgb_model = lgb.train(params,
                lgb_train,
                num_boost_round = 150,
                valid_sets = lgb_eval,     # Validation set used to prevent overfitting
                early_stopping_rounds=10)

del lgb_train, X_train, y_train
gc.collect()

print('applying model to test data ...')
test['reordered'] = lgb_model.predict(test[test.columns.difference(['order_id', 'product_id'])], num_iteration = lgb_model.best_iteration)



# ### Formatting model output to submission csv format

# In[ ]:

print('formatting and writing to submission file ...')
prd_bag = dict()
for row in test.itertuples():
    if row.reordered > 0.21:   ## Cutoff for lableing product as positive (can be tweaked with cross validation)
        try:
            prd_bag[row.order_id] += ' ' + str(row.product_id)
        except:
            prd_bag[row.order_id] = str(row.product_id)

for order in test.order_id:
    if order not in prd_bag:
        prd_bag[order] = 'None'

submit = pd.DataFrame.from_dict(prd_bag, orient='index')

submit.reset_index(inplace=True)
submit.columns = ['order_id', 'products']
submit.to_csv('sub_lgb6.csv', index=False)

