
import data_loads
import evaluate
import numpy as np
import xgboost as xgb
params = {'booster': 'gbtree',
             'class_weight': 'balanced',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1}

def xgbtrain():
    loads = data_loads.Data_Load()
    x_train, x_test, y_train, y_test = loads.allinsplit()
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtrain.save_binary('train.buffer')
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=1500, evals=watchlist)

    ypred = bst.predict(dtrain)
    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    y_pred = (ypred >= 0.5) * 1
    print('-------train---------')
    evaluate.evaluate(y_train, y_pred)
    ypred = bst.predict(dtrain)
    y_pred = (ypred >= 0.5) * 1
    evaluate.evaluate(y_train, y_pred)
    print('-------test---------')
    dtest = xgb.DMatrix(x_test)
    ypred = bst.predict(dtest)
    y_pred = (ypred >= 0.5) * 1
    evaluate.evaluate(y_test, y_pred)
    bst.save_model('xgb.model')


def xgbtrain01():
    loads = data_loads.Data_Load()
    x_train, y_train = loads.all()
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtrain.save_binary('train.buffer')
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=1200, evals=watchlist)

    ypred = bst.predict(dtrain)
    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    y_pred = (ypred >= 0.5) * 1
    print('-------train---------')
    evaluate.evaluate(y_train, y_pred)
    ypred = bst.predict(dtrain)
    y_pred = (ypred >= 0.5) * 1
    evaluate.evaluate(y_train, y_pred)
    bst.save_model('xgb.model')

def xgbtest():
    loads = data_loads.Data_Load()
    x,y = loads.all()
    tar = xgb.Booster(model_file='xgb.model')
    dtest = xgb.DMatrix(x)
    ypred = tar.predict(dtest)
    # 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
    y_pred = (ypred >= 0.5) * 1
    print('-------test---------')
    evaluate.evaluate(y,y_pred)
def xgbpredict(x):
    tar = xgb.Booster(model_file='xgb.model')
    dtest = xgb.DMatrix(x)
    predict = tar.predict(dtest)
    predict = (predict >= 0.5) * 1
    return predict
def xgbpredictBatch(x):
    tar = xgb.Booster(model_file='xgb.model')
    dtest = xgb.DMatrix(x)
    predict = tar.predict(dtest)
    predict = (predict >= 0.5) * 1
    return predict




# def simply_evalue(true_labels,prediction_labels,classnum=2):
#     dmatix = {}
#     realy = {}
#     for i in range(classnum):
#         dmatix[i] = 0
#         realy[i] = 0
#     for xindex, x_i in enumerate(true_labels):
#         if x_i == prediction_labels[xindex]:
#             dmatix[x_i] += 1
#         realy[x_i] += 1
#     for key in dmatix:
#         print('类别%d正确率%f'%(key, dmatix[key]/realy[key]))
# simply_evalue(y_train,y_pred)
if __name__ == '__main__':
    xgbtrain01()
    # xgbtest()