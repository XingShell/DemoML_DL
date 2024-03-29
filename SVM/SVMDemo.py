# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from util import data_loads
from util import evaluate


class PredictClass(object):
    def train_predict_evalueate_model(classifier, train_features,train_labels, test_features, test_labels):
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(train_features)
        evaluate.evaluate(train_labels, predictions)
        predictions = classifier.predict(test_features)
        evaluate.evaluate(test_labels, predictions)
        return predictions

class SVM(PredictClass):
    def __init__(self,*,x_train=None,x_test=None,y_train=None,y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.svm = SGDClassifier(loss='hinge', class_weight='balanced', penalty='l2', random_state=1, n_iter_no_change=500)
    def train_predict_evalueate_model(self):
        return PredictClass.train_predict_evalueate_model(classifier=self.svm, train_features=self.x_train, train_labels=self.y_train,
                                                    test_features=self.x_test, test_labels=self.y_test)
    def train(self):
        self.svm.fit(x_train,y_train)
        predictions = self.svm.predict(self.x_train)
        evaluate.evaluate(self.y_train, predictions)
        joblib.dump(self.svm, 'Modelsave/train_svm.m')

    def loadModelPredict(self,x):
        self.svm = joblib.load("Modelsave/train_svm.m")
        scaler = StandardScaler()
        scaler.fit(x)
        predicted = self.svm.predict(x)
        return predicted


if __name__ == '__main__':
    loads = data_loads.Data_Load()
    # x_train, x_test, y_train, y_test = loads.allinsplit()
    scaler = StandardScaler()
    x_train, y_train = loads.all()
    scaler.fit(x_train)
    y_test = None
    x_test = None
    d = {}
    d = {'x_train':x_train,'y_train':y_train,'y_test':y_test,'x_test':x_test}
    a = SVM(**d)
    a.train()





# b = SVM()
# evaluate.evaluate(b.loadModelPredict(x_train))
# print('--------SVM----------')
# prediction = SVM(**d).train_predict_evalueate_model()
# print('----------LR----------')
# prediction = LR_Prediction(**d).train_predict_evalueate_model()

# mnb = MultinomialNB()
# class LR_Prediction(PredictClass):
#     def __init__(self,*,x_train,x_test,y_train,y_test):
#         self.x_train = x_train
#         self.y_train = y_train
#         self.x_test = x_test
#         self.y_test = y_test
#         self.lr = LogisticRegression(class_weight='balanced')
#     def train_predict_evalueate_model(self):
#         return PredictClass.train_predict_evalueate_model(classifier=self.lr, train_features=self.x_train, train_labels=self.y_train,
#                                                           test_features=self.x_test, test_labels=self.y_test)
# print(lr_predictions)



# positive_xs = pow(batch_xs, 2)
# mnb_predictions = train_predict_evalueate_model(classifier=mnb, train_features=positive_xs, train_labels=batch_ys)