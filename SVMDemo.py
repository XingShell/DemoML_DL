from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import evaluate
import data_loads


mnb = MultinomialNB()



class PredictClass(object):

    def train_predict_evalueate_model(classifier, train_features,train_labels, test_features, test_labels):
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(train_features)
        evaluate.evaluate(train_labels, predictions)
        predictions = classifier.predict(test_features)
        evaluate.evaluate(test_labels, predictions)
        return predictions


class SVM(PredictClass):
    def __init__(self,*,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.svm = SGDClassifier(loss='hinge', class_weight='balanced', penalty='l2', random_state=1, n_iter_no_change=100)
    def train_predict_evalueate_model(self):
        return PredictClass.train_predict_evalueate_model(classifier=self.svm, train_features=self.x_train, train_labels=self.y_train,
                                                    test_features=self.x_test, test_labels=self.y_test)

class LR_Prediction(PredictClass):
    def __init__(self,*,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lr = LogisticRegression(class_weight='balanced')
    def train_predict_evalueate_model(self):
        return PredictClass.train_predict_evalueate_model(classifier=self.lr, train_features=self.x_train, train_labels=self.y_train,
                                                          test_features=self.x_test, test_labels=self.y_test)
# print(lr_predictions)



# positive_xs = pow(batch_xs, 2)
# mnb_predictions = train_predict_evalueate_model(classifier=mnb, train_features=positive_xs, train_labels=batch_ys)

if __name__ == '__main__':
    loads = data_loads.Data_Load()
    x_train, x_test, y_train, y_test = loads.allinsplit()
    scaler = StandardScaler()
    scaler.fit(x_train)
    scaler.fit(x_test)
    d = {}
    d = {'x_train':x_train,'y_train':y_train,'y_test':y_test,'x_test':x_test}
    # print('--------SVM----------')
    # prediction = SVM(**d).train_predict_evalueate_model()
    print('----------LR----------')
    prediction = LR_Prediction(**d).train_predict_evalueate_model()