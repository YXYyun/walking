import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import cross_val_score
from sklearn import svm

train_file = r"D:\\xxx\\train.csv"
test_file = r"D:\\xxx\\test.csv"


def data_precess(filename):
    df = pd.resd_csv(filename)
    data = []
    for i in df:
        data.append(df[i].astype(np.flost32))
        if "CLASS" in df:
            label = np.array(df['CLASS'].astype(np.compat.long))
        # 用于去掉前面的序号和后面的标签
        data = np.array(data)[1:-1].T
    else:
        label = np.array([])
        # 去掉前面的序号
        data = np.array(data)[1:].T
    return data, label


train_data, train_labels = data_precess(train_file)
test_data, _ = data_precess(test_file)

train_data = (train_data + 1)
test_data = (test_data + 1)



clf = svm.SVC(C=1, kernel="poly", degree=5, gamma="scale", coef0=0.0, shrinking=True, probability=True,
              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr",
              break_ties=False, random_state=None)

scores = cross_val_score(clf, train_data, train_labels, cv=10)
# print(scores)
print("准确率 %0.1f(+/- %0.1f)" % (scores.mean(), scores.std() * 3))


clf.fit(train_data, train_labels)
out = clf.predict(test_data)
sbmit = pd.read_csv(r"D:\\xxx\\submission.csv")
sbmit["CLASS"] = out
sbmit.to_csv('D:\\xxx\\submission.csv')


class shixu_Model1(nn.Module):
    def __init__(self):
        # super(final_Model1,self).__init__()
        self.bn0 = nn.BatchNorm1d(240)
        self.fc1 = nn.Linear(240, 64)
        self.dr1 = nn.Dropout(0.5)
        self.rl1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.dr2 = nn.Dropout(0.8)
        self.rl2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)
        self.dr3 = nn.Dropout(0.8)

    def forward(self, x):
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.dr1(x)

        x = self.rl1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.dr2(x)

        x = self.rl2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.dr3(x)
        return x


