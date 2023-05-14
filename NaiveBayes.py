import numpy as np
from sklearn.model_selection import KFold

class NaiveBayes:
    def __init__(self):
        self.cpt = dict()
        self.X_train = None
        self.y_train = None
        self.labels = None

    # Huấn luyện dữ liệu
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.labels = self.y_train.unique()

        for feat in self.X_train.columns:
            if is_categorical(self.X_train[feat]):
                self._cat_prob(feat)

    def _cat_prob(self, feat):
        if feat not in self.cpt.keys():
            self.cpt[feat] = {}

        l1 = self.X_train[self.y_train == self.labels[0]]
        l2 = self.X_train[self.y_train == self.labels[1]]
        n = self.X_train[feat].unique().shape[0]

        for val in self.X_train[feat].unique():
            self.cpt[feat].update({(val, self.labels[0]): (sum(l1[feat] == val) + 1) / (l1.shape[0] + n)})
            self.cpt[feat].update({(val, self.labels[1]): (sum(l2[feat] == val) + 1) / (l2.shape[0] + n)})

    def _num_prob(self, feat, label, x):
        col = self.X_train[self.y_train == label][feat]
        m = np.mean(col)
        v = np.var(col)
        return 1/np.sqrt(2*np.pi*v) * np.exp(-1*((x-m)**2)/(2*v))

    def _class_prob(self, label):
        return sum(self.y_train == label)/self.y_train.shape[0]

    def predict(self, X_test):
        predictions = np.array([])

        for i in range(X_test.shape[0]):
            instance = X_test.iloc[i]
            prob = np.array([])
            for label in self.labels:
                p = self._class_prob(label)
                for feat in X_test.columns:
                    if is_categorical(X_test[feat]):
                        p *= self.cpt[feat][(instance[feat], label)]
                    else:
                        p *= self._num_prob(feat, label, instance[feat])
                prob = np.append(prob, p)
            prob = prob / sum(prob)
            predictions = np.append(predictions, self.labels[prob.argmax()])
        return predictions
    
    # Tính độ ổn định
    def stability_score(self, X, y, n_folds=5):
        # Tạo danh sách lưu các giá trị
        scores = []
        # Chia tập dữ liệu thành các tập huấn luyện  thành các tập con bằng nhau và kiểm tra với n_Folds là 5
        # shuffle sáo trộn dữ liệu
        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, test_index in kf.split(X):
            # Lấy dữ liệu huấn luyện và kiểm tra dựa trên chỉ mục
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            # Lấy nhãn huấn luyện và kiểm tra.
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.fit(X_train, y_train)
            # Dự đoán nhãn kiểm tra 
            y_pred = self.predict(X_test)
            # Tính toán điểm số của mô hình trên tập kiểm tra
            score = self.score(y_test, y_pred)
            scores.append(score)
        return np.std(scores)

    def display_stability_score(self, X, y, n_folds=5):
        score = self.stability_score(X, y, n_folds=n_folds)
        print("Stability score:", score)

    def score(self, y_true, y_test):
        return round(100.0 * sum(y_test == y_true)/len(y_true),3)

def is_categorical(col):
    if set(col) == set(range(col.unique().shape[0])):
        return True
    if col.dtype in (object, bool):
        return True
    return False


