import pandas as pd


class ClassificationReport(object):
    def __init__(self, y_true, y_pred, labels = None, target_name = None):
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        if labels:
            self.labels = labels
        else:
            self.labels = y_true.unique()
            
        if target_name:
            self.target_name = target_name
        else:
            self.target_name = self.labels

        self.df = pd.concat([y_true, y_pred], axis = 1)
    
    def cal_confusion_matrix(self):
        cm = []
        df = self.df
        labels = self.labels
        for t_0 in labels:
            one_row = []
            for t_1 in labels:
                num = df[(df.iloc[:, 0] == t_1) & (df.iloc[:, 1] == t_0)].shape[0]
                one_row.append(num)
                
            cm.append(one_row)
        self.confusion_matrix = pd.DataFrame(cm, index = self.target_name, columns = self.target_name)
        
    def one_report(self, t):
        df = self.df
        labels = self.labels
        TP = df[(df.iloc[:, 0] == t) & (df.iloc[:, 1] == t)].shape[0]
        FP = df[(df.iloc[:, 0] != t) & (df.iloc[:, 1] == t)].shape[0]
        TN = df[(df.iloc[:, 0] != t) & (df.iloc[:, 1] != t)].shape[0]
        FN = df[(df.iloc[:, 0] == t) & (df.iloc[:, 1] != t)].shape[0]
        try:
            recall = TP / (TP + FN)
        except:
            recall = None
        
        try:
            precision = TP / (TP + FP)
        except:
            precision = None
            
        sensitivity = recall
        try:
            specifity = TN / (TN + FP)
        except:
            specifity = None
        
        F_1 = 2 * TP / (2 * TP + FP + FN)
        accuracy = None
        kappa = None
        row = [TP, FP, TN, FN, recall, precision, sensitivity, specifity, accuracy, kappa]
        self.row = row
        
    def cal_report(self):
        classification_report = []
        for t in self.labels:
            self.one_report(t)
            classification_report.append(self.row)
            
        self.classification_report = pd.DataFrame(classification_report, index = self.target_name,
                                                 columns = ['TP', 'FP', 'TN', 'FN', 
                                                            'recall', 'precision', 'sensitivity', 
                                                            'specifity', 'accuracy', 'kappa'])
        accuracy = self.df[self.df.iloc[:, 0] == self.df.iloc[:, 1]].shape[0] / self.df.shape[0]
        N = self.df.shape[0]
        n_true = []
        n_pred = []
        for t in self.labels:
            n_true.append(self.df[self.df.iloc[:, 0] == t].shape[0])
            n_pred.append(self.df[self.df.iloc[:, 1] == t].shape[0])
        
        N_tmp = 0
        for i in range(len(n_true)):
            N_tmp += n_true[i] * n_pred[i]
        
        kappa = 1 - (1 - accuracy) / (1 - (N_tmp / N ** 2))
        self.classification_report.loc['overall'] = [None, None, None, None, None,
                                                     None, None, None, accuracy, kappa]
         
    def main(self):
        self.cal_confusion_matrix()
        self.cal_report()

y_true = [0, 1, 2, 2, 1, 0, 2]
y_pred = [0, 2, 1, 2, 1, 2, 1]
#当某些指标在计算时，如果出现分母为0，则相应指标定义为None
#使用说明，labels指明排序的方式，target_name将labels进行说明，两者一一对应
cr = ClassificationReport(y_true, y_pred, labels = [0, 1, 2],target_name = ['c_0', 'c_1', 'c_2'])
cr.main()
#输出混淆矩阵，数据框格式
print(cr.confusion_matrix)
#输出报告，数据框格式
cr.classification_report
