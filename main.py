import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib
matplotlib.use('TkAgg')
np.random.seed(1)
train_file_path = 'data/cs-training.csv'
test_file_path = 'data/cs-test.csv'


def set_pandas():
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置列宽
    pd.set_option('max_colwidth', 1000)
    # 设置行宽
    pd.set_option('display.width', None)


def preprocess():
    df = pd.read_csv('data/cs-training.csv')
    df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    df.drop_duplicates()
    for c in df.columns:
        if c in ["no","SeriousDlqin2yrs"]:
            continue
        #print("------------------------------------------------")
        #print(c, type(df[c]))
        mean_val = df[c].mean()
        max_val = df[c].max()
        df[c] = df[c].fillna(value=mean_val)
        df[c] = df[c].apply(lambda x: float(x) / max_val)
    #print(df.info())
    #print(df['SeriousDlqin2yrs'].value_counts())

    #corr = df.corr()


    dataset = df.values
    new_dataset = []
    d0=0
    d1=0
    for d in dataset:
        if d1<10000 and d[1]>0.90:
            d1+=1
            new_dataset.append(d)
        if d0<10000 and d[1]<0.10:
            d0+=1
            new_dataset.append(d)
    dataset=np.asarray(new_dataset)




    validate_ids = np.random.choice(range(dataset.shape[0]), int(0.01 * dataset.shape[0]), replace=False)
    #validate_ids = np.asarray(range(140000,150000))
    train_ids = set(range(dataset.shape[0])) - set(validate_ids)
    train_ids, validate_ids = sorted(train_ids), sorted(validate_ids)
    train_dataset = dataset[train_ids]
    validate_dataset = dataset[validate_ids]
    train_input = train_dataset[:, 2:]
    train_label = np.asarray(train_dataset[:, 1], dtype=np.int8)
    validate_input = validate_dataset[:, 2:]
    validate_label = np.asarray(validate_dataset[:, 1], dtype=np.int8)
    return train_input, train_label, validate_input, validate_label


def eval(y_true, y_score):
    y_pred = np.asarray(list(map(lambda x: 0 if x<=0.5 else 1, y_score)), np.int8)
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    presion_score = metrics.precision_score(y_true, y_pred)
    recall_score = metrics.recall_score(y_true, y_pred)
    ap = metrics.average_precision_score(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    return accuracy_score, presion_score, recall_score, ap, auc



def main():
    set_pandas()
    preprocess()
    train_input, train_label, validate_input, validate_label = preprocess()
    #validate_input, validate_label = train_input, train_label
    #from Knn import call
    #from SVM import call
    from GBDT import call
    predict_score = call(train_input, train_label, validate_input, validate_label)
    accuracy_score, presion_score, recall_score, ap, auc = eval(validate_label, predict_score)
    print(accuracy_score, presion_score, recall_score, ap, auc)



if __name__ == '__main__':
    main()
