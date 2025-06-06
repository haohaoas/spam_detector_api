import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from src.features.FeatureExtractor import FeatureExtractor

def train_and_save_all_models(data_path, model_prefix, fe_path):
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    labels = df['label'].apply(lambda x: 1 if x == 'spam' or x == '垃圾短信' else 0).values

    fe = FeatureExtractor(max_features=1000)
    X = fe.fit_transform(df['text'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    # 1. XGBoost
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale)
    model_xgb.fit(X_train, y_train)
    print(f"{model_prefix}_xgb.pkl 准确率：", accuracy_score(y_test, model_xgb.predict(X_test)))
    print("XGBoost\n", classification_report(y_test, model_xgb.predict(X_test), digits=4))
    joblib.dump(model_xgb, f"{model_prefix}_xgb.pkl")

    # 2. 朴素贝叶斯
    model_nb = MultinomialNB()
    model_nb.fit(X_train, y_train)
    print(f"{model_prefix}_nb.pkl 准确率：", accuracy_score(y_test, model_nb.predict(X_test)))
    print("NaiveBayes\n", classification_report(y_test, model_nb.predict(X_test), digits=4))
    joblib.dump(model_nb, f"{model_prefix}_nb.pkl")

    # 3. 逻辑回归
    model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_lr.fit(X_train, y_train)
    print(f"{model_prefix}_lr.pkl 准确率：", accuracy_score(y_test, model_lr.predict(X_test)))
    print("LogisticRegression\n", classification_report(y_test, model_lr.predict(X_test), digits=4))
    joblib.dump(model_lr, f"{model_prefix}_lr.pkl")

    # 保存特征器
    fe.save(fe_path)

if __name__ == "__main__":
    train_and_save_all_models('../../data/en_balanced.csv', 'model_en', 'fe_en.pkl')
    train_and_save_all_models('../../data/cn_balanced.csv', 'model_cn', 'fe_cn.pkl')