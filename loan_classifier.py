import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

        # normalize gender values
        self.data['person_gender'] = self.data['person_gender'].str.lower().replace({'fe male': 'female', 'female': 'female', 'male': 'male'})

        # filter age between 18 and 100
        self.data = self.data[(self.data['person_age'] >= 18) & (self.data['person_age'] <= 100)]

        # remove 'other' from home ownership
        self.data = self.data[self.data['person_home_ownership'].str.lower() != 'other']

        # replace NA values in income with median
        self.data['person_income'].fillna(self.data['person_income'].median(), inplace=True)

    def encode_categorical_columns(self):
        cat_cols = self.data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in cat_cols:
            self.data[col] = le.fit_transform(self.data[col])

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.create_model()

    def create_model(self):
        self.model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def make_prediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def create_report(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))

    def tuning_parameter(self):
        params_xgb = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3]
        }
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)
        random_search = RandomizedSearchCV(xgb_model, param_distributions=params_xgb, n_iter=30, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        random_search.fit(self.x_train, self.y_train)
        print("Best Parameters:", random_search.best_params_)
        self.model = random_search.best_estimator_

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

file_path = 'Dataset_A_loan.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.encode_categorical_columns()
data_handler.create_input_output('loan_status')

model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
model_handler.split_data()
print("Before Tuning")
model_handler.train_model()
print("Accuracy:", model_handler.evaluate_model())
model_handler.make_prediction()
model_handler.create_report()

print("After Tuning")
model_handler.tuning_parameter()
model_handler.train_model()
print("Accuracy:", model_handler.evaluate_model())
model_handler.make_prediction()
model_handler.create_report()
model_handler.save_model_to_file('xgboost_tuned_model.pkl')