from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import unittest

class IrisDataProcessor:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.df = None

    def prepare_data(self):

        iris = load_iris()


        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


        data_dict = {}
        for i, name in enumerate(feature_names):
            data_dict[name] = iris.data[:, i]


        self.df = pd.DataFrame(data_dict)


        X = self.df
        y = iris.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=feature_names
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=feature_names
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_stats(self):
        if self.df is None:
            raise ValueError("Data not prepared yet. Call prepare_data() first.")
        return {
            'mean': self.df.mean(),
            'std': self.df.std(),
            'min': self.df.min(),
            'max': self.df.max()
        }

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=200),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.best_model = None
        self.best_score = 0
    
    def run_experiment(self):
        mlflow.set_experiment("iris_classification")


        X_train, X_test, y_train, y_test = self.data_processor.prepare_data()

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):

                model.fit(X_train, y_train)


                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                y_pred = model.predict(X_test)


                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, _, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted'
                )


                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model


                self.log_results(model_name, accuracy, precision, recall, cv_scores)

        return self.best_model

    def log_results(self, model_name, accuracy, precision, recall, cv_scores):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.quantized_model = None

    def quantize_model(self):
        if not isinstance(self.experiment.best_model, LogisticRegression):
            raise ValueError("Quantization is only supported for LogisticRegression")

        model = self.experiment.best_model


        self.quantized_model = LogisticRegression(max_iter=200)


        self.quantized_model.fit(
            self.experiment.data_processor.X_train,
            self.experiment.data_processor.y_train
        )


        self.quantized_model.coef_ = np.round(model.coef_ * 1000) / 1000
        self.quantized_model.intercept_ = np.round(model.intercept_ * 1000) / 1000

        return self.quantized_model

    def run_tests(self):
        test_suite = unittest.TestLoader().loadTestsFromTestCase(IrisModelTests)
        runner = unittest.TextTestRunner()
        runner.run(test_suite)

class IrisModelTests(unittest.TestCase):
    def setUp(self):
        self.processor = IrisDataProcessor()
        self.experiment = IrisExperiment(self.processor)
        self.model = self.experiment.run_experiment()

    def test_model_accuracy(self):
        self.assertGreater(self.experiment.best_score, 0.8)

    def test_data_scaling(self):
        self.processor.prepare_data()
        X_train_mean = self.processor.X_train.mean()
        X_train_std = self.processor.X_train.std()

        for col in X_train_mean.index:
            self.assertAlmostEqual(X_train_mean[col], 0, places=10)
            self.assertAlmostEqual(X_train_std[col], 1, places=10)

if __name__ == "__main__":

    processor = IrisDataProcessor()
    experiment = IrisExperiment(processor)
    best_model = experiment.run_experiment()


    optimizer = IrisModelOptimizer(experiment)
    if isinstance(best_model, LogisticRegression):
        quantized_model = optimizer.quantize_model()


    unittest.main()
