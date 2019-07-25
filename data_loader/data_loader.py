import pandas as pd
from sklearn.datasets import load_iris


class DataLoader:
    @staticmethod
    def load_data():
        """
        Method that returns two Python objects, the features and the corresponding targets. This will usually involve
        either querying a database or downloading one or more files from the internet. When credentials are needed,
        these can be added as "Secrets" in the UI. By adding `secrets` as argument to the `load_data` method, Cubonacci
        knows to inject the secrets in the method. The `secrets` argument is a dictionary where the keys are the names
        of the secrets and the values are the values given in the user interface.
        :return: (features, target) tuple. Both the features and the target are Python objects that need to be iterable
                 and need to be the same size. Cubonacci automatically infers the relevant schemas. Column names or
                 dictionary keys can only contain letters and underscores at the moment.
        """
        X, _ = load_iris(return_X_y=True)
        columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
        features = pd.DataFrame(X, columns=columns)
        y = ['Setosa'] * 50 + ['Versicolor'] * 50 + ['Virginica'] * 50
        target = pd.DataFrame({"Species": y})
        return features, target
