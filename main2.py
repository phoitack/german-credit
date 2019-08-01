from data_loader.data_loader import DataLoader

features, target = DataLoader.load_data()

print(features.head())
print(target.head())

from models.random_forest.model import Model

model = Model(criterion='gini', max_depth=4, max_features=4)
model.fit(features, target)
print(model.predict(features[:10]))
