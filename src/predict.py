import pandas as pd
from data_preprocessing import load_data, preprocess_data
from feature_engineering import add_features
from model import train_model


train, test  = load_data("data/train.csv","data/test.csv")

train_processed = preprocess_data(train)
train_processed = add_features(train_processed)

test_processed = preprocess_data(test)
test_processed = add_features(test_processed)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'Title', 'FamilySize', 'IsAlone']
x_train = train_processed[features]
y_train = train_processed['Survived']
x_test = test_processed[features]

model = train_model(x_train, y_train)

y_pred = model.predict(x_test)

submittion = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived': y_pred
                           })
submittion.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно создан!")