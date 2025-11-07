import pandas as pd

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(df):
    # Простая обработка: заполняем пропуски и кодируем пол
    df['Age'].fillna(df['Age'].median(), inplace=True)
    most_common_port = df['Embarked'].mode()[0]
    df['Embarked'].fillna(most_common_port, inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    return df


