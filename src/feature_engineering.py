import pandas as pd


def add_features(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Упрощаем редкие титулы
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Кодируем титулы в числа
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    # Размер семьи
    df['FamilySize'] = df['SibSp'] + df['Parch']

    #  Есть ли семья
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Палуба из номера каюты
    df['Deck'] = df['Cabin'].astype(str).str[0]  # Первая буква каюты
    df['Deck'] = df['Deck'].replace('n', 'U')  # U = Unknown (нет данных)
    df = pd.get_dummies(df, columns=['Deck'], drop_first=True)

    return df