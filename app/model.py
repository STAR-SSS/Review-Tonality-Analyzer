import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os


class SentimentAnalyzer:
    def __init__(self):
        self.model_path = "trained_model.pkl"

        if os.path.exists(self.model_path):
            print("Загружаем сохраненную модель...")
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            print("Модель загружена!")
        else:
            print("Создаем новую модель...")
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words=['и', 'в', 'на', 'с'])),
                ('clf', LogisticRegression(random_state=42))
            ])
            self.is_trained = False
            print("Модель готова")

    def train(self, train_file_path):
        if not self.is_trained:
            try:
                print("Обучение модели на train.csv")
                train_df = pd.read_csv(train_file_path)

                texts = train_df['text'].fillna('').astype(str)
                labels = train_df['label']

                self.model.fit(texts, labels)
                self.is_trained = True

                # СОХРАНЯЕМ МОДЕЛЬ
                joblib.dump(self.model, self.model_path)
                print("Модель обучена и сохранена!")

            except Exception as e:
                print(f"!!Не удалось обучить модель: {e}")
                print("Использование базовых правил..")
        else:
            print("Модель уже обучена!")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        results = []

        for text in texts:
            try:
                if self.is_trained:
                    prediction = self.model.predict([text])[0]
                    results.append(prediction)
                else:
                    text_lower = text.lower()
                    positive_words = ['хорош', 'отличн', 'прекрасн', 'супер', 'рекоменд', 'довол', 'люблю']
                    negative_words = ['плох', 'ужасн', 'кошмар', 'разочарован', 'недовол', 'отвратительн']

                    pos_count = sum(1 for word in positive_words if word in text_lower)
                    neg_count = sum(1 for word in negative_words if word in text_lower)

                    if neg_count > pos_count:
                        results.append(0)
                    elif pos_count > neg_count:
                        results.append(2)
                    else:
                        results.append(1)

            except Exception as e:
                print(f"Ошибка с текстом: {e}")
                results.append(1)

        return results

    def predict_dataframe(self, df, text_column='text'):
        texts = df[text_column].fillna('').astype(str).tolist()
        predictions = self.predict(texts)
        return predictions