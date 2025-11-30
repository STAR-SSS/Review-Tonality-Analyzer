from sklearn.metrics import f1_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from app.model import SentimentAnalyzer

# загружаем данные
train_df = pd.read_csv('data/train.csv')

# разделяем на train/val
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42
)

#обучаем и предсказываем
analyzer = SentimentAnalyzer()
analyzer.model.fit(X_train, y_train)
predictions = analyzer.model.predict(X_val)

#метрики
f1_macro = f1_score(y_val, predictions, average='macro')
report = classification_report(y_val, predictions)

print(f"Macro-F1 Score: {f1_macro:.4f}")
print("\nClassification Report:")
print(report)

# cохр отчет в файл
with open('macro_f1_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Macro-F1 Score: {f1_macro:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("\noтчет сохранен в macro_f1_report.txt")