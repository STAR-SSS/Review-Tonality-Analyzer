Анализатор тональности отзывов
Веб-приложение для автоматической классификации тональности русскоязычных отзывов.

Работает на всех устройствах

Результаты
-Macro-F1 Score: 0.6442
-точность: 64%

Установка
```bash
pip install -r requirements.txt  # установка библиотек
uvicorn app.main:app --host 0.0.0.0 --port 8000    # запуск сервера

ссылка на сайт: https://review-tonality-analyzer.onrender.com/

ссылка на гит: https://github.com/STAR-SSS/Review-Tonality-Analyzer/tree/main

ссылка на hub.mos: https://hub.mos.ru/diked12/review-tonality-analyzer