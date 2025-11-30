from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
from app.model import SentimentAnalyzer

app = FastAPI(title="Анализатор тональности отзывов")

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

#загрузка модели
print("запуск сервера...")
analyzer = SentimentAnalyzer()
#обучение модели
#try:
    #analyzer.train("data/train.csv")
#except Exception as e:
    #print(f"не удалось обучить модель: {e}")


#@app.post("/analyze/")
#async def analyze_file(file: UploadFile = File(...)):
    #try:
        #print(f"получен файл: {file.filename}")

        #чтение файла
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if 'text' not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": "Файл должен содержать колонку 'text'"}
            )

        print(f"Анализ {len(df)} отзывов...")

        # Анализ тональности
        predictions = analyzer.predict_dataframe(df)
        df['label'] = predictions

        # Статистика
        stats = {
            "total": len(predictions),
            "negative": predictions.count(0),
            "neutral": predictions.count(1),
            "positive": predictions.count(2)
        }

        #график
        plot_filename = create_visualization(stats)

        #Сохр результат
        output_filename = "result.csv"
        df.to_csv(output_filename, index=False)

        print("✅ Анализ завершен!")

        return {
            "message": "Анализ завершен успешно!",
            "stats": stats,
            "download_link": f"/download/{output_filename}",
            "plot_url": f"/static/{plot_filename}"
        }

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        raise HTTPException(500, detail=f"Ошибка: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(404, detail="Файл не найден")
    return FileResponse(filename, media_type='text/csv', filename=filename)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def create_visualization(stats):
    labels = ['Негативные', 'Нейтральные', 'Позитивные']
    counts = [stats['negative'], stats['neutral'], stats['positive']]
    colors = ['#ff6b6b', '#ffe66d', '#51cf66']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=colors, alpha=0.8)
    plt.title('Распределение тональностей отзывов', fontsize=16, pad=20)
    plt.ylabel('Количество отзывов', fontsize=12)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(count), ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plot_filename = "sentiment_plot.png"
    plt.savefig(f"static/{plot_filename}", dpi=100, bbox_inches='tight')
    plt.close()


    return plot_filename
