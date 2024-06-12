import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Игнорирование предупреждений
import warnings
warnings.filterwarnings('ignore')

def metrics_subplot(models:pd.DataFrame)->None:
    """
       Построение столбчатых диаграмм метрик для каждой модели.

       Параметры:
       ----------
       models: Входной DataFrame для стандартизации.

       Возвращает:
       ----------
       None.
    """

    plt.figure(figsize=(15, 10))
    palette = sns.color_palette("Set3")  # Выбор цветовой палитры

    for i, column in enumerate(models.columns[1:]):
        plt.subplot(2, 2, i + 1)
        sns.barplot(x=models['Модель'], y=models[column], palette=palette)
        plt.xlabel('Модели')
        plt.ylabel(column)
        plt.xticks(rotation=90)
        plt.grid()

    plt.tight_layout()
    plt.show()



def main(file_path: str, output_dir: str, top_n: int = 5) -> None:
    """
       Основная функция для выполнения всех операций по обработке данных и визуализации метрик моделей.

       Параметры:
       ----------
       file_path: Путь к файлу с итоговыми метриками моделей.
       output_dir: Директория для сохранения выходных данных.
       top_n: Количество топ моделей для выбора (по умолчанию 5).

       Возвращает:
       ----------
       None.
    """
    # Чтение файла с итоговыми метриками
    df = pd.read_csv(file_path)
    # Удаление моделей с R^2 меньше нуля
    df = df[df['R2'] >= 0]
    # Вывод метрик оставшихся моделей
    metrics_subplot(df)
    # Выбор топ-N моделей
    top_models = df.head(top_n)
    # Вывод метрик топ-N моделей
    metrics_subplot(top_models)
    # Вывод наименований моделей
    print(top_models)
    # Сохранение списка моделей в CSV файл
    models = top_models['Модель'].tolist()  # Преобразование столбца 'Модель' в список
    top_models.to_csv(output_dir + "models_for_optimization.csv", index=False)

main("C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/results_computation/model_comparison.csv",
     "C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/results_computation/",
     5
     )