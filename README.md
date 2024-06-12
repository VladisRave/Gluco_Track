# Glucko_Track

## Описание проекта

**Glucko_Track** - это проект, направленный на обработку и анализ данных с мультисенсорной системы по неинвазивному определению концентрации глюкозы оптическим путём. Проект состоит из четырёх основных компонентов, каждый из которых реализован в отдельной папке:

1. **Data Engineering**: блок программы, предназначенный для предварительной обработки и объединения разных типов данных.
2. **Data Mining**: блок программы, предназначенный для извлечения значимых признаков из мультисенсорной системы посредством корреляционного анализа.
3. **Data Science**: блок программы, предназначенный для построения базовых моделей машинного обучения и анализа применямости их в данной задаче, а также оптимизации 5 лучших моделей.
4. **ML Engineering**: блок программы, предназначенный для дообучения и визуализации модели, показавшей лучшие значения при определении концентрации глюкозы.

Кроме того, в проекте есть папка **0_Data**, куда необходимо поместить файлы данных, которые можно скачать по следующей ссылке: https://drive.google.com/drive/folders/1WuLmdpwQWfuyOkL-8eMjDGuEX06ng0GO?usp=sharing.

## Установка и настройка

1. **Скачивание данных**:
   - Перейдите по ссылке и скачайте файлы данных.
   - Поместите скачанные файлы в папку **0_Data**.

2. **Корректировка путей**:
   - В коде проекта содержатся пути вида `C:/Users/User/PycharmProjects/Gluco_Track/...`.
   - Пожалуйста, замените эти пути на актуальные пути, соответствующие вашему расположению проекта. Например, если ваш проект находится в папке `D:/Projects/Glucko_Track`, замените все вхождения `C:/Users/User/PycharmProjects/Gluco_Track/...` на `D:/Projects/Glucko_Track/...`.

3. **Установка зависимостей**:
   - Все необходимые библиотеки и их версии перечислены в файле `requirements.txt`.
   - Для установки зависимостей выполните команду:
     ```sh
     pip install -r requirements.txt
     ```

## Дополнительная информация

Для более подробного объяснения назначения каждого файла и примеров визуализаций обратитесь к файлу **Glucko_Track_manual**. Этот документ содержит исчерпывающую информацию о структуре проекта, предназначении папок и файлов, а также примеры использования кода и визуализаций данных.

## Структура проекта

```plaintext
Glucko_Track/
├── 0_Data/                      # Папка для данных (заполнить файлами по ССЫЛКА)
├── 1_Data_Engineering/          # Код для обработки данных
├── 2_Data_Mining/               # Код для извлечения данных
├── 3_Data_Science/              # Код для анализа данных
├── 4_ML_Engineering/            # Код для разработки моделей
├── Glucko_Track_manual.md       # Руководство по проекту
└── requirements.txt             # Файл с зависимостями
```

## Лицензия

Этот проект лицензирован в соответствии с условиями лицензии MIT. Подробнее см. в файле LICENSE.

---
Этот README-файл разработан для облегчения понимания структуры и настройки проекта Glucko_Track как для разработчиков, так и для обычных пользователей. Следуя приведенным инструкциям, вы сможете быстро настроить и начать работу с проектом.