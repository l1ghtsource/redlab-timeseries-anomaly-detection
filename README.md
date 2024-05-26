# RedLab Hack 2024

*MISIS PREZOY VYTASHIM team*

Team Members:
1) **Рыжичкин Кирилл** - ML Engineer, Frontend
2) **Чистов Егор** - Backend
3) **Плужников Артём** - ML Engineer, Design
4) **Кадомцев Андрей** - ML Engineer
5) **Силаев Всеволод** - ML Engineer

Презентация: [link](https://drive.google.com/)

Frontend: [prezoy.itatmisis.ru](https://prezoy.itatmisis.ru/)

API: 
[prezoy.itatmisis.ru/api](https://prezoy.itatmisis.ru/api/)

Swagger:
[prezoy.itatmisis.ru:8000/docs](http://prezoy.itatmisis.ru:8000/docs)

## Задача трека "Разработка модели для выявления аномалий во временном ряду"

Разработать минимальный прототип сервиса на python который будет анализировать временной ряд и размечать выявленные аномалии в данных. 

## Вводные данные

Cлепок данных телеметрии реальной системы для анализа и обучения модели. Данные представляют собой выгрузку из Clickhouse в формате TSV. 

Ключевые метрики: *Web Response, Throughput, APDEX, Error*

## Предложенное решение

Из данного нам слепка данных были извлечены необходимые метрики и загружены на Clickhouse, который развернут на удаленном сервере VK Remote. Мы предоставляем пользователю интерфейс, реализованный с помощью Streamlit, на котором он может получить данные из БД или загрузить собственноручно в формате csv. Далее предоставляется следующий пул моделей для детекции аномалий: 

- [AER: Auto-Encoder with Regression for Time Series Anomaly Detection](https://arxiv.org/pdf/2212.13558) - SOTA решение для поиска аномалий во временных рядах, хорошо работает с сезонностями, трендами

![aer_vs_all](aer_vs_all.pg)

- Prophet - качественная детекция аномалий на основе предсказаний и доверительного интервала

- Isolation Forest - 

- Multidimensional - 

## Инструкция к запуску:
все образы выложены на docker hub, поэтому запуск будет относительно быстрый
1. Из папки src вызвать docker compose up
2. подождать
3. на 8000 порту API, на 8501 графический интерфейс StreamLit
Пример запроса к API
```
Метод /find для получения всех аномалий, на вход принимает название колонки и модели \
            с помощью которых будет производиться обработка. необязательный параметр data_source указывает на \
                источник данных в ClickHouse, по умолчанию берутся данные из датасета
```
```json
{
    "models": [
        "Autoencoder",
        "Isolation Forest",
        "Prophet"
    ],
    "column_name": "web_response",
    "data_source": {
        "host": "clickhouse_url",
        "port": "8123",
        "query": "SELECT timestamp, web_response FROM \"default\".\"test2\" ORDER BY timestamp ASC"
    }
}
```
