# RedLab Hack 2024

*MISIS PREZOY VYTASHIM team*

Team Members:
1) **Рыжичкин Кирилл** - ML Engineer, Frontend
2) **Чистов Егор** - Backend
3) **Плужников Артём** - ML Engineer, Design
4) **Кадомцев Андрей** - ML Engineer
5) **Силаев Всеволод** - ML Engineer

Презентация: [link](https://drive.google.com/)

Frontend: link

API: link

## Задача трека "Разработка модели для выявления аномалий во временном ряду"

Разработать минимальный прототип сервиса на python который будет анализировать временной ряд и размечать выявленные аномалии в данных. 

## Вводные данные

Cлепок данных телеметрии реальной системы для анализа и обучения модели. Данные представляют собой выгрузку из Clickhouse в формате TSV. 

Ключевые метрики: *Web Response, Throughput, APDEX, Error*

## Предложенное решение

solution
  
Ноутбуки с тестированием идей и разработкой решения: [notebooks](https://github.com/l1ghtsource/redlab-timeseries-anomaly-detection/tree/main/notebooks)

ML алгоритмы: [ml](https://github.com/l1ghtsource/redlab-timeseries-anomaly-detection/tree/main/ml)

Извлеченные данные: [data](https://github.com/l1ghtsource/redlab-timeseries-anomaly-detection/tree/main/data)

Веб-сервис: [app](https://github.com/l1ghtsource/redlab-timeseries-anomaly-detection/tree/main/app.py)

API: [api](https://github.com/l1ghtsource/redlab-timeseries-anomaly-detection/tree/main/api.py)
