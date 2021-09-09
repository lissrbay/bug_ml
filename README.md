## Bug Localization Model

Структура проекта:

- В папке data_aggregation/ лежат скрипты, использовавшиеся для сбора и обработки данных.
- В папке embeddings/ содержатся файлы, добавляя которые к копии репозитория code2vec/code2seq можно получить соответствующие эмбеддинги при запуске по соответствующей инструкции к репозиториям
- В папке models/ находятся различные модели, полученные в ходе экспериментов. 

Как пользоваться:

- Для сбора данных необходимо запустить скрипт collect_data.py, , но перед этим следует указать параметры обучения в collect_data_properties.json.
    - "reports_path" - путь к стектрейсам,
    - "intellij_path" - путь к репозиторию с intellij idea,
    - "embeddings_type" - тип эмбеддингов,
    - "files_limit" - ограничение на количество методов в стектрейсах,
- Для обучения модели необходимо запустить скрипт train_model.py, но перед этим следует указать параметры обучения в train_properties.json.
    - "reports_path" - путь к стектрейсам,
    - "frame_limit" - ограничение на количество методов в стектрейсах,
    - "embeddings_path" - путь к эмбеддингам методов,
    - "labels_path" - путь к разметке,
    - "report_ids_path" - путь к айди стектрейсов,
    - "report_code_path" - путь к коду методов в стектрейсах,
    - "save_dir": "./data" - путь к папке где будут сохранены модели
    
### Как поставить pycode2seq

1. conda env create -f ./env.yml --force
2. pip install code2seq==0.0.2
3. pip install -r requirements.txt
4. pip install --no-deps pycode2seq==0.0.4

### Как запустить тест апи

    model = Code2Seq.load("java") # запускаем pycode2seq модельку
    stacktrace = json.load(open("ex_api_stacktrace.json", "r")) # загружаем стектрейс с кодом внутри в base64 формате
    api = BugLocalizationModelAPI(lstm_model_path='./data/lstm_20210909_1459', cb_model_path='./data/cb_model_20210909_1459') # пути до обученных моделек
    top_k_pred, scores = api.predict(stacktrace, pred_type='all') # предикт номеров подозрительных методов и их скоры от модельки

### Used libraries

* GitPython==3.1.8
* PyTorch
* code2seq==0.0.2
* pycode2seq==0.0.4


