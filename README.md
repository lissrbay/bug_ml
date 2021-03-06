## Bug Localization Model

Структура проекта:

- В папке data_aggregation/ лежат скрипты, использовавшиеся для сбора и обработки данных.
    - parser_java_kotlin.py, get_all_changed_methods.py, get_java_methods.py - находят изменившиеся методы в заданном коммите при помощи парсинга файлов в текущем и предыдущем коммите
    - add_path_info.py, match_reports_fixes.py, collect_sources.py - дополняют существующие стеки вызовов разметкой, путями до файлов с кодом и подтягивают актуальные на момент коммита файлы с методами
    - add_w2v_embeddings.py - добавляем эмбеддинги code2vec на основании имени метода
- В папке embeddings/ содержатся файлы, добавляя которые к копии репозитория code2vec/code2seq можно получить соответствующие эмбеддинги при запуске по соответствующей инструкции к репозиториям
    - match_embeddings_with_methods.py - соединяет полученные эмбеддинги и информацию о стеках вызовов в единую структуру
- В папке models/ находятся различные модели, полученные в ходе экспериментов. 
    - baseline.py - наивный бейзлайн - виноват всегда верхний метод в стеке
    - train_model.py - необходим для обучения выбранной модели. В качестве параметров при запуске необходимо указать
        - вид используемых эмбеддингов (code2seq/code2vec/code2vec(wv)) для обучения моделей на соответствующих векторных представлениях
        - rank, если должен использовать RankLoss
        - flat, если ожидается обучение модели на эмбеддингах вне структуры стектрейсов
- В папке results/ лежат результаты обучения моделей.
    
### Used libraries

* GitPython==3.1.8
* PyTorch
