# Telecom Churn — анализ оттока клиентов

> Учебный проект по исследованию поведения клиентов телеком-компании и подготовке данных для модели бинарной классификации оттока.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## О проекте

Задача проекта — разобраться, какие характеристики клиента могут быть связаны с расторжением договора, а затем подготовить данные для предсказания целевого признака `Churn`.

Про��кт оформлен как исследовательский Jupyter Notebook: от первичного знакомства с данными и проверки типов до очистки, кодирования категориальных признаков и подготовки набора признаков для классификатора.

**Целевая аудитория:** начинающие специалисты по Data Science, аналитики и разработчики, которым нужен понятный пример полного базового ML-пайплайна на табличных данных.

## Что внутри

- первичный анализ таблицы с помощью `DataFrame.info()`, `head()`, `describe()` и `nunique()`;
- проверка пропусков и преобразование `TotalCharges` в числовой формат;
- заполнение некорректно распознанных значений `TotalCharges` средним значением;
- преобразование бинарных категорий (`Yes`/`No`, пол, наличие услуг) в числа;
- one-hot encoding для договора, типа интернет-сервиса, дополнительных линий и способа оплаты;
- удаление иден��ификатора клиента `customerID`, который не должен использоваться как предиктор;
- подготовка целевой переменной `Churn` в формате `0/1`;
- импорт инструментов `train_test_split`, `RandomForestClassifier`, `classification_report` и `GridSearchCV` для следующего этапа моделирования.

## Структура репозитория

```text
Telecom_churn-draft/
├── README.md             # описание проекта и инструкция по запуску
├── pd.ipynb              # исследование, очистка и подготовка данных
└── telecom_churn.csv     # CSV-датасет с признаками использования телефонной связи
```

## Как устроен анализ

1. Notebook загружает таблицу в `pandas.DataFrame` и изучает её структуру.
2. Числовые и категориальные признаки приводятся к подходящему виду.
3. Категориальные признаки кодируются: часть — бинарным отображением, часть — через `pd.get_dummies()`.
4. `customerID` исключается, а `Churn` становится целевой переменной.
5. На выходе получается готовая матрица признаков: в ноутбуке показан результат размером **7043 строки × 27 столбцов**.
6. Для дальнейшего обучения предусмотрен классификатор Random Forest и подбор гиперпараметров через Grid Search.

## Используемый стек

- **Python** — основной язык проекта;
- **Jupyter Notebook** — интерактивный формат исследования;
- **pandas** — загрузка, очистка и преобразование таблиц;
- **NumPy** — работа с числовыми данными;
- **scikit-learn** — разбиение выборки, Random Forest, метрики и подбор параметров.

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/sudoHOlden/Telecom_churn-draft.git
cd Telecom_churn-draft
```

### 2. Создать виртуальное окружение

```bash
python -m venv .venv
```

Активация в Linux/macOS:

```bash
source .venv/bin/activate
```

Активация в Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

```bash
pip install pandas numpy scikit-learn jupyter
```

### 4. Запустить Notebook

```bash
jupyter notebook pd.ipynb
```

## Данные

В репозитории сейчас находятся два разных источника данных:

- `telecom_churn.csv` — таблица с признаками телефонного использования: штат, длительность аккаунта, тарифные планы, минуты и стоимость звонков, количество обращений и целевой признак `Churn`;
- `pd.ipynb` — ноутбук, рассчитанный на таблицу из классического Telco Customer Churn набора: **7043 записи и 21 исходный столбец**, включая `customerID`, `Contract`, `InternetService`, `MonthlyCharges`, `TotalCharges` и `Churn`.

### Важное замечание

В текущей версии ноутбук вызывает:

```python
df = pd.read_csv('data.csv')
```

Но файла `data.csv` в репозитории нет, а доступный CSV называется `telecom_churn.csv` и имеет другую структуру. Поэтому ноутбук в исходном виде не является полностью воспроизводимым: простая замена имени файла может привести к ошибке на этапах обработки, поскольку названия и смысл столбцов различаются.

Чтобы запуск был стабильным, рекомендуется выбрать один из вариантов:

1. добавить в репозиторий именно Telco Customer Churn датасет под именем `data.csv`;
2. адаптировать notebook под `telecom_churn.csv`, заново определить признаки и целевой столбец;
3. переименовать входной файл и привести preprocessing к фактической схеме данных.

## Пример подготовки данных

Основные операции, реализованные в notebook:

```python
# Преобразование начислений в число
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Заполнение пропу��ков средним значением
df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = (
    df['TotalCharges'].mean()
)

# One-hot encoding категориальных признаков
df = pd.get_dummies(
    df,
    columns=['Contract', 'InternetService', 'MultipleLines', 'PaymentMethod'],
    dtype=int
)
```

После этого бинарные признаки и целевая переменная переводятся в числовой формат, а идентификатор клиента удаляется.

## Моделирование

В notebook импортированы следующие инструменты для обучения и оценки модели:

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
```

Рекомендуемый следующий этап:

- разделить данные на обучающую и тестовую выборки;
- обучить `RandomForestClassifier`;
- оценить accuracy, precision, recall и F1-score;
- проверить матрицу ошибок;
- подобрать гиперпараметры через `GridSearchCV`;
- сравнит�� качество модели с базовой стратегией.

Пример каркаса:

```python
X = df.drop(columns='Churn')
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    random_state=42,
    n_estimators=200
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

## Ограничения текущей версии

- отдельный файл зависимостей пока не добавлен;
- notebook и CSV используют разные схемы данных;
- в проекте пока нет отдельного скрипта обучения, сохранённой модели и API;
- метрики финальной модели не зафиксированы в README;
- preprocessing лучше собрать в единый `Pipeline`, чтобы одинаково обрабатывать train и test данные;
- заполнение пропусков средним значением стоит выполнять внутри pipeline и рассчитывать только по обучающей выборке.

## План развития

- [ ] согласовать датасет и имя входного файла;
- [ ] вынести подготовку данных в отдельный Python-модуль;
- [ ] добавить `requirements.txt` или `pyproject.toml`;
- [ ] собрать preprocessing и модель в `sklearn.pipeline.Pipeline`;
- [ ] добавить полноценное разбиение на train/test;
- [ ] сравнить Random Forest с Logistic Regression и Gradient Boosting;
- [ ] добавить визуализации распределения оттока и важности признаков;
- [ ] зафиксировать результаты экспериментов и лучшие гиперпараметры;
- [ ] добавить тестовый пример предсказания для нового клиента.

## Лицензия

Лицензия в репозитории пока не указана. Перед публичным использованием проекта рекомендуется добавить подходящую лицензию.

## Автор

**sudoHOlden** — [GitHub](https://github.com/sudoHOlden)
