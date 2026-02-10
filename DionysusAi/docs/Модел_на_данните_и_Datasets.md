Използван датасет

Име: Wine Quality Dataset
Източник: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
Брой записи: ~6497 (обединени червени и бели вина)
Характеристики: 12 (11 физико-химични + type: red/white)
Целева променлива: quality (3–9, експертна оценка)
Тип данни: регресия (в проекта се използва като непрекъсната величина)

Модел на данните в SQLite:
CREATE TABLE IF NOT EXISTS wine_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type INTEGER NOT NULL,
    fixed_acidity REAL NOT NULL,
    volatile_acidity REAL NOT NULL,
    citric_acid REAL NOT NULL,
    residual_sugar REAL NOT NULL,
    chlorides REAL NOT NULL,
    free_sulfur_dioxide REAL NOT NULL,
    density REAL NOT NULL,
    pH REAL NOT NULL,
    sulphates REAL NOT NULL,
    alcohol REAL NOT NULL,
    quality_score REAL NOT NULL,
    quality_label TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
