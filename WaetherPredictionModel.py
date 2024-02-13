import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

filename = "SriLanka_Weather_Dataset.csv"

weather = pd.read_csv(filename, index_col="time")

null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]

valid_columns = weather.columns[null_pct < 0.05]

weather = weather[valid_columns].copy()

weather.drop(
    [
        "country",
        "temperature_2m_max",
        "temperature_2m_min",
        "sunrise",
        "sunset",
        "apparent_temperature_mean",
        "longitude",
        "latitude",
        "elevation",
        "weathercode",
    ],
    axis=1,
    inplace=True,
)

weather = weather[weather["city"] == "Colombo"]

weather.columns = weather.columns.str.lower()

weather = weather.ffill()

weather.index = pd.to_datetime(weather.index)

# weather["snowfall_sum"].plot()

weather["target"] = weather.shift(-1)["apparent_temperature_max"]

weather = weather.ffill()

rr = Ridge(alpha=0.1)

predictors = weather.columns[~weather.columns.isin(["target", "city"])]


def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)


predictions = backtest(weather, rr, predictors)

meanAbsoluteError = mean_absolute_error(
    predictions["actual"], predictions["prediction"]
)


def pct_diff(old, new):
    return (new - old) / old


def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"

    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather


rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in [
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
    ]:
        weather = compute_rolling(weather, horizon, col)

weather = weather.iloc[14:, :]

weather = weather.fillna(0)


def expand_mean(df):
    return df.expanding(1).mean()


for col in [
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
]:
    weather[f"month_av_{col}"] = (
        weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    )
    weather[f"day_avg_{col}"] = (
        weather[col]
        .groupby(weather.index.day_of_year, group_keys=False)
        .apply(expand_mean)
    )

print(predictions.sort_values("diff", ascending=False))

print(weather.loc["2020-03-07":"2020-03-17"])

print(predictions["diff"].round().value_counts().sort_index)
