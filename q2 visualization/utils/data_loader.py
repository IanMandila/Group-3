import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    if len(df.columns) == 1:
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"
        ]
        df = df[df.columns[0]].str.split(",", expand=True)
        df.columns = columns

    return df
