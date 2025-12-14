import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    df["age"] = df["age"].astype(int)
    df["hours_per_week"] = df["hours_per_week"].astype(int)

    df["income_binary"] = df["income"].map({"<=50k": 0, ">50k": 1})

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30–39", "40–49", "50–59", "60+"]
    )

    return df
