import numpy as np
import pandas as pd

def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)

if __name__ == "__main__":
    # Read CSV with columns: x, y
    df = pd.read_csv("data.csv")
    x = df["x"].values
    y = df["y"].values

    b_0, b_1 = estimate_coef(x, y)
    print(f"Intercept (b0): {b_0}")
    print(f"Slope (b1): {b_1}")
