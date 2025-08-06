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
    df = pd.read_csv("data.csv")
    x = df["x"].values
    y = df["y"].values

    b_0, b_1 = estimate_coef(x, y)

    with open("regression_output.txt", "w") as f:
        f.write(f"Intercept (b0): {b_0}\n")
        f.write(f"Slope (b1): {b_1}\n")

    print("Regression results saved to regression_output.txt")
