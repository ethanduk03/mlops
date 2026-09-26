import pandas as pd
from sklearn.linear_model import LinearRegression

# Read in sampregdata
df = pd.read_csv("data/sampregdata.csv")

# Drop index column, extract y
df = df.drop(columns = ["Unnamed: 0"])
y = df["y"]

scores = {}
for col in df.columns:
    if col != "y":
        
        # For all x_i, fit model and get R^2 value
        model = LinearRegression()
        X = df[[col]]
        model.fit(X, y)
        r2 = model.score(X, y)
        scores[col] = r2
        
# Fit model using the x_i with the highest R^2 value
best_x = max(scores, key = scores.get)
best_model = LinearRegression()
best_model.fit(df[[best_x]], y)

print(f"Best predictor: {best_x}")
print(f"Intercept: {best_model.intercept_}")
print(f"Coefficient: {best_model.coef_}")
