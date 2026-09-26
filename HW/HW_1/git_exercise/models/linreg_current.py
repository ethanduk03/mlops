import pandas as pd
from sklearn.linear_model import LinearRegression

# Read in sampregdata
df = pd.read_csv("data/sampregdata.csv")

# Drop index column, extract y, get list of columns of X
df = df.drop(columns = ["Unnamed: 0"])
y = df["y"]
X_cols = [col for col in df.columns if col != "y"]

scores = {}
for i in range(len(X_cols)):
    for j in range(i + 1, len(X_cols)):
        
        x1 = X_cols[i]
        x2 = X_cols[j]
        X = df[[x1, x2]]
        
        # For each possible pair of x_i and x_j (with i != j), get the R^2 for that combination of predictors
        model = LinearRegression()
        model.fit(X, y)
        scores[(x1, x2)] = model.score(X, y)
        
# Fit model using the pair of predictors resulting in the highest R^2 value
best_xs = max(scores, key = scores.get)
best_X = df[[best_xs[0], best_xs[1]]]
best_model = LinearRegression()
best_model.fit(best_X, y)

print(f"Best predictor: {best_xs}, R^2 = {scores[best_xs]}")
print(f"Intercept: {best_model.intercept_}")
print(f"Coefficient: {best_model.coef_}")