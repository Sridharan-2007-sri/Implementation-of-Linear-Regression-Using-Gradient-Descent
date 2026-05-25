# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: Sridharan B
RegisterNumber:  212225230272

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

data = {
    'Size': [1000, 1200, 1500, 1800, 2000],
    'Price': [300000, 350000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)

X = df[['Size']]
y = df['Price']

model = SGDRegressor()

model.fit(X, y)

prediction = model.predict([[1600]])

print("Predicted Price:", prediction[0])

plt.scatter(X,y)

plt.plot(X, model.predict(X))

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Prediction using SGD Regressor")

plt.show()

```

## Output:

<img width="1246" height="752" alt="Screenshot 2026-05-25 100350" src="https://github.com/user-attachments/assets/c1fa47e2-e6ac-4595-bcef-c55bfdadf37e" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
