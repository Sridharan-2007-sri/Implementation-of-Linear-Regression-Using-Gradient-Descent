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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("ex1.txt", header=None)
data = data.values

# Plot data
plt.scatter(data[:, 0], data[:, 1])
plt.xticks(np.arange(5, 30, step=5))
plt.yticks(np.arange(-5, 30, step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")


# Compute cost function
def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    square_err = (h - y) ** 2
    return 1 / (2 * m) * np.sum(square_err)


# Gradient Descent
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(theta)
        error = X.T.dot(predictions - y)
        descent = alpha * (1 / m) * error
        theta = theta - descent

        J_history.append(computeCost(X, y, theta))

    return theta, J_history


# Prepare data
m = data.shape[0]
X = np.append(np.ones((m, 1)), data[:, 0].reshape(m, 1), axis=1)
y = data[:, 1].reshape(m, 1)

# Initialize theta
theta = np.zeros((2, 1))

# Run Gradient Descent
theta, J_history = gradientDescent(X, y, theta, 0.01, 1500)

# Print hypothesis
print("h(x) = " + str(round(theta[0, 0], 2)) +
      " + " + str(round(theta[1, 0], 2)) + "x")

# Plot cost history
plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Cost function using Gradient Descent")

# Plot regression line
plt.figure()
plt.scatter(data[:, 0], data[:, 1])

x_value = [x for x in range(25)]
y_value = [x * theta[1] + theta[0] for x in x_value]

plt.plot(x_value, y_value, color="r")
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

plt.show()

```

## Output:

<img width="847" height="662" alt="image" src="https://github.com/user-attachments/assets/802cdacf-32b3-4ed4-874e-0062c4f43a5f" />

<img width="466" height="457" alt="592219689-a65749d0-0589-4130-81d2-5bccb856a497" src="https://github.com/user-attachments/assets/797f9733-ed30-46d4-a8b1-7bb748048b90" />

<img width="815" height="650" alt="592219003-415f8f0e-8099-4f4c-9ed7-d1a9b3141e69" src="https://github.com/user-attachments/assets/ce80108a-4cbb-4621-bcf6-65b9757273f5" />

<img width="892" height="637" alt="592219306-c4a25b88-0c21-493d-a819-f6683ac4430c" src="https://github.com/user-attachments/assets/83612f8e-d4b2-4be6-a2e9-ca55baf2203a" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
