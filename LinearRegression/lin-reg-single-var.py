from pandas import *
from numpy import array
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = read_csv("data.csv")
print(data)


plt.scatter(data.Area, data.Price, color="red", marker="+")
plt.xlabel("Area( M )")
plt.ylabel("Price( TND ) ")
plt.show()
# Create Model
model = LinearRegression()
# Train Model
# convert area is obligatoir to 2d array with reshape or with this
model.fit(data[["Area"]], data.Price)
# Predict Result Of An Area
areas = read_csv("areas.csv" )
pred = model.predict(areas)
areas["Price"] = pred
areas.to_csv("prediction.csv")
print(pred)
