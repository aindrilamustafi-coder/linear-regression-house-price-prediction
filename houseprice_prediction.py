import pandas as pd
import matplotlib.pyplot as plt
#loading dataset
data = {
    'Area': [500, 800, 1000, 1200, 1500, 1800],
    'Bedrooms': [1, 2, 2, 3, 3, 4],
    'Price': [100000, 150000, 200000, 240000, 300000, 360000]
}
p1=pd.DataFrame(data)
print(p1)
x=p1.drop('Price',axis=1)
y=p1['Price']
#train-test-split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
#prediction
y_pred=lr.predict(x_test)
print("Prediction is :" , y_pred)
#accuracy and error
from sklearn.metrics import mean_absolute_error, r2_score
print("error is:",mean_absolute_error(y_test,y_pred))
print("accuracy is :", r2_score(y_test,y_pred))
#plotting of graph
plt.scatter(p1['Area'],p1['Price'],color="blue",label="actual_points")
plt.plot(p1['Area'], lr.predict(x), color='red', label='Prediction Line')
plt.title("Area vs price Houseprice Prediction Graph")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.savefig("output_graph.png")
plt.show()


