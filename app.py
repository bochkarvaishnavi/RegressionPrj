import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#load the data
st.title("Simple Linear Regression - Salary Prediction")

st.header("Step1: Load and Display the Data")
uploaded_file=st.file_uploader("Upload CSV file",type="csv")


if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.dataframe(data)

    #step 2: Visualize the data
    st.header("Step 2: Visualize the Salary vs Experience")
    fig, ax=plt.subplots()
    ax.scatter(data["YearsExperience"],data["Salary"],color='blue')
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Scatter Plot")
    st.pyplot(fig) 

    #step:3 train model
    st.header("step3 : Train Linear Regression Model")
    x=data[["YearsExperience"]]
    y=data["Salary"]


    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=LinearRegression()
    model.fit(x_train,y_train)


    st.success("Model Trained Successfully!")
 
    #step 4: Visualize the Regession Line
    st.header("step4: visualize the regression line")
    fig2,ax2=plt.subplots()
    ax2.scatter(x,y,color='blue',label='Actual Data')
    ax2.plot(x,model.predict(x),color='red',label='Regression Line')
    ax2.set_xlabel("Years of Experience")
    ax2.set_ylabel("Salary")
    ax2.legend()
    st.pyplot(fig2)


    #step 5: Make Predictions
    st.header("Step:5 Predict Salary")
    experience=st.number_input("Enter the years of experience",min_value=0.0,step=0.1)
    if st.button("Predict Salary"):
        prediction = model.predict([[experience]])
        st.success(f"Predicted Salary: {prediction[0]:,.2f}")