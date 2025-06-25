import streamlit as st
import pandas as pd
import pickle
import time

st.set_page_config(page_title="ML Portfolio", layout="wide", page_icon=':sparkles:',)

st.write("""
# Welcome to my machine learning dashboard

This dashboard created by : [@defi.mediana](https://www.linkedin.com/in/defi-mediana/)
""")

add_selectitem = st.sidebar.selectbox("Want to open about?", (" ", "Iris Species", "Heart Disease"))

def iris():
    st.write("""
    This app predicts the **Iris Species**
    
    Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', min_value=4.3, value=6.5, max_value=10.0)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, value=3.3, max_value=5.0)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)',min_value= 1.0, value=4.5, max_value=9.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)',min_value= 0.1, value=1.4, max_value=5.0)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()
    # img = Image.open("iris.JPG")
    st.image("https://www.easytogrowbulbs.com/cdn/shop/products/BeardedIrisColorfullMix_VIS-sqWeb_8a293612-7bc0-4a9f-89ac-917e820d0ccb.jpg?v=1664472481&width=1920", width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("generate_iris.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)
        result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")

def heart():
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', min_value=0, value=1, max_value=3, step=1)
            if cp == 0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 1:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 2:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien : ", wcp)

            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            # exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            exang=st.sidebar.radio("Exercise induced angina", options=["Yes","No"])
            if exang == "Yes":
                exang = 1
            else:
                exang = 0 
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Wanita', 'Pria'))
            if sex == "Wanita":
                sex = 0
            else:
                sex = 1 
            # age = st.sidebar.slider("Usia", min_value=29, max_value=77, value=30, step=1)
            age=st.sidebar.number_input("Usia", min_value=29, max_value=77, value=30, step=1)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
    
    input_df = user_input_features()
    st.image("https://drramjimehrotra.com/wp-content/uploads/2022/09/Women-Heart-Disease-min-resize.png", width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("generate_heart_disease.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        # prediction = loaded_model.predict(df)  
        prediction_proba = loaded_model.predict_proba(df)    
        if prediction_proba[:,1] > 0.4:
            prediction = 1
        else: 
            prediction=0 
        result = ['No Heart Disease Risk' if prediction == 0 else 'Heart Disease Risk Detected']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            if output == "No Heart Disease Risk":
                st.success(f"Prediction : {output}")
            if output == "Heart Disease Risk Detected":
                st.error(f"Prediction : {output}")
                st.info("Please consult a doctor for further evaluation and advice.")
        st.write("### Probability of Heart Disease Risk: " + str(prediction_proba[:,1]))

if add_selectitem == "Iris Species":
    iris()
elif add_selectitem == "Heart Disease":
    heart()
