import streamlit as st 

st.set_page_config(
    page_title="Diabetes Prediction App", 
    page_icon=":hospital:",
    initial_sidebar_state="expanded",

)

st.write("# Diabetes Prediction App! 👋")

st.markdown(
    """
    This app predicts the probability of a person having diabetes using the Pima Indians Diabetes Dataset!
    The algorithm used is **K-Nearest Neighbors** with **K = 19**. Given inputs of Glucose, BMI, and Age, the model predicts whether the person is diabetic or not.

    ### Documentation and references 
    - Dataset source: [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
    - Strealit [documentation](https://docs.streamlit.io/en/stable/)
    - Flask [documentation](https://flask.palletsprojects.com/en/1.1.x/)
    - Ask a question in our [community

"""
)