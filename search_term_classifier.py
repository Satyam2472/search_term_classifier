# Pre Process function
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import fasttext
import streamlit as st
import numpy as np


def delimiter_remover(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()


def pre_processing(dataframe):

  dataframe = dataframe.drop(['Campaign ID', 'Campaign Name', 'AdGroup Name',
       'Views', 'Clicks', 'Ad spend', 'Average CPC', 'Click Through Rate in %',
       'Direct Units Sold', 'Indirect Units Sold', 'Total units sold',
       'Direct Conversion Rate in %', 'Indirect Conversion Rate in %',
       'Direct Revenue', 'Indirect Revenue', 'ROI'], axis = 1)

  dataframe['Query'] = dataframe['Query'].astype(str)
  dataframe['Type'] = dataframe['Type'].astype(str)
  dataframe['Type'] = '__label__' + dataframe['Type']
  dataframe['Type_description'] = dataframe['Type'] + ' ' + dataframe['Query']

  dataframe['Type_description'] = dataframe['Type_description'].map(delimiter_remover)

  return dataframe


# Cache the loading of the fastText model to avoid reloading each time
@st.cache_data(allow_output_mutation=True)
def load_model(model_path):
    model = fasttext.load_model(model_path)
    return model

# Function for predictions
def predict(model, queries):
    predictions = []
    for query in queries:
        query_array = np.asarray([query])  # Ensure the query is in the correct format
        prediction = model.predict(query_array[0])  # Predict on the array element
        predictions.append(prediction[0][0].replace('__label__', ''))  # Clean up label
    return predictions

# Streamlit app
def main():
    st.title("Query Classification with fastText")
    
    # Load the fastText model
    model = load_model("trained_classifier.bin")  # Path to your fastText model
    
    # Upload an Excel file
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    # uploaded_file = pre_processing(uploaded_file)


    if uploaded_file is not None:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        # st.write("Uploaded Data:", df.head())
        
        if 'Query' in df.columns:
            # Predict using the fastText model
            queries = df['Query'].astype(str).tolist()  # Ensure all queries are strings
            df['Prediction'] = predict(model, queries)
            
            st.write("Data with Predictions:", df.head())

            # Provide download option for the new data with predictions
            st.download_button(
                label="Download predictions as Excel",
                data=df.to_excel(index=False),
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()



