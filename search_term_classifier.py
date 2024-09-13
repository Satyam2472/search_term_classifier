# Pre Process function
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import fasttext
import streamlit as st
import numpy as np
import io  # To handle in-memory file objects


def delimiter_remover(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()


def pre_processing(dataframe):
    dataframe = dataframe.drop(['Campaign ID', 'Campaign Name', 'AdGroup Name',
                                'Views', 'Clicks', 'Ad spend', 'Average CPC', 'Click Through Rate in %',
                                'Direct Units Sold', 'Indirect Units Sold', 'Total units sold',
                                'Direct Conversion Rate in %', 'Indirect Conversion Rate in %',
                                'Direct Revenue', 'Indirect Revenue', 'ROI'], axis=1)

    dataframe['Query'] = dataframe['Query'].astype(str)
    dataframe['Type'] = dataframe['Type'].astype(str)
    dataframe['Type'] = '__label__' + dataframe['Type']
    dataframe['Type_description'] = dataframe['Type'] + ' ' + dataframe['Query']

    dataframe['Type_description'] = dataframe['Type_description'].map(delimiter_remover)

    return dataframe


# Cache the loading of the fastText model to avoid reloading each time
@st.cache_data
def load_model(model_path):
    model = fasttext.load_model(model_path)
    return model


# Function for predictions
def predict(model, queries):
    predictions = []
    labels, confidences = model.predict(queries)
    for i in labels:
        predictions.append(i[0].replace('__label__', ''))  # Clean up the label
    return predictions


# Streamlit app
def main():
    st.title("Query Classification with fastText")

    # Load the fastText model
    model = load_model("trained_classifier.bin")  # Path to your fastText model

    # Upload an Excel file
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)

        if 'Query' in df.columns:
            # Predict using the fastText model
            queries = df['Query'].astype(str).tolist()  # Ensure all queries are strings
            df['Prediction'] = predict(model, queries)

            st.write("Data with Predictions:", df.head())

            # Provide download option for the new data with predictions
            output = io.BytesIO()
            df.to_excel(output, index=False)  # Write DataFrame to buffer
            output.seek(0)  # Reset the pointer

            st.download_button(
                label="Download predictions as Excel",
                data=output,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
