import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and vectorizer
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)


# Define the preprocessing function
def preprocess_new_data(question1, question2, cv):
    new_data = pd.DataFrame({'question1': [question1], 'question2': [question2]})

    # Feature engineering
    new_data['q1_len'] = new_data['question1'].str.len()
    new_data['q2_len'] = new_data['question2'].str.len()
    new_data['q1_num_words'] = new_data['question1'].apply(lambda row: len(row.split(" ")))
    new_data['q2_num_words'] = new_data['question2'].apply(lambda row: len(row.split(" ")))

    def common_words(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
        return len(w1 & w2)

    new_data['word_common'] = new_data.apply(common_words, axis=1)

    def total_words(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
        return (len(w1) + len(w2))

    new_data['word_total'] = new_data.apply(total_words, axis=1)
    new_data['word_share'] = round(new_data['word_common'] / new_data['word_total'], 2)

    # Vectorize the questions using the loaded CountVectorizer
    questions = [question1, question2]
    q1_arr, q2_arr = np.vsplit(cv.transform(questions).toarray(), 2)

    # Combine features
    features = np.hstack([
        new_data[
            ['q1_len', 'q2_len', 'q1_num_words', 'q2_num_words', 'word_common', 'word_total', 'word_share']].values,
        q1_arr,
        q2_arr
    ])

    return features


# Streamlit App
st.title('Quora Question Pair Duplicate Detection')

st.write("Enter two questions to check if they are duplicates:")

question1 = st.text_input('Question 1')
question2 = st.text_input('Question 2')

if st.button('Check Duplicate'):
    if question1 and question2:
        # Preprocess the input data
        input_features = preprocess_new_data(question1, question2, cv)

        # Predict using the loaded model
        prediction = model.predict(input_features)

        # Display the result
        if prediction[0] == 1:
            st.success("The questions are duplicates.")
        else:
            st.warning("The questions are not duplicates.")
    else:
        st.error("Please enter both questions.")


