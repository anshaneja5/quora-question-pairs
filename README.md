# Quora Question Pairs Project

## Overview

This project involves building a model to predict whether a pair of questions from the Quora dataset are duplicates or not. The model uses machine learning techniques and leverages Git LFS (Large File Storage) for handling large files.

## Project Structure

- `model1.pkl`: The trained model file, managed with Git LFS.
- `questions.csv`: The dataset containing question pairs.
- `main.py`: Streamlit application for interacting with the model.
- Other smaller files related to the project.

After lots of EDA and model building, I selected Random Forest as it had the best accuracy and less FP value. It had around 80% accuracy as of now which can be further improve by 
- by taking full dataset of 400,000 rows (currently taking only 100,000 rows)
- by adding some more advance features using fuzzy logic and length ratios.

  
![image](https://github.com/user-attachments/assets/4d4a8de2-5051-40bc-96c7-fc43f2dd1147)
![image](https://github.com/user-attachments/assets/361c062d-8aba-460a-81ab-ec1bb190315d)



