import streamlit as st
import pandas as pd
import joblib

#This is to  Load the trained model

#model = joblib.load('dt_trained_model.pkl')

model = joblib.load('decision_tree_model.pkl')

# Function to convert input values to numeric
def convert_to_numeric(d_cgpa, sleep, study, program, hall, work):
    # Convert d_cgpa, sleep, and study to float
    d_cgpa_numeric = float(d_cgpa)

      # Map sleep and study categories to numeric according to provided mapping
    sleep_mapping = {
        'Less than 6 hours': 0,
        '6 - 8 hours': 1,
        'More than 8 hours': 2
    }
    sleep_numeric = sleep_mapping[sleep]

    study_mapping = {
        'Less than 2 hours': 0,
        '2 - 4 hours': 1,
        'More than 4 hours': 2,
        'I do not study': 3
    }
    study_numeric = study_mapping[study]


    # Map program to numeric (you can assign numbers to each program as per your encoding)
    program_mapping = {
        'ACCOUNTING': 0,  # Start with 0
        'AGRICULTURE IN AGRONOMY AND LANDSCAPE DESIGN': 1,
        'AGRICULTURAL ECONOMICS AND EXTENSION': 2,
        'ANATOMY': 3,
        'ANIMAL SCIENCE': 4,
        'BIOCHEMISTRY': 5,
        'BIOLOGY': 6,
        'BUSINESS ADMINISTRATION': 7,
        'BUSINESS EDUCATION': 8,
        'CHRISTIAN RELIGIOUS STUDIES': 9,
        'COMPUTER SCIENCE': 10,
        'COMPUTER SCIENCE (INFORMATION SYSTEMS)': 11,
        'COMPUTER SCIENCE (TECHNOLOGY)': 12,
        'ECONOMICS': 13,
        'ECONOMICS EDUCATION': 14,
        'EDUCATIONAL PLANNING & ADMINISTRATION': 15,
        'ENGLISH LANGUAGE EDUCATION': 16,
        'ENGLISH STUDIES': 17,
        'FINANCE': 18,
        'FRENCH AND INTERNATIONAL RELATIONS': 19,
        'GUIDANCE AND COUNSELING': 20,
        'HISTORY & INTERNATIONAL STUDIES': 21,
        'INFORMATION RESOURCES MANAGEMENT': 22,
        'INFORMATION TECHNOLOGY': 23,
        'INTERNATIONAL LAW AND DIPLOMACY': 24,
        'LAW': 25,
        'MARKETING': 26,
        'MASS COMMUNICATION': 27,
        'MEDICAL LABORATORY SCIENCE': 28,
        'MEDICINE': 29,
        'MICROBIOLOGY': 30,
        'MUSIC': 31,
        'NUTRITION AND DIETETICS': 32,
        'PHYSICS': 33,
        'NURSING SCIENCE': 34,
        'PHYSIOLOGY': 35,
        'POLITICAL SCIENCE': 36,
        'PUBLIC ADMINISTRATION': 37,
        'PUBLIC HEALTH': 38,
        'SOCIAL WORK AND HUMAN SERVICES': 39,
        'SOFTWARE ENGINEERING': 40,
        'OTHER': 41  # Note the change here
    }
    program_numeric = program_mapping[program]

     # Map hall to numeric according to provided mapping
    hall_mapping = {
        'Classic': 0,
        'Premium': 1,
        'Regular': 2,
        'Off-campus': 3
    }
    hall_numeric = hall_mapping[hall]

    # Map work to numeric according to provided mapping
    work_mapping = {
        'Yes': 0,
        'No': 1
    }
    work_numeric = work_mapping[work]

    return d_cgpa_numeric, sleep_numeric, study_numeric, program_numeric, hall_numeric, work_numeric


# Function to make predictions
def predict_cgpa(d_cgpa, sleep, study, program, hall, work):
    # Convert inputs to numeric
    d_cgpa, sleep, study, program, hall, work = convert_to_numeric(d_cgpa, sleep, study, program, hall, work)

    # Create a DataFrame with the input data, ensuring the correct order of features
    user_data = pd.DataFrame({
        'd_cgpa': [d_cgpa],
        'sleep': [sleep],
        'study': [study],
        'program': [program],
        'hall': [hall],
        'work': [work]
    })[model.feature_names_in_]  # Use the feature names from the trained model

    # Make prediction
    prediction = model.predict(user_data)
    return prediction[0]

def main():
    # Title and subtitle
    st.title('CGPA Prediction Program')
    st.subheader('Enter Feature Values:')

    # User inputs for the new features
    d_cgpa = st.number_input('Desired CGPA', min_value=0.0, max_value=5.0, step=0.01)
    sleep = st.selectbox('On average, how many hours do you sleep per day?', ['Less than 6 hours', '6 - 8 hours', 'More than 8 hours'])
    study = st.selectbox('On average, how many hours do you study per day, excluding attending classes?', ['Less than 2 hours', '2 - 4 hours', 'More than 4 hours', 'I do not study'])
    program = st.selectbox('Undergraduate Program', [
        'ACCOUNTING', 'AGRICULTURE IN AGRONOMY AND LANDSCAPE DESIGN', 'AGRICULTURAL ECONOMICS AND EXTENSION',
        'ANATOMY', 'ANIMAL SCIENCE', 'BIOCHEMISTRY', 'BIOLOGY', 'BUSINESS ADMINISTRATION', 'BUSINESS EDUCATION',
        'CHRISTIAN RELIGIOUS STUDIES', 'COMPUTER SCIENCE', 'COMPUTER SCIENCE (INFORMATION SYSTEMS)', 'COMPUTER SCIENCE (TECHNOLOGY)',
        'ECONOMICS', 'ECONOMICS EDUCATION', 'EDUCATIONAL PLANNING & ADMINISTRATION', 'ENGLISH LANGUAGE EDUCATION',
        'ENGLISH STUDIES', 'FINANCE', 'FRENCH AND INTERNATIONAL RELATIONS', 'GUIDANCE AND COUNSELING', 'HISTORY & INTERNATIONAL STUDIES',
        'INFORMATION RESOURCES MANAGEMENT', 'INFORMATION TECHNOLOGY', 'INTERNATIONAL LAW AND DIPLOMACY', 'LAW',
        'MARKETING', 'MASS COMMUNICATION', 'MEDICAL LABORATORY SCIENCE', 'MEDICINE', 'MICROBIOLOGY', 'MUSIC',
        'NUTRITION AND DIETETICS', 'PHYSICS', 'NURSING SCIENCE', 'PHYSIOLOGY', 'POLITICAL SCIENCE', 'PUBLIC ADMINISTRATION',
        'PUBLIC HEALTH', 'SOCIAL WORK AND HUMAN SERVICES', 'SOFTWARE ENGINEERING', 'OTHER'
    ])
    hall = st.radio('Are you residing in a classic, premium, or regular hall?', ['Classic', 'Premium', 'Regular', 'Off-campus'], horizontal=True)
    work = st.radio('Do you have a part-time job or participate in the work-study program?', ['Yes', 'No'], horizontal=True)

    # Make prediction when user clicks the 'Predict' button
    if st.button('Predict'):
        prediction = predict_cgpa(d_cgpa, sleep, study, program, hall, work)
        st.success(f'Predicted CGPA: {prediction:.2f}')

# Run the Streamlit app
if __name__ == '__main__':
    main()


    # to run , open command prompt, naviagte to the directory of this file
    #type in this command  "jupyter nbconvert --to script project0.3.ipynb"
    #type in this command "streamlit run project0.3.py"
