# Disaster-Response-Pipeline

## Project Overview
This project implements a machine learning pipeline to categorize emergency messages during disasters. Using a dataset containing real messages sent during disaster events, the system classifies these messages into various categories, enabling emergency response organizations to quickly identify relevant communications and dispatch aid efficiently.

## Features
- **ETL Pipeline**: Extracts data from CSV files, cleans and transforms the data, and loads it into a SQLite database.
- **ML Pipeline**: Trains a multi-output classifier using NLP techniques to categorize disaster messages.
- **Web Application**: Provides an interface where users can input messages and receive classification results in real-time.
- **Data Visualizations**: Displays insights about the training dataset through interactive Plotly visualizations.

## Project Structure
```
disaster-response-pipeline/
│
├── app/
│   ├── templates/
│   │   ├── go.html           # Classification results page
│   │   └── master.html       # Main page with message input and visualizations
│   └── run.py                # Flask application script
│
├── data/
│   ├── DisasterResponse.db   # SQLite database with processed data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py       # ETL pipeline script
│
├── models/
│   ├── classifier.pkl        # Trained model (saved as pickle file)
│   └── train_classifier.py   # ML pipeline script
│
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - flask
  - plotly
  - sqlalchemy
  - joblib

### Instructions
1. Run the ETL pipeline to process data and create the SQLite database:
   ```
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

2. Run the ML pipeline to train the classifier and save the model:
   ```
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. Run the web application:
   ```
   python app/run.py
   ```

4. Open a web browser and navigate to http://localhost:3000/ to access the application.
   

## Web Application Usage
- Enter a message in the text input field on the main page and click "Classify" to see the categories the message belongs to.
- Explore data visualizations that provide insights about the training dataset, including:
  - Distribution of message genres
  - Top 10 most frequent categories
  - Distribution of "related" messages
  - Request vs. offer messages
  - Word count distribution

## Model Details
The machine learning pipeline:
- Processes text data using NLTK for tokenization and lemmatization
- Uses a pipeline with CountVectorizer, TF-IDF Transformer, and a multi-output Random Forest Classifier
- Optimizes the model using GridSearchCV for hyperparameter tuning
- Evaluates performance using classification reports for each category

## Acknowledgements
- Data provided by [Figure Eight](https://www.figure-eight.com/) (now Appen)

