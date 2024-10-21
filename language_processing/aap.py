import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
import csv

nltk.download('punkt', quiet=True)

def prepare_dataset():
    # Define the data
    data = [
        {"sentence": "This is a test sentence.", "language": "English"},
        {"sentence": "यह एक परीक्षण वाक्य है।", "language": "Hindi"},
        {"sentence": "Esto es una frase de prueba.", "language": "Spanish"},
        {"sentence": "これはテストの文章です。", "language": "Japanese"},
        {"sentence": "Dies ist ein Testsatz.", "language": "German"}
    ]
    
    # Specify the CSV file name
    csv_file = 'language_data.csv'
    
    # Write data to the CSV file
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["sentence", "language"])
        
        # Write the header
        writer.writeheader()
        
        # Write the rows
        for row in data:
            writer.writerow(row)

    print(f"Dataset prepared and saved to {csv_file}")
    return csv_file

def process_language_data(csv_file):
    # Load dataset
    df = pd.read_csv(csv_file)
    X = df['sentence']
    y = df['language']

    # Tokenize and prepare data
    X = X.apply(lambda x: ' '.join(word_tokenize(x)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train.values.reshape(-1, 1), y_train)

    # Evaluate the model
    accuracy = model.score(X_test.values.reshape(-1, 1), y_test)
    print(f"Model accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    csv_file = prepare_dataset()
    process_language_data(csv_file)