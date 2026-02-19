import argparse
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
from csv import writer
import numpy as np

def train_and_evaluate(dataset_loc, train_x, train_y, test_x, test_y, result_file=""):
    # Convert sequences to string format for TF-IDF vectorization
    train_x_str = [' '.join(map(str, seq)) for seq in train_x]
    test_x_str = [' '.join(map(str, seq)) for seq in test_x]

    test_y = ["failure" if l==1.0 else "normal" for l in test_y]
    train_y = ["failure" if l==1.0 else "normal" for l in train_y]

    # Compute TF-IDF embeddings
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None)
    train_x_tfidf = vectorizer.fit_transform(train_x_str)
    test_x_tfidf = vectorizer.transform(test_x_str)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=10, random_state= 42)
    clf.fit(train_x_tfidf, train_y)

    # Predict on test data
    test_y_pred = clf.predict(test_x_tfidf)

    #test_y = [float(1) if label == 'failure' else float(0) for label in test_y]
    #test_y_pred = [float(1) if label == 'failure' else float(0) for label in test_y_pred]

    # Evaluate the classifier
    print("Accuracy:", accuracy_score(test_y, test_y_pred))

    report = classification_report(test_y, test_y_pred, output_dict = True)
    print(report)

        # create new row of results for this dataset
    row = [round(report['failure']['precision'], 3), round(report['failure']['recall'],3), round(report['failure']['f1-score'],3),
         round(report['normal']['precision'], 3), round(report["normal"]['recall'],3), round(report["normal"]['f1-score'],3)]        # extract the name of the dataset to be added to the row
    _, sample_name  = os.path.split(dataset_loc)
        # add the dataset name to the row
    sample_combination = list(map(int, sample_name[:-4].split("_")))
    row = sample_combination + row

        # check if the name of csv file has been specified otherwise would create the file
    if result_file =="":
      with open('results.csv', 'w') as my_new_csv_file:
            pass
      result_file = 'results.csv'

        # add the result row to the csv file
    with open(result_file, 'a', newline='') as f_object:  
          # Pass the CSV  file object to the writer() function
      writer_object = writer(f_object)
          # Pass the data in the list as an argument into the writerow() function
      writer_object.writerow(row)  
          # Close the file object
      f_object.close()

    return clf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="the non-dl model",
                        type=str)
    parser.add_argument('embedding_strategy', help="the type of the embedding strategy use for classification",
                        type=str)
    parser.add_argument('dataset', help="the location of the dataset of labeled log sequences in csv file",
                        type=str)
    parser.add_argument('log_template_loc', help="the location of the csv file of log template ID to log template text",
                        type=str)
    parser.add_argument('--result_file', help="the csv file to store the results of the model",
                        type=str, default="")
    args = parser.parse_args()

    parent_file_name = os.path.dirname(args.dataset)

    filename = os.path.join(parent_file_name, "train_x.txt")

    with open(filename, 'rb') as file_object:
      train_x = pickle.load(file_object)
    
    filename = os.path.join(parent_file_name, "train_y.txt")

    with open(filename, 'rb') as file_object:
      train_y = pickle.load(file_object)
    
    filename = os.path.join(parent_file_name, "test_x.txt")

    with open(filename, 'rb') as file_object:
      test_x = pickle.load(file_object)

    filename = os.path.join(parent_file_name, "test_y.txt")

    with open(filename, 'rb') as file_object:
      test_y = pickle.load(file_object)

    train_and_evaluate(args.dataset, train_x, train_y, test_x, test_y, args.result_file)