import os
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# quelques variables globales utiles
PATH_TO_HAM_DIR = "D:\DEV\PYTHON_PROGRAMMING\emails\ham"
PATH_TO_SPAM_DIR = "D:\DEV\PYTHON_PROGRAMMING\emails\spam"

SPAM_TYPE = "SPAM"
HAM_TYPE = "HAM"

#les tableaux X et Y seront de la meme taille et ordonnes
X = [] # represente l'input Data (ici les mails)
#indique s'il s'agit d'un mail ou non
Y = [] #les etiquettes (labels) pour le training set


def readFilesFromDirectory(path, classification):
    os.chdir(path)
    files_name = os.listdir(path)
    for current_file in files_name:
        message = extract_mail_body(current_file)
        X.append(message)
        Y.append(classification)
       
           
#fonction de lecture du contenu d'un fichier texte donne.
#ici, on fait un peu de traitement pour ne prendre en compte que le "corps du mail".
# On ignorer les en-tetes des mails
def extract_mail_body(file_name_str):
    inBody = False
    lines = []
    file_descriptor = io.open(file_name_str,'r', encoding='latin1')
    for line in file_descriptor:
        if inBody:
            lines.append(line)
        elif line == '\n':
            inBody = True
        message = '\n'.join(lines)
    file_descriptor.close()
    return message

#appel de la fonction de chargement des mails (charger les mail normaux ensuite les SPAM)
readFilesFromDirectory(PATH_TO_HAM_DIR, HAM_TYPE)
readFilesFromDirectory(PATH_TO_SPAM_DIR, SPAM_TYPE)

training_set = pd.DataFrame({'X': X, 'Y': Y})


#------------------


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(training_set['X'].values)

classifier = MultinomialNB()
targets = training_set['Y'].values
classifier.fit(counts, targets)


examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print predictions