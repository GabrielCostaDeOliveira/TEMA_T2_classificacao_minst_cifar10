from models.models import Models
from dataloader.data import minst, cifar10

import os


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sns




def save_conf_matrix_img(conf_matrix, path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(path + 'Confusion_Matrix.png')




x_train, x_test, y_train, y_test = minst();

def run(data, name_dataset):

    x_train, x_test, y_train, y_test = data;

## carregando modelos
    print("carregando modelos")
    models = Models();

## treinando
    for name, clf in models.get_grid_search():

        print ("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

        print ("treinando " + name)

        clf.fit(x_train, y_train);


        predictions = clf.predict(x_test)
        report = classification_report(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        

        print("report")
        print(report)

        print("best params")
        print(clf.best_params_)

        print("conf_matrix")
        print(conf_matrix)


        ## save classification_report
        path = './res/' + name_dataset + '/' + name + '/';
        if not os.path.exists(path):
            os.makedirs(path)


        with open (path + "classification_report.txt", 'w') as f:
            f.write(report)

        with open (path + "conf_matrix.txt", 'w') as f:
            f.write(str(conf_matrix))

        save_conf_matrix_img(conf_matrix, path)

        with open(path + "params" + '.txt', 'w') as f:
            f.write(str(clf.best_params_))

#run(minst(), 'minst')

run(cifar10(), 'cifar10')


