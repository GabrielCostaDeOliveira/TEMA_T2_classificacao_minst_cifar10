import os

from models.models import Models
from dataloader.data import minst

from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report


df = minst();

x_train, x_test, y_train, y_test = train_test_split(df.x, df. y);

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


    ## save classification_report
    path = './res/'+name;
    if not os.path.exists(path):
        os.makedirs(path)

    print(report)

    with open (path + "_classification_report.txt", 'w') as f:
        f.write(report)

    ## save params
    print("Best parameters: \n", clf.best_params_)

    with open(path + "_params" + '.txt', 'w') as f:
        f.write(str(clf.best_params_))
