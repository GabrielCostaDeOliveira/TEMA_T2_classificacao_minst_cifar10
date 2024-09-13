from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class Models:
    def __init__(self):
        self.models_dict = {

            "LogisticRegression": (
                LogisticRegression (max_iter= 1000),  
                {'clf__C': [10**x for x in range(-5, 10)], 
                 'clf__penalty': ['l1', 'l2'], 
                 'clf__solver': ['liblinear', 'saga']}
            ),  

            "LinearDiscriminantAnalysis": (
                LinearDiscriminantAnalysis(),
                {'clf__solver': ['svd', 'lsqr', 'eigen']}  
            ),

            "QuadraticDiscriminantAnalysis": (
                QuadraticDiscriminantAnalysis(),
                {'clf__reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}  
            ),

            "NaiveBayes": (
                GaussianNB(),
                {'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]} 
            ),
        }
        

    def get_grid_search(self):

        models = []

        for name in self.models_dict:

            model, param_grid = self.models_dict[name]

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])

            combined_param_grid = {
                **param_grid
            }

            grid_search = GridSearchCV (pipeline, combined_param_grid, n_jobs = -1, cv= 5)

            models.append ((name, grid_search))

        return models
