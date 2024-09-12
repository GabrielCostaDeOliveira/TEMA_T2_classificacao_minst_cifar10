from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class Models:
    def __init__(self):
        self.models_dict = {
            "LogisticRegression": (
                LogisticRegression(),
                {'clf__C': [10**x for x in range(-2, 4)], 'clf__penalty': ['l1', 'l2']}
            ),
        }
        

    def get_grid_search(self):

        models = []

        for name in self.models_dict:

            model, param_grid = self.models_dict[name]

            pipeline = Pipeline([
                ('clf', model)
            ])

            combined_param_grid = {
                **param_grid
            }

            grid_search = GridSearchCV (pipeline, combined_param_grid, n_jobs = -1, cv= 5)

            models.append ((name, grid_search))

        return models
