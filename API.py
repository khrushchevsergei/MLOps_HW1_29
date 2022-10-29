import pandas as pd
import flask
from flask_restx import Api, fields, Resource
from Model_Warehouse import SaveModel, AddModel, LoadModel, FitModel, MakePrediction, DeleteModel
import json

app = flask.Flask(__name__)
api = Api(app, title='API ML model using Flask')

models_lib = {}

add_params = api.model('Observre the parameters of the added model',
                             {'model_ID': fields.Integer(description='Type: Model ID', example=1),
                              'model_class': fields.String(description='Type : Model Class',
                                                           example='GradientBoostingClassifier'),
                              'Dataset': fields.Arbitrary(description='Type: Dataset')})

fitting_the_model = api.model('Insert fitting parameters for the model',
                             {'model_ID': fields.Integer(description='Model ID', example=1),
                              'model_params': fields.Arbitrary(description=' Type: Parameters of the model',
                                                               example=json.dumps({'random_state': 10}))})

parameters_for_prediction = api.model('Insert parameters to make a prediction',
                            {'model_ID': fields.Integer(description='Type :Model ID', example=1)})

deleting_the_model = api.model('Insert parameter to delete the model',
                            {'model_ID': fields.Integer(description='Type: Model ID', example=1)})


@api.route('/models_lib')
class ModelsLibrary(Resource):
    @api.response(200, 'OK, so far so good !')
    @api.response(500, 'Server Error - Internal Server Error')
    def get(self):
        return models_lib


@api.route('/model/add')
class AddingModel(Resource):
    @api.expect(add_params)
    @api.response(200, 'OK, so far so good !')
    @api.response(400, 'Client Error - Not Such a Clever Boy / Girl')
    @api.response(500, 'Server Error - Internal Server Error')
    def post(self):
        model_ID = api.payload['model_ID']
        model_class = api.payload['model_class']
        Dataset = pd.DataFrame(json.loads(api.payload['Dataset']))
        if model_class not in ['GradientBoostingClassifier', 'LogisticRegression']:
            return 'Requested type of model is not available. Please, try: Random Forest Classifier or Logistic Regression', 400
        else:
            models_lib[model_ID] = model_ID
            AddModel(model_ID, model_class, Dataset)
            return f'Model {model_ID} addition completed', 200


@api.route('/model/fit')
class FittingModel(Resource):
    @api.expect(fitting_the_model)
    @api.response(200, 'OK, so far so good !')
    @api.response(400, 'Client Error - Not Such a Clever Boy / Girl')
    @api.response(500, 'Server Error - Internal Server Error')
    def post(self):
        model_ID = api.payload['model_ID']
        model_params = api.payload['model_params']
        if model_ID not in models_lib:
            return ' Retreiving the Model ID failed', 400
        else:
            FitModel(model_ID, model_params)
            return f'The model {model_ID} learning completed', 200


@api.route('/model/predict')
class MakingPrediction(Resource):
    @api.expect(parameters_for_prediction)
    @api.response(200, 'OK, so far so good !')
    @api.response(400, 'Client Error - Not Such a Clever Boy / Girl')
    @api.response(500, 'Server Error - Internal Server Error')
    def get(self):
        model_ID = api.payload['model_ID']
        if model_ID not in models_lib:
            return 'Model with this ID does not exist', 400
        else:
            pred = MakePrediction(model_ID)
            return json.dumps({f'The model {model_ID} has generated predicition for you': pred.astype(int).tolist()})


@api.route('/model/delete')
class DeletingModel(Resource):
    @api.expect(deleting_the_model)
    @api.response(200, 'OK, so far so good !')
    @api.response(400, 'Client Error - Not Such a Clever Boy / Girl')
    @api.response(500, 'Server Error - Internal Server Error')
    def delete(self):
        model_ID = api.payload['model_ID']
        if model_ID not in models_lib:
            return 'Can not retrieve the model ID', 400
        else:
            DeleteModel(model_ID)
            del models_lib[model_ID]
            return f'Model {model_ID} deletion completed', 200


if __name__ == '__main__':
    app.run()