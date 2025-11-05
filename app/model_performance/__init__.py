from flask import Blueprint

model_performance = Blueprint('model_performance', __name__)

from . import routes
