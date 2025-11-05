from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nids.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from .auth import auth as auth_blueprint
    from .dashboard import dashboard as dashboard_blueprint
    from .predict import predict as predict_blueprint
    from .model_performance import model_performance as model_performance_blueprint
    from .history import history as history_blueprint
    from .profile import profile as profile_blueprint

    app.register_blueprint(auth_blueprint)
    app.register_blueprint(dashboard_blueprint)
    app.register_blueprint(predict_blueprint)
    app.register_blueprint(model_performance_blueprint)
    app.register_blueprint(history_blueprint)
    app.register_blueprint(profile_blueprint)

    return app
