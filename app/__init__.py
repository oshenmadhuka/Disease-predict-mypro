from flask import Flask 
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register the routes blueprint
    from .routes import main  # Ensure that the relative import uses ".routes"
    app.register_blueprint(main)
    return app
