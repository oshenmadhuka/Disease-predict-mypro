from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register the routes blueprint
    from .routes import main  # Ensure that the relative import uses ".routes"
    app.register_blueprint(main)

    return app
