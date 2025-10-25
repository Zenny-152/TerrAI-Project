from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

# eventos do SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

database = SQLAlchemy()


def create_app():

    load_dotenv(encoding='utf-8')

    app = Flask(__name__)

    # monta a string de conexão com base no .env
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")


    user_enc = quote_plus(user)
    password_enc = quote_plus(password)

    uri = f"postgresql+psycopg2://{user_enc}:{password_enc}@{host}:{port}/{name}"
    app.config["SQLALCHEMY_DATABASE_URI"] = uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "devkey")

    database.init_app(app)

    # === Ajuste do search_path por conexão (Option B) ===
    # Coloque public antes do seu schema pra garantir resolução correta das funções PostGIS.
    @event.listens_for(Engine, "connect")
    def _set_search_path(dbapi_conn, connection_record):
        try:
            cur = dbapi_conn.cursor()
            cur.execute('SET search_path TO public, terrai_db, "$user"')
            cur.close()
        except Exception:
            # não interromper a aplicação se, por algum motivo, o SET falhar
            try:
                cur.close()
            except Exception:
                pass

    # Blueprints
    from .routes.health import bp as health_bp
    from .routes.ingest import bp as ingest_bp
    from .routes.predict import bp as predict_bp
    from .routes.images import bp as images_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(ingest_bp, url_prefix='/ingest')
    app.register_blueprint(predict_bp, url_prefix='/predict')
    app.register_blueprint(images_bp, url_prefix='/images')

    return app


# expose app for gunicorn
app = create_app()