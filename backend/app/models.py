# app/models.py
from . import database as db
from sqlalchemy.sql import func

class SlideEvent(db.Model):
    __tablename__ = 'slide_events'
    __table_args__ = {'schema': 'terrai_db'}

    id = db.Column(db.Integer, primary_key=True)
    event_date = db.Column(db.Date, nullable=True)
    severity = db.Column(db.String, nullable=True)
    source = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, server_default=func.now())


class Prediction(db.Model):
    __tablename__ = 'predictions'
    __table_args__ = {'schema': 'terrai_db'}

    id = db.Column(db.Integer, primary_key=True)
    prob = db.Column(db.Float, nullable=True)
    model_version = db.Column(db.String, nullable=True)
    meta_info = db.Column(db.JSON, nullable=True)  # guarda polygon geojson / image_id / etc
    created_at = db.Column(db.DateTime, server_default=func.now())


class ImageUpload(db.Model):
    __tablename__ = 'images'
    __table_args__ = {'schema': 'terrai_db'}

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, nullable=True)
    filename = db.Column(db.String, nullable=False)
    filepath = db.Column(db.String, nullable=False)
    lat = db.Column(db.Float, nullable=True)
    lon = db.Column(db.Float, nullable=True)
    exif = db.Column(db.JSON, nullable=True)
    model_prob = db.Column(db.Float, nullable=True)
    model_version = db.Column(db.String, nullable=True)
    meta_info = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, server_default=func.now())
