from flask import Blueprint, request, jsonify, current_app
from .. import database as db
from ..models import ImageUpload, Prediction
from datetime import datetime
from sqlalchemy import func, cast, String

bp = Blueprint('images', __name__)


@bp.route('', methods=['GET'])
def list_images():
    """
    GET /images
    Query params:
      - page (int, default 1)
      - per_page (int, default 10)
      - name (str) : substring search on filename (ILIKE)
      - date (YYYY-MM-DD) : filter by created_at date
    Response: paginated list of images and (when exists) latest prediction metadata.
    NOTE: intentionally does NOT select geom from predictions (avoids ST_AsEWKB issues).
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    name_filter = request.args.get('name', type=str)
    date_filter = request.args.get('date', type=str)

    query = ImageUpload.query

    if name_filter:
        # busca case-insensitive por filename
        query = query.filter(ImageUpload.filename.ilike(f"%{name_filter}%"))

    if date_filter:
        try:
            d = datetime.strptime(date_filter, "%Y-%m-%d").date()
            # supondo campo created_at
            query = query.filter(func.date(ImageUpload.created_at) == d)
        except ValueError:
            return jsonify({'error': 'Formato de data inválido. Use YYYY-MM-DD'}), 400

    pagination = query.order_by(ImageUpload.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    items = []
    for img in pagination.items:
        # buscar a predição mais recente associada a essa imagem, mas selecionar APENAS campos não-geom
        try:
            pred_row = (
                db.session.query(
                    Prediction.id,
                    Prediction.prob,
                    Prediction.model_version,
                    Prediction.meta_info,
                    Prediction.created_at
                )
                .filter(cast(Prediction.meta_info['image_id'], String) == str(img.id))
                .order_by(Prediction.created_at.desc())
                .first()
            )
        except Exception as e:
            current_app.logger.debug(f"Erro ao consultar prediction meta_info: {e}")
            pred_row = None

        pred_info = None
        if pred_row:
            pred_info = {
                'id': pred_row.id,
                'prob': float(pred_row.prob) if pred_row.prob is not None else None,
                'model_version': pred_row.model_version,
                'meta_info': pred_row.meta_info,
                'created_at': pred_row.created_at.isoformat() if pred_row.created_at else None
            }

        items.append({
            'id': img.id,
            'filename': img.filename,
            'filepath': img.filepath,
            'lat': img.lat,
            'lon': img.lon,
            'exif': img.exif,
            'meta_info': img.meta_info,
            'created_at': img.created_at.isoformat() if img.created_at else None,
            'prediction': pred_info
        })

    return jsonify({
        'page': pagination.page,
        'per_page': pagination.per_page,
        'total': pagination.total,
        'pages': pagination.pages,
        'items': items
    }), 200


@bp.route('/images/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """
    Retorna os detalhes de uma imagem (por id), incluindo predição se existir.
    """
    img = ImageUpload.query.get(image_id)
    if not img:
        return jsonify({'error': 'Image not found'}), 404

    try:
        pred_row = (
            db.session.query(
                Prediction.id,
                Prediction.prob,
                Prediction.model_version,
                Prediction.meta_info,
                Prediction.created_at
            )
            .filter(cast(Prediction.meta_info['image_id'], String) == str(img.id))
            .order_by(Prediction.created_at.desc())
            .first()
        )
    except Exception as e:
        current_app.logger.debug(f"Erro ao consultar prediction meta_info: {e}")
        pred_row = None

    pred_info = None
    if pred_row:
        pred_info = {
            'id': pred_row.id,
            'prob': float(pred_row.prob) if pred_row.prob is not None else None,
            'model_version': pred_row.model_version,
            'meta_info': pred_row.meta_info,
            'created_at': pred_row.created_at.isoformat() if pred_row.created_at else None
        }

    return jsonify({
        'id': img.id,
        'filename': img.filename,
        'filepath': img.filepath,
        'lat': img.lat,
        'lon': img.lon,
        'exif': img.exif,
        'meta_info': img.meta_info,
        'created_at': img.created_at.isoformat() if img.created_at else None,
        'prediction': pred_info
    }), 200
