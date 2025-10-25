from flask import Blueprint, jsonify, current_app
from .. import database as db
from ..models import ImageUpload, Prediction
from ml import predict as ml_predict
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

bp = Blueprint("predict", __name__)


@bp.route("/image/<int:image_id>", methods=["GET", "POST"])
def predict_image(image_id):
    """
    Executa inferência para uma imagem já armazenada.
    Método GET/POST (uso para testes).
    """
    img = ImageUpload.query.get(image_id)
    if not img:
        return jsonify({"error": "image not found"}), 404

    # lê arquivo do disco
    try:
        with open(img.filepath, "rb") as fh:
            image_bytes = fh.read()
    except Exception:
        current_app.logger.exception("Failed to open image file")
        return jsonify({"error": "failed to open image file"}), 500

    # chama o ML
    try:
        res = ml_predict.predict_from_bytes(image_bytes)
        probs = res.get("probs")              # lista [p_low,p_med,p_high,p_extreme]
        pred_class = res.get("pred_class")
        percentage = res.get("percentage")
        model_version = res.get("model_version")

        # prob: podemos gravar a probabilidade da classe prevista (ou o maior prob)
        prob_value = None
        if probs:
            prob_value = float(max(probs))

        pred = Prediction(
            prob = prob_value,
            model_version = model_version,
            meta_info = {"method":"ml-resnet", "image_id": img.id,
                        "probs": probs, "pred_class": pred_class, "percentage": percentage},
            created_at = datetime.utcnow()
        )
    except Exception:
        current_app.logger.exception("ML prediction failed")
        return jsonify({"error": "prediction failed"}), 500


    # Salva Prediction sem coluna geom (guardamos qualquer polígono como JSON em meta_info)
    try:
        pred_kwargs = {
            "prob": res.get("prob"),
            "model_version": res.get("model_version"),
            "meta_info": {
                "method": "ml-stub",
                "image_id": img.id,
                "percentage": res.get("percentage"),
                # se o seu ML retornar bbox/polygon, coloque aqui também:
                # "ml_polygon_geojson": res.get("polygon_geojson")
            },
            "created_at": datetime.utcnow()
        }

        pred = Prediction(**pred_kwargs)
        db.session.add(pred)
        db.session.commit()
    except Exception:
        current_app.logger.exception("Failed to save prediction")
        db.session.rollback()
        return jsonify({"error": "failed to save prediction"}), 500

    return jsonify({
        "image_id": img.id,
        "prediction_id": pred.id,
        "prob": pred.prob,
        "percentage": pred.meta_info.get("percentage") if pred.meta_info else None,
        "model_version": pred.model_version
    }), 201