from datetime import datetime
import requests
from flask import Blueprint, jsonify, current_app, request
from .. import database as db
from ..models import ImageUpload, Prediction
from ml import predict as ml_predict

bp = Blueprint("predict", __name__)


ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/jpg"}  # ajuste se quiser mais


def _download_image(url: str, timeout: int = 8) -> bytes:
    """Baixa a imagem e retorna bytes. Lança exceção em erro."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "")
    if not any(mt in content_type for mt in ("image/",)):
        raise ValueError(f"URL does not point to an image (Content-Type={content_type})")
    return r.content


@bp.route("/image/<int:image_id>", methods=["POST"])
def predict_image(image_id):
    """
    POST /predict/image/<image_id>
    - Se mandar multipart/form-data com campo 'image' usa o arquivo enviado.
    - Se mandar JSON {"image_url": "..."} baixa a imagem.
    - Caso nenhum dos dois seja enviado, usa o ImageUpload.filepath do DB (se existir).
    Retorna JSON com resultado e grava Prediction no banco (sem geom).
    """
    # 1) Busca o registro de imagem (se existir) — não é obrigatório para aceitar upload,
    #    mas mantemos compatibilidade com seu fluxo que usa image_id
    img = ImageUpload.query.get(image_id)
    if not img:
        return jsonify({"error": "image not found"}), 404

    # 2) Obter bytes da imagem: preferência por upload -> image_url -> arquivo salvo
    image_bytes = None
    try:
        # upload via form-data
        if "image" in request.files:
            f = request.files["image"]
            # valida mimetype
            if f.mimetype and f.mimetype not in ALLOWED_MIMETYPES:
                return jsonify({"error": f"unsupported image type: {f.mimetype}"}), 400
            image_bytes = f.read()

        else:
            # JSON body com image_url?
            j = request.get_json(silent=True) or {}
            image_url = j.get("image_url")
            if image_url:
                try:
                    image_bytes = _download_image(image_url)
                except Exception as ex:
                    current_app.logger.exception("Failed to download image_url")
                    return jsonify({"error": f"failed to download image_url: {ex}"}), 400

        # se ainda não temos bytes, tentar ler do disco (ImageUpload.filepath)
        if image_bytes is None:
            try:
                with open(img.filepath, "rb") as fh:
                    image_bytes = fh.read()
            except Exception:
                current_app.logger.exception("Failed to open image file from disk")
                return jsonify({"error": "failed to open image file"}), 500

    except Exception:
        current_app.logger.exception("Error reading image content")
        return jsonify({"error": "error reading image content"}), 500

    # 3) Chama o serviço de ML
    try:
        ml_res = ml_predict.predict_from_bytes(image_bytes)
    except Exception:
        current_app.logger.exception("ML prediction failed")
        return jsonify({"error": "prediction failed"}), 500

    # Normaliza nomes esperados (compatibilidade)
    class_name = ml_res.get("class_name") or ml_res.get("pred_class") or ml_res.get("class")
    class_pretty = ml_res.get("class_pretty") or ml_res.get("class_label") or class_name
    probs = ml_res.get("probs") or ml_res.get("probabilities") or []
    top_prob = ml_res.get("prob") or (max(probs) if probs else None)
    percentage = ml_res.get("percentage")
    severity_label = ml_res.get("severity_label") or ml_res.get("bucket")

    # 4) Persistir Prediction (assumindo meta_info é coluna JSON/JSONB)
    try:
        meta = {
            "method": "ml-model",
            "image_id": img.id,
            "class_name": class_name,
            "class_pretty": class_pretty,
            "probs": probs,
            "percentage": percentage,
            "severity_label": severity_label,
        }

        pred = Prediction(
            prob=float(top_prob) if top_prob is not None else None,
            model_version=str(ml_res.get("model_version") or ""),
            meta_info=meta,  # se sua coluna for db.JSON/JSONB, isso grava como JSON
            created_at=datetime.utcnow(),
        )
        db.session.add(pred)
        db.session.commit()
    except Exception:
        current_app.logger.exception("Failed to save prediction")
        db.session.rollback()
        return jsonify({"error": "failed to save prediction"}), 500

    # 5) Resposta ao cliente (detalhada)
    resp = {
        "image_id": img.id,
        "prediction_id": pred.id,
        "class_name": class_name,
        "class_pretty": class_pretty,
        "prob": pred.prob,
        "percentage": percentage,
        "severity_label": severity_label,
        "probs": probs,
        "model_version": pred.model_version,
        "created_at": pred.created_at.isoformat() if pred.created_at else None,
    }
    return jsonify(resp), 201
