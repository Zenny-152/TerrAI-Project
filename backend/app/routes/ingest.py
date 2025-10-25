from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from shapely.geometry import shape, MultiPolygon
import os
from datetime import datetime
from PIL import Image
import exifread

from .. import database as db
from ..models import SlideEvent, ImageUpload, Prediction
from ..utils.geo import buffer_in_meters
from geoalchemy2.shape import from_shape


bp = Blueprint('ingest', __name__)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(PROJECT_ROOT, "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# limites básicos
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}
ALLOWED_GEOM_TYPES = {'Polygon', 'MultiPolygon'}
MAX_DEGREE_SPAN = 2.0 # maximum allowed bbox span in degrees (to block huge geometries)
ALLOWED_SEVERITIES = {'low', 'moderate', 'high', 'critical', 'unknown'}



def _parse_event_date(date_str):
    if not date_str:
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except Exception:
            continue
    # last resort: try ISO8601 parse from datetime
    try:
        return datetime.date.fromisoformat(date_str)
    except Exception:
        return None




def _to_float_ratio(r):
    # r é um exifread.utils.Ratio ou similar
    try:
        return float(r.num) / float(r.den)
    except Exception:
        try:
            return float(r)
        except Exception:
            return 0.0
    
def _dms_to_deg(dms):
    # dms é um Sequence de 3 Ratios
    d = _to_float_ratio(dms[0])
    m = _to_float_ratio(dms[1])
    s = _to_float_ratio(dms[2])
    return d + m/60.0 + s/3600.0

def _is_allowed_filename(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def _extract_latlon_from_tags(tags):
    """
    Recebe o dict retornado por exifread.process_file (tags) e tenta extrair lat/lon.
    Retorna (lat, lon) ou (None, None).
    """
    try:
        lat_tag = tags.get('GPS GPSLatitude')
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_tag = tags.get('GPS GPSLongitude')
        lon_ref = tags.get('GPS GPSLongitudeRef')
        if lat_tag and lon_tag:
            lat = _dms_to_deg(lat_tag.values)
            lon = _dms_to_deg(lon_tag.values)
            if lat_ref and str(lat_ref).upper().strip() == 'S':
                lat = -lat
            if lon_ref and str(lon_ref).upper().strip() == 'W':
                lon = -lon
            return lat, lon
    except Exception as e:
        current_app.logger.debug(f"_extract_latlon_from_tags failed: {e}")
    return None, None

def _sanitize_exif_tags(tags, max_len=2000):
    """
    Gera um dicionário JSON-safe a partir das tags retornadas por exifread.
    - Remove tags com probabilidade de conter binários (thumbnails, maker notes)
    - Converte valores para strings unicode (com fallback)
    - Trunca valores muito longos
    """
    if not tags:
        return None

    out = {}
    # fragmentos de chave que normalmente indicam dados binários/preview
    skip_key_fragments = ('JPEGThumbnail', 'TIFFThumbnail', 'EXIF MakerNote', 'MakerNote', 'Thumbnail', 'PreviewImage')

    for k, v in tags.items():
        key = str(k)
        if any(fragment in key for fragment in skip_key_fragments):
            current_app.logger.debug(f"Skipping EXIF tag (binary/thumbnail): {key}")
            continue

        # stringify the value safely
        try:
            if isinstance(v, (bytes, bytearray)):
                try:
                    sval = v.decode('utf-8')
                except Exception:
                    sval = v.decode('latin-1', errors='replace')
            else:
                # exifread values often have a .printable or are already readable
                sval = str(v)
        except Exception as e:
            current_app.logger.debug(f"Failed to stringify EXIF tag {key}: {e}")
            continue

        if not sval:
            continue

        # truncate very large values
        if len(sval) > max_len:
            sval = sval[:max_len] + '...[truncated]'

        # remove control/unprintable characters (keep valid Unicode)
        sval = ''.join(ch if (32 <= ord(ch) <= 0x10FFFF) else '?' for ch in sval)

        out[key] = sval

    return out if out else None

@bp.route('/image', methods=['POST'])
def ingest_image():
    """
    Recebe multipart/form-data:
      - file: arquivo de imagem (obrigatório)
      - lat (opcional): latitude float (fallback se não houver EXIF)
      - lon (opcional): longitude float
    Retorna JSON com image_id e (se disponível) prediction com polígono buffer.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    upload = request.files['file']
    if not upload or upload.filename == '':
        return jsonify({'error': 'Nome de arquivo inválido'}), 400

    # secure filename and extension check
    safe_name = secure_filename(upload.filename)
    if not _is_allowed_filename(safe_name):
        return jsonify({'error': 'Tipo de arquivo não permitido; use jpg/png'}), 400

    # basic size check
    upload.stream.seek(0, os.SEEK_END)
    size = upload.stream.tell()
    upload.stream.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({'error': 'Arquivo muito grande (limite 10MB)'}), 400

    try:
        # build filename and save binary
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{safe_name}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(save_path, 'wb') as fh:
            fh.write(upload.stream.read())

        # read EXIF tags (as dict) and attempt to extract lat/lon from tags
        exif_dict = None
        lat = None
        lon = None
        with open(save_path, 'rb') as fh:
            try:
                fh.seek(0)
                tags = exifread.process_file(fh, details=False)
                # sanitize tags to JSON-safe dict (removes thumbnails and binary blobs)
                if tags:
                    exif_dict = _sanitize_exif_tags(tags, max_len=2000)
                # extract lat/lon if present
                lat, lon = _extract_latlon_from_tags(tags)
            except Exception as e:
                current_app.logger.debug(f"exifread failed: {e}")
                exif_dict = None
            finally:
                fh.seek(0)
                # open image with PIL to get format (and ensure it's a valid image)
                image = Image.open(fh)
                image_format = image.format

        # if form lat/lon provided, prefer them (explicit fallback)
        form_lat = request.form.get('lat', type=float)
        form_lon = request.form.get('lon', type=float)
        if form_lat is not None and form_lon is not None:
            lat, lon = float(form_lat), float(form_lon)

        # create image record (exif_dict is None or dict -> SQLAlchemy will store JSON or NULL)
        img_rec = ImageUpload(
            filename=filename,
            filepath=save_path,
            lat=lat,
            lon=lon,
            exif=exif_dict,               # None or dict (JSON-safe)
            model_prob=None,
            model_version=None,
            meta_info={'uploaded_at': datetime.utcnow().isoformat()}
        )
        db.session.add(img_rec)
        db.session.flush()  # obtém img_rec.id antes do commit

        pred_id = None
        pred_geojson = None
        # create prediction polygon buffer only if coords exist
        if lat is not None and lon is not None:
            poly = buffer_in_meters(lat, lon, radius_m=50)  # default 50m buffer
            geom_wkb = from_shape(poly, srid=4326)
            pred = Prediction(
                geom=geom_wkb,
                prob=0.0,
                model_version='v0-stub',
                metadata={'method': 'buffer', 'radius_m': 50, 'image_id': img_rec.id}
            )
            db.session.add(pred)
            db.session.commit()
            pred_id = pred.id
            pred_geojson = poly.__geo_interface__
        else:
            db.session.commit()

        # return useful info including image_id
        response = {
            "image_id": img_rec.id,
            "filename": filename,
            "format": image_format,
            "lat": lat,
            "lon": lon,
            "prediction_id": pred_id,
            "prediction_polygon": pred_geojson,
            "message": "Imagem processada com sucesso"
        }
        return jsonify(response), 201

    except Exception:
        current_app.logger.exception("Erro processando upload de imagem")
        db.session.rollback()
        return jsonify({'error': 'Erro interno ao processar imagem'}), 500




@bp.route('/event', methods=['POST'])
def ingest_event():
    """Recebe um GeoJSON Feature (Polygon ou MultiPolygon) e salva na tabela slide_events.
    Body aceito (exemplos):
    - GeoJSON Feature: {"type":"Feature","geometry":{...},"properties":{...}}
    - FeatureCollection: {"type":"FeatureCollection","features":[{...}, ...]}
    - Apenas um objeto com chave "geometry": {"geometry": {...}, "properties": {...}}


    Validações realizadas:
    - presença de geometry
    - tipo geométrico permitido (Polygon / MultiPolygon)
    - geometria não vazia e válida (tenta corrigir com buffer(0) quando possível)
    - bounding box não muito grande (proteção básica)
    - event_date parseável em formatos comuns
    - severity limitada a um conjunto controlado
    """
    content = request.get_json() or {}


    # Support FeatureCollection by taking first feature
    geom_obj = None
    props = {}


    try:
        t = content.get('type')
        if t == 'FeatureCollection':
            features = content.get('features', [])
            if not features:
                return jsonify({'error': 'FeatureCollection contains no features'}), 400
            feature = features[0]
            geom_obj = feature.get('geometry')
            props = feature.get('properties', {}) or {}
        elif t == 'Feature':
            geom_obj = content.get('geometry')
            props = content.get('properties', {}) or {}
        elif 'geometry' in content:
            geom_obj = content.get('geometry')
            props = content.get('properties', {}) or {}
        else:
            return jsonify({'error': 'no geometry found in payload'}), 400

        if not geom_obj:
            return jsonify({'error': 'geometry missing or null'}), 400


        geom_type = geom_obj.get('type')
        if geom_type not in ALLOWED_GEOM_TYPES:
            return jsonify({'error': f'geometry type {geom_type} not allowed; allowed: {ALLOWED_GEOM_TYPES}'}), 400


        # Convert to shapely geometry
        shp = shape(geom_obj)
        if shp.is_empty:
            return jsonify({'error': 'geometry is empty'}), 400


        # If geometry is MultiPolygon, pick the largest polygon to store (avoids changing DB type)
        if isinstance(shp, MultiPolygon):
            # choose largest polygon by area
            try:
                largest = max(shp.geoms, key=lambda g: g.area)
                shp = largest
            except Exception:
                return jsonify({'error': 'failed to extract polygon from MultiPolygon'}), 400


        # Try to fix invalid geometries
        if not shp.is_valid:
            try:
                fixed = shp.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    shp = fixed
                else:
                    return jsonify({'error': 'geometry invalid and could not be fixed automatically'}), 400
            except Exception:
                return jsonify({'error': 'geometry invalid and buffer-fix failed'}), 400


        # Basic bbox/span check (in degrees) to avoid huge inputs
        minx, miny, maxx, maxy = shp.bounds
        if (maxx - minx) > MAX_DEGREE_SPAN or (maxy - miny) > MAX_DEGREE_SPAN:
            return jsonify({'error': 'geometry bbox too large; please submit smaller area'}), 400


        # Parse properties
        event_date = _parse_event_date(props.get('event_date'))
        severity = props.get('severity')
        if severity:
            severity = str(severity).strip().lower()
            if severity not in ALLOWED_SEVERITIES:
                severity = 'unknown'
        else:
            severity = None


        source = props.get('source')

        # Convert shapely to WKB with SRID
        geom_wkb = from_shape(shp, srid=4326)

        # Create DB record
        ev = SlideEvent(geom=geom_wkb, event_date=event_date, severity=severity, source=source)
        db.session.add(ev)
        db.session.commit()


        return jsonify({'status': 'ok', 'id': ev.id}), 201

    except Exception as e:
        current_app.logger.exception('Error ingesting event')
        db.session.rollback()
        return jsonify({'error': str(e)}), 500