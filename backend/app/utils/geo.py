from shapely.geometry import Point
from shapely.ops import transform
import pyproj

def buffer_in_meters(lat, lon, radius_m=50):
    """
    Return a Shapely Polygon (in EPSG:4326 lon/lat) that is a buffer
    of radius_m meters around (lat, lon).
    """
    # create shapely point using (x=lon, y=lat)
    p = Point(lon, lat)

    # define projections
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    # azimuthal equidistant centered on the point for local meter distances
    proj_aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m +no_defs")
    to_aeqd = pyproj.Transformer.from_crs(proj_wgs84, proj_aeqd, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(proj_aeqd, proj_wgs84, always_xy=True).transform

    # project to local meters, buffer, and project back
    p_aeqd = transform(to_aeqd, p)
    buf_aeqd = p_aeqd.buffer(radius_m)
    buf_wgs84 = transform(to_wgs84, buf_aeqd)

    return buf_wgs84
