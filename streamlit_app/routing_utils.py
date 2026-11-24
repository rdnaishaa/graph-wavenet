import os
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)


TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")


# ============================================================
# CHECK API KEY
# ============================================================


def check_tomtom_api_key():

    if not TOMTOM_API_KEY:
        raise RuntimeError(
            "TOMTOM_API_KEY tidak ditemukan. Pastikan file .env berisi TOMTOM_API_KEY."
        )

    return True


# ============================================================
# ROUTE REQUEST
# ============================================================


def get_tomtom_route(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    traffic=True,
):

    check_tomtom_api_key()

    url = (
        "https://api.tomtom.com/"
        "routing/1/"
        "calculateRoute/"
        f"{start_lat},{start_lon}:"
        f"{end_lat},{end_lon}/"
        "json"
    )

    response = requests.get(
        url,
        params={
            "key": TOMTOM_API_KEY,
            "traffic": str(bool(traffic)).lower(),
            "routeType": "fastest",
            "travelMode": "car",
            "computeTravelTimeFor": "all",
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if "routes" not in data or not data["routes"]:
        raise ValueError("TomTom tidak mengembalikan route.")

    route = data["routes"][0]

    # ========================================================
    # ROUTE GEOMETRY
    # ========================================================

    points = []

    for leg in route["legs"]:
        for point in leg["points"]:
            points.append(
                {
                    "latitude": float(point["latitude"]),
                    "longitude": float(point["longitude"]),
                }
            )

    if not points:
        raise ValueError("Geometry route kosong.")

    route_df = pd.DataFrame(points)

    # ========================================================
    # ROUTE SUMMARY
    # ========================================================

    summary = route.get("summary", {})

    distance_m = float(summary.get("lengthInMeters", 0))

    travel_time_s = float(summary.get("travelTimeInSeconds", 0))

    traffic_delay_s = float(summary.get("trafficDelayInSeconds", 0))

    no_traffic_time_s = float(summary.get("noTrafficTravelTimeInSeconds", 0))

    return {
        "route_df": route_df,
        "distance_m": distance_m,
        "distance_km": (distance_m / 1000.0),
        "travel_time_seconds": travel_time_s,
        "travel_time_minutes": (travel_time_s / 60.0),
        "traffic_delay_seconds": traffic_delay_s,
        "traffic_delay_minutes": (traffic_delay_s / 60.0),
        "no_traffic_time_seconds": no_traffic_time_s,
        "no_traffic_time_minutes": (no_traffic_time_s / 60.0),
        "raw_summary": summary,
    }


# ============================================================
# ROUTE FROM CORRIDOR RECORDS
# ============================================================


def get_route_between_corridors(
    origin_record,
    destination_record,
    traffic=True,
):

    return get_tomtom_route(
        start_lat=float(origin_record["latitude"]),
        start_lon=float(origin_record["longitude"]),
        end_lat=float(destination_record["latitude"]),
        end_lon=float(destination_record["longitude"]),
        traffic=traffic,
    )


def get_alternative_routes(origin, destination):
    """Request up to three road geometries, without live traffic assumptions."""
    check_tomtom_api_key()
    url = ('https://api.tomtom.com/routing/1/calculateRoute/'
           f"{float(origin['latitude'])},{float(origin['longitude'])}:"
           f"{float(destination['latitude'])},{float(destination['longitude'])}/json")
    # Retry temporary network/server failures; never expose the URL containing the key.
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504],
            allowed_methods=['GET'], respect_retry_after_header=False)))
        try:
            response = session.get(url, params={
                'key': TOMTOM_API_KEY, 'traffic': 'false', 'routeType': 'fastest',
                'travelMode': 'car', 'maxAlternatives': 2, 'instructionsType': 'text',
                'language': 'en-GB',
            }, timeout=(10, 30))
        except requests.Timeout:
            raise RuntimeError('TomTom terlalu lama merespons. Coba hitung perjalanan lagi.') from None
        except requests.RequestException:
            raise RuntimeError('Koneksi ke TomTom gagal. Periksa internet lalu coba lagi.') from None
    if response.status_code in (401, 403):
        raise RuntimeError('Akses routing TomTom ditolak. Periksa API key dan izin Routing API.')
    if response.status_code == 429:
        raise RuntimeError('Batas permintaan TomTom tercapai. Tunggu sebentar lalu coba lagi.')
    if not response.ok:
        raise RuntimeError(f'TomTom gagal mencari rute (HTTP {response.status_code}). Coba lagi atau pilih tujuan lain.')
    result = []
    for route in response.json().get('routes', []):
        points = [p for leg in route['legs'] for p in leg['points']]
        streets = list(dict.fromkeys(i.get('street', '') for i in
            route.get('guidance', {}).get('instructions', []) if i.get('street')))
        result.append({'route_df': pd.DataFrame(points),
                       'distance_km': route['summary']['lengthInMeters'] / 1000,
                       'streets': streets})
    if not result:
        raise ValueError('Tidak ada alternatif rute yang tersedia.')
    return result
