from streamlit_app.data_loader import (
    get_corridor_metadata,
)

# ============================================================
# GET CORRIDOR
# ============================================================


def get_corridor_by_name(corridor_name):

    df = get_corridor_metadata()

    match = df[df["corridor_name"] == corridor_name]

    if match.empty:
        raise ValueError(f"Corridor '{corridor_name}' tidak ditemukan.")

    return match.iloc[0]


# ============================================================
# MAP DATA
# ============================================================


def build_trip_map_data(
    origin_name,
    destination_name,
):

    df = get_corridor_metadata()

    map_df = df[
        [
            "node_index",
            "corridor_name",
            "latitude",
            "longitude",
        ]
    ].copy()

    map_df["status"] = "Corridor"

    map_df.loc[
        map_df["corridor_name"] == origin_name,
        "status",
    ] = "Asal"

    map_df.loc[
        map_df["corridor_name"] == destination_name,
        "status",
    ] = "Tujuan"

    return map_df


# ============================================================
# MAP CENTER
# ============================================================


def get_map_center(
    origin_name,
    destination_name,
):

    origin = get_corridor_by_name(origin_name)

    destination = get_corridor_by_name(destination_name)

    latitude = (float(origin["latitude"]) + float(destination["latitude"])) / 2

    longitude = (float(origin["longitude"]) + float(destination["longitude"])) / 2

    return {
        "latitude": latitude,
        "longitude": longitude,
    }


# ============================================================
# CORRIDOR PAIR
# ============================================================


def get_trip_nodes(
    origin_name,
    destination_name,
):

    origin = get_corridor_by_name(origin_name)

    destination = get_corridor_by_name(destination_name)

    return {
        "origin_node": int(origin["node_index"]),
        "destination_node": int(destination["node_index"]),
    }


# ============================================================
# SIMPLE CONDITION LABEL
# ============================================================


def relative_speed_condition(
    speed,
    historical_average,
):

    if historical_average is None:
        return "Data tidak tersedia"

    if historical_average == 0:
        return "Data tidak tersedia"

    ratio = speed / historical_average

    if ratio >= 1.10:
        return "Relatif lancar"

    if ratio >= 0.90:
        return "Relatif normal"

    if ratio >= 0.70:
        return "Relatif padat"

    return "Relatif sangat padat"
