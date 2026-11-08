MODEL_OPTIONS = {
    "Hybrid Graph WaveNet": {
        "short_name": "Hybrid",
        "description": (
            "Menggabungkan graf geografis tetap "
            "dengan graf adaptif yang dipelajari "
            "langsung dari data."
        ),
        "graph_strategy": ("Fixed geographic graph + adaptive learned graph"),
        "recommended": True,
    },
    "Adaptive Graph WaveNet": {
        "short_name": "Adaptive",
        "description": (
            "Mempelajari hubungan laten antarkoridor secara otomatis selama proses training."
        ),
        "graph_strategy": "Adaptive learned graph",
        "recommended": False,
    },
    "Fixed Graph WaveNet": {
        "short_name": "Fixed",
        "description": (
            "Menggunakan hubungan geografis antarkoridor yang telah ditentukan sebelum training."
        ),
        "graph_strategy": "Fixed geographic graph",
        "recommended": False,
    },
}


def get_model_names():

    return list(MODEL_OPTIONS.keys())


def get_model_info(model_name):

    return MODEL_OPTIONS[model_name]
