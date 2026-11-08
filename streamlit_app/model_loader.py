from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "outputs" / "final_64" / "fixed_graph" / "fixed_graph_64.npz"

MODEL_DIR = ROOT / "outputs" / "final_64" / "graphwavenet_audited"


# ============================================================
# GRAPH WAVENET CONFIG
# ============================================================

NUM_NODES = 64
NUM_HORIZONS = 3
INPUT_LENGTH = 12

GW_BLOCKS = 4
GW_LAYERS = 2
GW_KERNEL_SIZE = 2

GW_RESIDUAL_CHANNELS = 32
GW_DILATION_CHANNELS = 32
GW_SKIP_CHANNELS = 256
GW_END_CHANNELS = 512

GW_DIFFUSION_ORDER = 2
GW_ADAPTIVE_EMBED_DIM = 10
GW_DROPOUT = 0.3


# ============================================================
# RECEPTIVE FIELD
# ============================================================

receptive_field = 1

for _ in range(GW_BLOCKS):
    additional_scope = GW_KERNEL_SIZE - 1

    dilation = 1

    for _ in range(GW_LAYERS):
        receptive_field += additional_scope

        dilation *= 2
        additional_scope *= 2


GW_RECEPTIVE_FIELD = receptive_field


# ============================================================
# FIXED GRAPH
# ============================================================


def row_normalize_adjacency(adj):

    adj = np.asarray(adj, dtype=np.float32)

    degree = adj.sum(axis=1)

    degree_inv = np.zeros_like(degree, dtype=np.float32)

    nonzero = degree > 0

    degree_inv[nonzero] = 1.0 / degree[nonzero]

    normalized = degree_inv[:, None] * adj

    return normalized.astype(np.float32)


def load_fixed_supports():

    graph_npz = np.load(GRAPH_FILE, allow_pickle=True)

    A_directed = graph_npz["directed_knn_adjacency"].astype(np.float32)

    A_forward = row_normalize_adjacency(A_directed)

    A_reverse = row_normalize_adjacency(A_directed.T)

    return [A_forward, A_reverse]


# ============================================================
# GRAPH PROPAGATION
# ============================================================


def graph_propagate(x, support):

    return tf.einsum("btnc,nm->btmc", x, support)


# ============================================================
# DIFFUSION GRAPH CONVOLUTION
# ============================================================


class DiffusionGraphConv(tf.keras.layers.Layer):
    def __init__(self, output_channels, diffusion_order=2, dropout=0.0, **kwargs):

        super().__init__(**kwargs)

        self.output_channels = output_channels

        self.diffusion_order = diffusion_order

        self.dropout_rate = dropout

        self.projection = tf.keras.layers.Dense(output_channels)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, supports, training=False):

        diffusion_outputs = [x]

        for support in supports:
            support = tf.cast(support, x.dtype)

            x1 = graph_propagate(x, support)

            diffusion_outputs.append(x1)

            xk = x1

            for _ in range(2, self.diffusion_order + 1):
                xk = graph_propagate(xk, support)

                diffusion_outputs.append(xk)

        h = tf.concat(diffusion_outputs, axis=-1)

        h = self.projection(h)

        h = self.dropout(h, training=training)

        return h


# ============================================================
# ADAPTIVE ADJACENCY
# ============================================================


class AdaptiveAdjacency(tf.keras.layers.Layer):
    def __init__(self, num_nodes, embed_dim=10, **kwargs):

        super().__init__(**kwargs)

        self.num_nodes = num_nodes

        self.embed_dim = embed_dim

    def build(self, input_shape):

        self.nodevec1 = self.add_weight(
            name="nodevec1",
            shape=(self.num_nodes, self.embed_dim),
            initializer="random_normal",
            trainable=True,
        )

        self.nodevec2 = self.add_weight(
            name="nodevec2",
            shape=(self.embed_dim, self.num_nodes),
            initializer="random_normal",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs=None):

        logits = tf.matmul(self.nodevec1, self.nodevec2)

        logits = tf.nn.relu(logits)

        adjacency = tf.nn.softmax(logits, axis=1)

        return adjacency


# ============================================================
# GATED TEMPORAL CONVOLUTION
# ============================================================


class GatedTemporalConv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=2, dilation_rate=1, **kwargs):

        super().__init__(**kwargs)

        self.channels = channels

        self.kernel_size = kernel_size

        self.dilation_rate = dilation_rate

        self.filter_conv = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=(kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            padding="valid",
        )

        self.gate_conv = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=(kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            padding="valid",
        )

    def call(self, x):

        filter_out = self.filter_conv(x)

        filter_out = tf.nn.tanh(filter_out)

        gate_out = self.gate_conv(x)

        gate_out = tf.nn.sigmoid(gate_out)

        return filter_out * gate_out


# ============================================================
# SINGLE GRAPH WAVENET LAYER
# ============================================================


class GraphWaveNetLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        residual_channels,
        dilation_channels,
        skip_channels,
        kernel_size,
        dilation_rate,
        diffusion_order,
        dropout,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.residual_channels = residual_channels

        self.dilation_channels = dilation_channels

        self.skip_channels = skip_channels

        self.temporal = GatedTemporalConv(
            channels=dilation_channels, kernel_size=kernel_size, dilation_rate=dilation_rate
        )

        self.graph_conv = DiffusionGraphConv(
            output_channels=residual_channels, diffusion_order=diffusion_order, dropout=dropout
        )

        self.skip_conv = tf.keras.layers.Conv2D(
            filters=skip_channels, kernel_size=(1, 1), padding="valid"
        )

        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x, supports, skip=None, training=False):

        residual = x

        x = self.temporal(x)

        s = self.skip_conv(x)

        if skip is None:
            skip = s

        else:
            skip = skip[:, -tf.shape(s)[1] :, :, :]

            skip = skip + s

        x = self.graph_conv(x, supports, training=training)

        residual = residual[:, -tf.shape(x)[1] :, :, :]

        x = x + residual

        x = self.batch_norm(x, training=training)

        return (x, skip)


# ============================================================
# GRAPH WAVENET BACKBONE
# ============================================================


class GraphWaveNetBackbone(tf.keras.layers.Layer):
    def __init__(
        self,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        blocks=4,
        layers_per_block=2,
        kernel_size=2,
        diffusion_order=2,
        dropout=0.3,
        receptive_field=13,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.residual_channels = residual_channels

        self.skip_channels = skip_channels

        self.receptive_field = receptive_field

        self.start_conv = tf.keras.layers.Conv2D(
            filters=residual_channels, kernel_size=(1, 1), padding="valid"
        )

        self.gw_layers = []

        for block_idx in range(blocks):
            dilation = 1

            for layer_idx in range(layers_per_block):
                layer = GraphWaveNetLayer(
                    residual_channels=residual_channels,
                    dilation_channels=dilation_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    diffusion_order=diffusion_order,
                    dropout=dropout,
                    name=(f"gw_block_{block_idx + 1}_layer_{layer_idx + 1}"),
                )

                self.gw_layers.append(layer)

                dilation *= 2

    def call(self, x, supports, training=False):

        current_length = tf.shape(x)[1]

        pad_length = tf.maximum(self.receptive_field - current_length, 0)

        x = tf.pad(x, paddings=[[0, 0], [pad_length, 0], [0, 0], [0, 0]])

        x = self.start_conv(x)

        skip = None

        for layer in self.gw_layers:
            x, skip = layer(x, supports, skip=skip, training=training)

        return (x, skip)


# ============================================================
# OUTPUT HEAD
# ============================================================


class GraphWaveNetHead(tf.keras.layers.Layer):
    def __init__(self, end_channels, num_horizons, **kwargs):

        super().__init__(**kwargs)

        self.end_conv_1 = tf.keras.layers.Conv2D(
            filters=end_channels, kernel_size=(1, 1), padding="valid"
        )

        self.end_conv_2 = tf.keras.layers.Conv2D(
            filters=num_horizons, kernel_size=(1, 1), padding="valid"
        )

    def call(self, skip):

        x = tf.nn.relu(skip)

        x = self.end_conv_1(x)

        x = tf.nn.relu(x)

        x = self.end_conv_2(x)

        x = tf.squeeze(x, axis=1)

        x = tf.transpose(x, perm=[0, 2, 1])

        return x


# ============================================================
# FULL GRAPH WAVENET MODEL
# ============================================================


class GraphWaveNetModel(tf.keras.Model):
    def __init__(self, mode, fixed_supports, num_nodes, num_horizons, **kwargs):

        super().__init__(**kwargs)

        valid_modes = ["fixed", "adaptive", "hybrid"]

        if mode not in valid_modes:
            raise ValueError(f"mode harus salah satu dari {valid_modes}")

        self.mode = mode

        self.fixed_supports = [tf.constant(support, dtype=tf.float32) for support in fixed_supports]

        if mode in ["adaptive", "hybrid"]:
            self.adaptive_graph = AdaptiveAdjacency(
                num_nodes=num_nodes, embed_dim=GW_ADAPTIVE_EMBED_DIM, name="adaptive_adjacency"
            )

        else:
            self.adaptive_graph = None

        self.backbone = GraphWaveNetBackbone(
            residual_channels=GW_RESIDUAL_CHANNELS,
            dilation_channels=GW_DILATION_CHANNELS,
            skip_channels=GW_SKIP_CHANNELS,
            blocks=GW_BLOCKS,
            layers_per_block=GW_LAYERS,
            kernel_size=GW_KERNEL_SIZE,
            diffusion_order=GW_DIFFUSION_ORDER,
            dropout=GW_DROPOUT,
            receptive_field=GW_RECEPTIVE_FIELD,
            name="graph_wavenet_backbone",
        )

        self.head = GraphWaveNetHead(
            end_channels=GW_END_CHANNELS, num_horizons=num_horizons, name="graph_wavenet_head"
        )

    def get_supports(self):

        if self.mode == "fixed":
            return self.fixed_supports

        adaptive_support = self.adaptive_graph(tf.zeros((1, 1), dtype=tf.float32))

        if self.mode == "adaptive":
            return [adaptive_support]

        return self.fixed_supports + [adaptive_support]

    def call(self, x, training=False):

        supports = self.get_supports()

        _, skip = self.backbone(x, supports, training=training)

        prediction = self.head(skip)

        return prediction


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_FILES = {
    "fixed": MODEL_DIR / "graphwavenet_fixed_seed42.weights.h5",
    "adaptive": MODEL_DIR / "graphwavenet_adaptive_seed42.weights.h5",
    "hybrid": MODEL_DIR / "graphwavenet_hybrid_seed42.weights.h5",
}


# ============================================================
# NORMALIZE MODEL NAME
# ============================================================


def normalize_model_mode(model_name):

    value = str(model_name).strip().lower()

    if "hybrid" in value:
        return "hybrid"

    if "adaptive" in value:
        return "adaptive"

    if "fixed" in value:
        return "fixed"

    raise ValueError(f"Model tidak dikenali: {model_name}")


# ============================================================
# BUILD MODEL
# ============================================================


def build_graphwavenet(mode):

    mode = normalize_model_mode(mode)

    fixed_supports = load_fixed_supports()

    model = GraphWaveNetModel(
        mode=mode,
        fixed_supports=fixed_supports,
        num_nodes=NUM_NODES,
        num_horizons=NUM_HORIZONS,
        name=f"graph_wavenet_{mode}",
    )

    dummy_input = tf.zeros((1, INPUT_LENGTH, NUM_NODES, 1), dtype=tf.float32)

    _ = model(dummy_input, training=False)

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================


@lru_cache(maxsize=3)
def load_graphwavenet_model(mode):

    mode = normalize_model_mode(mode)

    weight_file = MODEL_FILES[mode]

    if not weight_file.exists():
        raise FileNotFoundError(f"Weight tidak ditemukan: {weight_file}")

    model = build_graphwavenet(mode)

    model.load_weights(weight_file)

    return model


# ============================================================
# MODEL STATUS
# ============================================================


def get_model_status():

    return {
        mode: {"exists": path.exists(), "path": str(path)} for mode, path in MODEL_FILES.items()
    }


# ============================================================
# ADAPTIVE GRAPH
# ============================================================


def get_learned_adjacency(mode):

    mode = normalize_model_mode(mode)

    if mode == "fixed":
        return None

    model = load_graphwavenet_model(mode)

    adjacency = model.adaptive_graph(tf.zeros((1, 1), dtype=tf.float32)).numpy()

    return adjacency
