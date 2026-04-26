import torch
import torch.nn as nn
from torch.nn.functional import relu
import os
import torch.nn.utils.weight_norm as weightNorm
from functools import partial
from torch.nn import functional as F

class TemporalSpatialBackbone(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.head_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        self.lhand_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        self.rhand_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        self.lfoot_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
        self.rfoot_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
        self.pelvis_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
        self.lhand_inhead_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        self.rhand_inhead_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])

        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        
        self.num_block = block_num
        num_rnn_layer = 1
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([torch.nn.LSTM(hidden_size, hidden_size, num_rnn_layer, 
            bidirectional=False, batch_first=True) for _ in range(8)]) for _ in range(self.num_block)]
        )
   
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer, num_layers=number_layer) for _ in range(self.num_block)]
        )

        # weight norm and initialization
        for encoder_i in self.time_encoder:
            for model_i in encoder_i:
                for layeridx in range(num_rnn_layer):
                    model_i = weightNorm( model_i, f"weight_ih_l{layeridx}"  )
                    model_i = weightNorm( model_i, f"weight_hh_l{layeridx}"  )
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)
    
    def forward(self, x_in, rnn_state=None):
        # separate the input features
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        head_feats = [x_in[..., 0:6], x_in[..., 36:42], x_in[..., 72:75], x_in[..., 81:84]]
        lhand_feats = [x_in[..., 6:12], x_in[..., 42:48], x_in[..., 75:78], x_in[..., 84:87]]
        rhand_feats = [x_in[..., 12:18], x_in[..., 48:54], x_in[..., 78:81], x_in[..., 87:90]]
        lfoot_feats = [x_in[..., 18:24], x_in[..., 54:60], x_in[..., 126:129]]
        rfoot_feats = [x_in[..., 24:30], x_in[..., 60:66], x_in[..., 129:132]]
        pelvis_feats = [x_in[..., 30:36], x_in[..., 66:72], x_in[..., 132:135]]
        lhand_inhead_feats = [x_in[..., 90:96], x_in[..., 102:108], x_in[..., 114:117], x_in[..., 120:123]]
        rhand_inhead_feats = [x_in[..., 96:102], x_in[..., 108:114], x_in[..., 117:120], x_in[..., 123:126]]
        
        # MLP embedding
        head_emb = []
        for idx in range(len(head_feats)):
            head_emb.append(self.head_linear_embeddings[idx](head_feats[idx]))
        head_emb = self.norm(torch.cat(head_emb, dim=-1))
        
        lhand_emb = []
        for idx in range(len(lhand_feats)):
            lhand_emb.append(self.lhand_linear_embeddings[idx](lhand_feats[idx]))
        lhand_emb = self.norm(torch.cat(lhand_emb, dim=-1))

        rhand_emb = []
        for idx in range(len(rhand_feats)):
            rhand_emb.append(self.rhand_linear_embeddings[idx](rhand_feats[idx]))
        rhand_emb = self.norm(torch.cat(rhand_emb, dim=-1))

        lfoot_emb = []
        for idx in range(len(lfoot_feats)):
            lfoot_emb.append(self.lfoot_linear_embeddings[idx](lfoot_feats[idx]))
        lfoot_emb = self.norm(torch.cat(lfoot_emb, dim=-1))

        rfoot_emb = []
        for idx in range(len(rfoot_feats)):
            rfoot_emb.append(self.rfoot_linear_embeddings[idx](rfoot_feats[idx]))
        rfoot_emb = self.norm(torch.cat(rfoot_emb, dim=-1))

        pelvis_emb = []
        for idx in range(len(pelvis_feats)):
            pelvis_emb.append(self.pelvis_linear_embeddings[idx](pelvis_feats[idx]))
        pelvis_emb = self.norm(torch.cat(pelvis_emb, dim=-1))

        lhand_inhead_emb = []
        for idx in range(len(lhand_inhead_feats)):
            lhand_inhead_emb.append(self.lhand_inhead_linear_embeddings[idx](lhand_inhead_feats[idx]))
        lhand_inhead_emb = self.norm(torch.cat(lhand_inhead_emb, dim=-1))

        rhand_inhead_emb = []
        for idx in range(len(rhand_inhead_feats)):
            rhand_inhead_emb.append(self.rhand_inhead_linear_embeddings[idx](rhand_inhead_feats[idx]))
        rhand_inhead_emb = self.norm(torch.cat(rhand_inhead_emb, dim=-1))

        collect_feats = torch.stack([head_emb, lhand_emb, rhand_emb, lfoot_emb, rfoot_emb, pelvis_emb,
            lhand_inhead_emb, rhand_inhead_emb], dim=-2).reshape(batch_size, time_seq, 8, -1)
        
        # temporal and spatial backbone
        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(8):
                collect_feats_temporal.append(self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :], None)[0])
            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)
            collect_feats = self.spatial_encoder[idx](collect_feats_temporal.reshape(batch_size*time_seq, 8, -1)).reshape(batch_size, time_seq, 8, -1)
        return collect_feats

class HMD_imu_HME_Universe(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = TemporalSpatialBackbone(input_dim, number_layer, hidden_size, dropout, nhead, block_num)

        self.pose_est = nn.Sequential(
                                nn.Linear(hidden_size*8, 256),
                                nn.LeakyReLU(),
                                nn.Linear(256, 22 * 6)
            )

        self.shape_est = nn.Sequential(
                            nn.Linear(hidden_size*8, 256),
                            nn.LeakyReLU(),
                            nn.Linear(256, 16)
            )


    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)

        pred_pose = self.pose_est(collect_feats)
        pred_shapes = self.shape_est(collect_feats)

        return pred_pose, pred_shapes


# ---------------------------------------------------------------------------
# Ablation variant backbones
# ---------------------------------------------------------------------------

def _build_embedding_layers(hidden_size):
    """Construct the 8 per-sensor MLP embedding layer groups.
    Extracted so all backbone variants can share the exact same construction."""
    head = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                          nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                          nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                          nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
    lhand = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
    rhand = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
    lfoot = nn.ModuleList([nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
    rfoot = nn.ModuleList([nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                           nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
    pelvis = nn.ModuleList([nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                            nn.Sequential(nn.Linear(6, (hidden_size-16)//2), nn.LeakyReLU()),
                            nn.Sequential(nn.Linear(3, 16), nn.LeakyReLU())])
    lhand_inhead = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                  nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                  nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                  nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
    rhand_inhead = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                  nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
                                  nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
                                  nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
    return head, lhand, rhand, lfoot, rfoot, pelvis, lhand_inhead, rhand_inhead


def _embed_sensors(x_in, embeddings, norm):
    """Run the shared sensor embedding forward pass.
    Returns collect_feats of shape (B, T, 8, hidden_size)."""
    batch_size, time_seq = x_in.shape[0], x_in.shape[1]
    head_emb, lhand_emb, rhand_emb, lfoot_emb, rfoot_emb, pelvis_emb, lhand_inhead_emb, rhand_inhead_emb = embeddings

    head_feats        = [x_in[..., 0:6],    x_in[..., 36:42], x_in[..., 72:75],  x_in[..., 81:84]]
    lhand_feats       = [x_in[..., 6:12],   x_in[..., 42:48], x_in[..., 75:78],  x_in[..., 84:87]]
    rhand_feats       = [x_in[..., 12:18],  x_in[..., 48:54], x_in[..., 78:81],  x_in[..., 87:90]]
    lfoot_feats       = [x_in[..., 18:24],  x_in[..., 54:60], x_in[..., 126:129]]
    rfoot_feats       = [x_in[..., 24:30],  x_in[..., 60:66], x_in[..., 129:132]]
    pelvis_feats      = [x_in[..., 30:36],  x_in[..., 66:72], x_in[..., 132:135]]
    lhand_inhead_feats= [x_in[..., 90:96],  x_in[..., 102:108], x_in[..., 114:117], x_in[..., 120:123]]
    rhand_inhead_feats= [x_in[..., 96:102], x_in[..., 108:114], x_in[..., 117:120], x_in[..., 123:126]]

    def embed(layers, feats):
        return norm(torch.cat([layers[i](feats[i]) for i in range(len(feats))], dim=-1))

    h  = embed(head_emb,        head_feats)
    lh = embed(lhand_emb,       lhand_feats)
    rh = embed(rhand_emb,       rhand_feats)
    lf = embed(lfoot_emb,       lfoot_feats)
    rf = embed(rfoot_emb,       rfoot_feats)
    p  = embed(pelvis_emb,      pelvis_feats)
    li = embed(lhand_inhead_emb, lhand_inhead_feats)
    ri = embed(rhand_inhead_emb, rhand_inhead_feats)

    collect_feats = torch.stack([h, lh, rh, lf, rf, p, li, ri], dim=-2).reshape(batch_size, time_seq, 8, -1)
    return collect_feats


class CausalTransformerTemporalSpatialBackbone(torch.nn.Module):
    """Ablation variant: replace per-sensor LSTM with a causal (masked) Transformer.
    All other components (embedding, spatial Transformer) are identical to the baseline."""

    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        embs = _build_embedding_layers(hidden_size)
        (self.head_linear_embeddings, self.lhand_linear_embeddings,
         self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
         self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
         self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings) = embs

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        self.num_block = block_num

        # Causal Transformer temporal module (1 layer per sensor, dim_feedforward=hidden_size
        # to keep parameter count comparable to the single-layer unidirectional LSTM)
        self.time_encoder = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        hidden_size, nhead=nhead,
                        dim_feedforward=hidden_size,
                        batch_first=True
                    ),
                    num_layers=1
                )
                for _ in range(8)
            ])
            for _ in range(self.num_block)
        ])

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer, num_layers=number_layer) for _ in range(self.num_block)]
        )

    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        embeddings = (self.head_linear_embeddings, self.lhand_linear_embeddings,
                      self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
                      self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
                      self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings)
        collect_feats = _embed_sensors(x_in, embeddings, self.norm)

        # Lower-triangular causal mask: position i may only attend to positions 0..i
        causal_mask = torch.triu(
            torch.full((time_seq, time_seq), float('-inf'), device=x_in.device), diagonal=1
        )

        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(8):
                out = self.time_encoder[idx][idx_num](
                    collect_feats[:, :, idx_num, :],
                    mask=causal_mask
                )
                collect_feats_temporal.append(out)
            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)
            collect_feats = self.spatial_encoder[idx](
                collect_feats_temporal.reshape(batch_size * time_seq, 8, -1)
            ).reshape(batch_size, time_seq, 8, -1)

        return collect_feats


class SpatialOnlyBackbone(torch.nn.Module):
    """Ablation variant: remove temporal (LSTM) module, keep spatial Transformer only.
    Each timestep is processed independently — no temporal context."""

    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        embs = _build_embedding_layers(hidden_size)
        (self.head_linear_embeddings, self.lhand_linear_embeddings,
         self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
         self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
         self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings) = embs

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        self.num_block = block_num

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer, num_layers=number_layer) for _ in range(self.num_block)]
        )

    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        embeddings = (self.head_linear_embeddings, self.lhand_linear_embeddings,
                      self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
                      self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
                      self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings)
        collect_feats = _embed_sensors(x_in, embeddings, self.norm)

        for idx in range(self.num_block):
            # No temporal step: apply spatial encoder across the 8 sensor tokens at each timestep
            collect_feats = self.spatial_encoder[idx](
                collect_feats.reshape(batch_size * time_seq, 8, -1)
            ).reshape(batch_size, time_seq, 8, -1)

        return collect_feats


class TemporalOnlyBackbone(torch.nn.Module):
    """Ablation variant: remove spatial (Transformer) module, keep temporal LSTM only.
    Sensor features are processed independently in time; no cross-sensor attention."""

    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        embs = _build_embedding_layers(hidden_size)
        (self.head_linear_embeddings, self.lhand_linear_embeddings,
         self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
         self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
         self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings) = embs

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        self.num_block = block_num

        num_rnn_layer = 1
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([torch.nn.LSTM(hidden_size, hidden_size, num_rnn_layer,
             bidirectional=False, batch_first=True) for _ in range(8)]) for _ in range(self.num_block)]
        )

        # Weight norm and orthogonal init (same as baseline)
        for encoder_i in self.time_encoder:
            for model_i in encoder_i:
                for layeridx in range(num_rnn_layer):
                    model_i = weightNorm(model_i, f"weight_ih_l{layeridx}")
                    model_i = weightNorm(model_i, f"weight_hh_l{layeridx}")
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)

    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        embeddings = (self.head_linear_embeddings, self.lhand_linear_embeddings,
                      self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
                      self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
                      self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings)
        collect_feats = _embed_sensors(x_in, embeddings, self.norm)

        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(8):
                collect_feats_temporal.append(
                    self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :], None)[0]
                )
            # No spatial encoder: just stack the per-sensor temporal outputs
            collect_feats = torch.stack(collect_feats_temporal, dim=-2)

        return collect_feats


# ---------------------------------------------------------------------------
# Capacity-matched ablation backbones
#
# Motivation: plain SpatialOnly / TemporalOnly variants drop ~8.5M params from
# the 17.4M backbone, so any performance drop could be a capacity effect rather
# than an architectural one. The variants below keep the module slot occupied
# by a parameter-matched module that deliberately does NOT mix information
# along the ablated axis (time for NoTemporalMix, sensors for NoSpatialMix).
# ---------------------------------------------------------------------------

class ResidualMLPBlock(torch.nn.Module):
    """Pre-norm 2-layer residual MLP with no cross-token mixing.

    Operates elementwise on the last dim, so when applied to a
    `(B, T, H)` slice it does no temporal mixing, and when applied to a
    `(B*T, 8, H)` slice it does no cross-sensor mixing — the caller
    chooses by controlling the shape.

    Parameter count: ``3*H + (2*H+1)*ffn_dim``.
    """

    def __init__(self, hidden_size, ffn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class NoTemporalMixBackbone(torch.nn.Module):
    """Replace per-sensor LSTM with a parameter-matched frame-wise MLP.

    The MLP is applied independently at each timestep, so there is zero
    information flow across the temporal axis. Everything else — sensor
    embedding, spatial Transformer, weight sharing across blocks — is
    identical to the baseline.

    At ``hidden_size=256`` the per-sensor LSTM has 526,336 params and
    ``ResidualMLPBlock(256, 1024)`` has 526,080 params, so the substitute
    matches the baseline within 0.05%.
    """

    # Precomputed so a YAML change to hidden_size fails loud (asserted below).
    _MLP_FFN_DIM_FOR_HIDDEN_256 = 1024

    def __init__(self, input_dim, number_layer=3, hidden_size=256,
                 dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        assert hidden_size == 256, (
            "NoTemporalMix ffn_dim is hardcoded for hidden_size=256. "
            "Recompute with d = (8*H^2 + 5*H) / (2*H + 1) to match LSTM params."
        )
        embs = _build_embedding_layers(hidden_size)
        (self.head_linear_embeddings, self.lhand_linear_embeddings,
         self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
         self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
         self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings) = embs

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        self.num_block = block_num

        ffn_dim = self._MLP_FFN_DIM_FOR_HIDDEN_256
        self.time_encoder = nn.ModuleList([
            nn.ModuleList([
                ResidualMLPBlock(hidden_size, ffn_dim) for _ in range(8)
            ])
            for _ in range(self.num_block)
        ])

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer, num_layers=number_layer) for _ in range(self.num_block)]
        )

    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        embeddings = (self.head_linear_embeddings, self.lhand_linear_embeddings,
                      self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
                      self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
                      self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings)
        collect_feats = _embed_sensors(x_in, embeddings, self.norm)

        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(8):
                # Shape (B, T, H): ResidualMLPBlock operates on the last dim,
                # so each timestep is processed independently — no temporal mixing.
                out = self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :])
                collect_feats_temporal.append(out)
            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)
            collect_feats = self.spatial_encoder[idx](
                collect_feats_temporal.reshape(batch_size * time_seq, 8, -1)
            ).reshape(batch_size, time_seq, 8, -1)

        return collect_feats


class NoSpatialMixBackbone(torch.nn.Module):
    """Replace spatial Transformer with a parameter-matched per-sensor MLP stack.

    The MLP stack is applied independently to each of the 8 sensor tokens at
    every timestep, so there is zero cross-sensor information flow.
    Everything else — sensor embedding, temporal LSTM, weight sharing across
    blocks — is identical to the baseline.

    At ``hidden_size=256`` a standard ``TransformerEncoderLayer(256, nhead=8,
    dim_feedforward=2048)`` has 1,315,072 params (MHA + FFN + 2×LayerNorm).
    ``ResidualMLPBlock(256, 2560)`` has 1,314,048 params — matches within
    0.08%. The Transformer stacks ``number_layer`` of these per block; we do
    the same.
    """

    _MLP_FFN_DIM_FOR_HIDDEN_256 = 2560

    def __init__(self, input_dim, number_layer=3, hidden_size=256,
                 dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        assert hidden_size == 256, (
            "NoSpatialMix ffn_dim is hardcoded for hidden_size=256. "
            "Recompute with d = (transformer_layer_params - 3*H) / (2*H + 1) "
            "to match one Transformer encoder layer's params."
        )
        embs = _build_embedding_layers(hidden_size)
        (self.head_linear_embeddings, self.lhand_linear_embeddings,
         self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
         self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
         self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings) = embs

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        self.num_block = block_num

        num_rnn_layer = 1
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([torch.nn.LSTM(hidden_size, hidden_size, num_rnn_layer,
             bidirectional=False, batch_first=True) for _ in range(8)]) for _ in range(self.num_block)]
        )

        # Stack `number_layer` ResidualMLPBlocks per block, plus a final
        # LayerNorm to cap activation magnitude before it flows into the next
        # block's LSTM. Without this final norm the pre-norm residual stack's
        # output is unbounded and training diverges around epoch 13 — the
        # baseline spatial Transformer's post-norm structure achieves the
        # same stabilisation implicitly.
        ffn_dim = self._MLP_FFN_DIM_FOR_HIDDEN_256
        self.spatial_encoder = nn.ModuleList([
            nn.Sequential(
                *[ResidualMLPBlock(hidden_size, ffn_dim) for _ in range(number_layer)],
                nn.LayerNorm(hidden_size),
            )
            for _ in range(self.num_block)
        ])

        # Weight norm and orthogonal init on the LSTM (same as baseline).
        for encoder_i in self.time_encoder:
            for model_i in encoder_i:
                for layeridx in range(num_rnn_layer):
                    model_i = weightNorm(model_i, f"weight_ih_l{layeridx}")
                    model_i = weightNorm(model_i, f"weight_hh_l{layeridx}")
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)

    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        embeddings = (self.head_linear_embeddings, self.lhand_linear_embeddings,
                      self.rhand_linear_embeddings, self.lfoot_linear_embeddings,
                      self.rfoot_linear_embeddings, self.pelvis_linear_embeddings,
                      self.lhand_inhead_linear_embeddings, self.rhand_inhead_linear_embeddings)
        collect_feats = _embed_sensors(x_in, embeddings, self.norm)

        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(8):
                collect_feats_temporal.append(
                    self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :], None)[0]
                )
            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)
            # Shape (B*T, 8, H): ResidualMLPBlock operates on the last dim, so
            # each of the 8 sensor tokens is processed independently — no
            # cross-sensor mixing.
            collect_feats = self.spatial_encoder[idx](
                collect_feats_temporal.reshape(batch_size * time_seq, 8, -1)
            ).reshape(batch_size, time_seq, 8, -1)

        return collect_feats


# ---------------------------------------------------------------------------
# Ablation top-level model classes (same output heads as HMD_imu_HME_Universe)
# ---------------------------------------------------------------------------

class HMD_imu_HME_TemporalTransformer(torch.nn.Module):
    """Ablation: causal Transformer temporal module + Transformer spatial module."""
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = CausalTransformerTemporalSpatialBackbone(
            input_dim, number_layer, hidden_size, dropout, nhead, block_num)
        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 22 * 6))
        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 16))

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)
        return self.pose_est(collect_feats), self.shape_est(collect_feats)


class HMD_imu_HME_SpatialOnly(torch.nn.Module):
    """Ablation: no temporal module, Transformer spatial module only."""
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = SpatialOnlyBackbone(
            input_dim, number_layer, hidden_size, dropout, nhead, block_num)
        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 22 * 6))
        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 16))

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)
        return self.pose_est(collect_feats), self.shape_est(collect_feats)


class HMD_imu_HME_TemporalOnly(torch.nn.Module):
    """Ablation: LSTM temporal module only, no spatial Transformer."""
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = TemporalOnlyBackbone(
            input_dim, number_layer, hidden_size, dropout, nhead, block_num)
        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 22 * 6))
        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 16))

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)
        return self.pose_est(collect_feats), self.shape_est(collect_feats)


class HMD_imu_HME_NoTemporalMix(torch.nn.Module):
    """Capacity-matched ablation: per-frame MLP (no temporal mixing) + spatial Transformer."""
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = NoTemporalMixBackbone(
            input_dim, number_layer, hidden_size, dropout, nhead, block_num)
        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 22 * 6))
        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 16))

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)
        return self.pose_est(collect_feats), self.shape_est(collect_feats)


class HMD_imu_HME_NoSpatialMix(torch.nn.Module):
    """Capacity-matched ablation: LSTM temporal + per-sensor MLP (no spatial mixing)."""
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 135, "invalid input dim"
        self.backbone = NoSpatialMixBackbone(
            input_dim, number_layer, hidden_size, dropout, nhead, block_num)
        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 22 * 6))
        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.LeakyReLU(), nn.Linear(256, 16))

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)
        return self.pose_est(collect_feats), self.shape_est(collect_feats)
