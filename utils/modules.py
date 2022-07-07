import math

import dgl.function as fn
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax

from utils.data_utils import NODE_TYPE
from utils.generation_utils import *


class Attention(nn.Module):
    """ Cross attention to entities in the decoder layer. """

    def __init__(self, args):
        super(Attention, self).__init__()
        # We only project the query
        self.num_heads = 1
        self.head_dim = args.hidden_size
        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.scaling = args.hidden_size ** (-0.5)
        self.cache_key = "ent_encoder_decoder"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            k = prev_key
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            v = prev_value
        assert k is not None and v is not None
        prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1)
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(key_padding_mask, prev_key_padding_mask,
                                   batch_size, src_len):
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            new_key_padding_mask = prev_key_padding_mask

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def forward(self, query, key, key_padding_mask, layer_state, output_attentions):
        """
        q: output of the cross attention in decoder layer
        k: entitiy embeddings
        Input shape: Time(SeqLen) x Batch x Channel
        """
        # Get here for encoder decoder cause of static_kv
        tgt_len, bsz, embed_dim = query.size()
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if key is None:
            k = v = None
        else:
            k = v = key

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_probs = F.softmax(attn_weights, dim=-1)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights


class GAT(nn.Module):
    """ Graph Attention Networks """

    def __init__(self, args):
        super(GAT, self).__init__()
        self.in_feats = args.graph_hidden_size
        self.out_feats = args.graph_hidden_size // args.attn_head
        self.num_heads = args.attn_head
        self.ffn_drop = args.ffn_drop

        self.q_proj = nn.Linear(self.in_feats, self.num_heads * self.out_feats, bias=False)
        self.k_proj = nn.Linear(self.in_feats, self.num_heads * self.out_feats, bias=False)
        self.v_proj = nn.Linear(self.in_feats, self.num_heads * self.out_feats, bias=False)

        self.attn_drop = nn.Dropout(args.attn_drop)

        self.ln1 = nn.LayerNorm(self.in_feats)
        self.ln2 = nn.LayerNorm(self.in_feats)

        self.FFN = nn.Sequential(
            nn.Linear(self.in_feats, self.in_feats * 4),
            nn.PReLU(self.in_feats * 4),
            nn.Linear(self.in_feats * 4, self.in_feats),
            nn.Dropout(args.ffn_drop)
        )

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat_c = feat.clone().detach().requires_grad_(False)
        q, k, v = self.q_proj(feat), self.k_proj(feat_c), self.v_proj(feat_c)
        q = q.view(-1, self.num_heads, self.out_feats)
        k = k.view(-1, self.num_heads, self.out_feats)
        v = v.view(-1, self.num_heads, self.out_feats)
        # k, q instead of q, k, the edge_softmax is applied on incoming edges
        graph.ndata.update({'ft': v, 'el': k, 'er': q})
        # compute edge attention
        graph.apply_edges(fn.u_dot_v('el', 'er', 'e'))
        e = graph.edata.pop('e') / math.sqrt(self.out_feats * self.num_heads)
        graph.edata['a'] = edge_softmax(graph, e)

        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft2'))
        rst = graph.ndata['ft2']

        # residual
        rst = rst.view(feat.shape) + feat

        rst = self.ln1(rst)
        rst = self.ln2(rst + self.FFN(rst))
        return rst


class GraphTransformer(nn.Module):
    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        self.gat = nn.ModuleList([GAT(args) for _ in range(args.prop)])
        self.prop = args.prop

    def forward(self, graph, feat):
        for i in range(self.prop):
            feat = self.gat[i](graph, feat)

        g_entity = feat.index_select(0, graph.filter_nodes(lambda x: x.data['type'] == NODE_TYPE['entity']))
        g_root = feat.index_select(0, graph.filter_nodes(lambda x: x.data['type'] == NODE_TYPE['root']))
        return g_entity, g_root