# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bart.py
import random
from copy import deepcopy

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from .modules import *


class GraphBARTMultiGPUWrapper(nn.Module):

    def __init__(self, args):
        super(GraphBARTMultiGPUWrapper, self).__init__()

        assert args.gpu in [0, 1, 2, 3]
        if args.gpu > 0:
            assert torch.cuda.is_available() and torch.cuda.device_count() > 0
            self._device = 'cuda'
        else:
            self._device = 'cpu'
        print(f'Using {self._device}...')

        self.args = args

        # We shard the model into multiple gpus if possible
        self._device_encoder = None
        self._device_decoder1 = self._device_decoder2 = None

        # BART
        self.interface = BartForConditionalGeneration.from_pretrained(args.checkpoint)
        self._tokenizer = BartTokenizer.from_pretrained(args.checkpoint)

        # Reference graph layer initialization
        # For prepend
        self.ref_prepend = nn.Linear(args.emb_dim, args.hidden_size)
        self.ref_prepend.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_prepend.bias.data.zero_()
        # For reduction after concatenation
        self.ref_redc = nn.Linear(args.emb_dim + args.hidden_size, args.hidden_size)
        self.ref_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_redc.bias.data.zero_()

        # Citation graph layer initialization
        # For prepend
        self.cit_prepend = nn.Linear(args.emb_dim, args.hidden_size)
        self.cit_prepend.weight.data.normal_(mean=0.0, std=0.02)
        self.cit_prepend.bias.data.zero_()
        # For reduction after concatenation
        self.cit_redc = nn.Linear(args.emb_dim + args.hidden_size, args.hidden_size)
        self.cit_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.cit_redc.bias.data.zero_()

        self.ref_cit_redc = nn.Linear(args.emb_dim * 2 + args.hidden_size, args.hidden_size)
        self.ref_cit_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_cit_redc.bias.data.zero_()

        # For prepend together ref and cit
        self.ref_cit_prepend = nn.Linear(args.emb_dim * 2, args.hidden_size)
        self.ref_cit_prepend.weight.data.normal_(mean=0.0, std=0.02)
        self.ref_cit_prepend.weight.data.zero_()

        # Concept graph layers initialization
        self.ent_type_emb = nn.Embedding(args.ent_type_num, args.hidden_size)
        nn.init.xavier_normal_(self.ent_type_emb.weight)
        self.rel_emb = nn.Embedding(args.rel_type_num, args.graph_hidden_size)
        nn.init.xavier_normal_(self.rel_emb.weight)
        # Used for dimension reduction after concatenating entity type embedding
        # and entity text embedding
        self.ent_dim_redc = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.ent_dim_redc.weight.data.normal_(mean=0.0, std=0.02)
        self.ent_dim_redc.bias.data.zero_()
        self.gtrans = GraphTransformer(args)
        # Used for dimension decrease before GAT
        self.gtrans_dec = nn.Linear(args.hidden_size, args.graph_hidden_size)
        # Used for dimension increase after GAT
        self.gtrans_inc = nn.Linear(args.graph_hidden_size, args.hidden_size)

        # Add knowledge cross attention layer and a layer norm for each BartDecoderLayer
        for i in range(len(self.decoder.layers)):
            self.decoder.layers[i].ent_attn = Attention(args)
            self.decoder.layers[i].ent_layer_norm = deepcopy(self.decoder.layers[i].encoder_attn_layer_norm)
            if args.knowledge_first:
                self.decoder.layers[i].knowledge_first = True
            else:
                self.decoder.layers[i].knowledge_first = False

        self._mode = None

    def set_mode(self, mode):
        assert mode in ['train', 'infer']

        if self._mode == mode:
            return

        if self.args.gpu == 3:
            if mode == 'train':
                assert torch.cuda.device_count() >= 3
                # make sure it has enough resources
                self._device_encoder = 'cuda:0'
                self._device_decoder1 = 'cuda:1'
                self._device_decoder2 = 'cuda:2'
            else:
                # During inference we only use one GPU
                self._device_encoder = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()
        elif self.args.gpu == 2:
            if mode == 'train':
                assert torch.cuda.device_count() >= 2
                self._device_encoder = 'cuda:0'
                self._device_decoder1 = self._device_decoder2 = 'cuda:1'
            else:
                # either only have 1 GPU, or during inference
                self._device_encoder = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()
        elif self._device == 'cuda':
            self._device_encoder = self._device_decoder1 = self._device_decoder2 = 'cuda:0'
            self.cuda()
        else:
            self._device_encoder = self._device_decoder1 = self._device_decoder2 = self._device

        # Model Sharding
        self.encoder.to(self._device_encoder)
        self.decoder.to(self._device_decoder1)

        # We shard the second half of decoder into another gpu if possible
        decoder_layer_num = len(self.decoder.layers)
        for i in range(decoder_layer_num):
            if i >= (decoder_layer_num // 2):
                self.decoder.layers[i].to(self._device_decoder2)
        if self.decoder.layer_norm:
            self.decoder.layer_norm.to(self._device_decoder2)

        # For calculating lm logits
        self.interface.final_logits_bias = move_device(
            self.interface.final_logits_bias, self._device_decoder2)
        self.model.shared = move_device(self.model.shared, self._device_decoder2)

        # Reference graph layers
        self.ref_prepend.to(self._device_encoder)
        self.ref_redc.to(self._device_encoder)

        # Citation graph layers
        self.cit_prepend.to(self._device_encoder)
        self.cit_redc.to(self._device_encoder)

        self.ref_cit_redc.to(self._device_encoder)
        self.ref_cit_prepend.to(self._device_encoder)

        # concept graph layers
        self.ent_type_emb.to(self._device_encoder)
        self.rel_emb.to(self._device_encoder)
        self.ent_dim_redc.to(self._device_encoder)
        self.gtrans.to(self._device_encoder)
        self.gtrans_inc.to(self._device_encoder)
        self.gtrans_dec.to(self._device_encoder)

        torch.cuda.empty_cache()

        # Set mode
        if mode == 'train':
            self.train()
        else:
            self.eval()

        self._mode = mode
        self.decoder.mode = mode  # use for decide whether to cache previous states

    def encode(self, sentence, max_length):
        """ Encode text (up to max_length)
            Example output:
            tensor([0, 9226, 16, 41, 15162, 2])
        """
        tokens = self._tokenizer([sentence], max_length=max_length,
                                 truncation=True, return_tensors='pt')['input_ids'][0].tolist()

        new_tokens = [elem for elem in tokens if self.is_valid(elem)]
        return torch.tensor(new_tokens).long()

    def is_valid(self, element: int):
        """ In the vocab: 50264 is <mask>, 50265 is None, we need to avoid 50264 and 50265 """
        if element != 50264 and element != 50265:
            return True
        else:
            return False

    def forward(self, src_tokens, prev_output_tokens, entity_type, node_data, graph,
                ref_emb, citation_emb):
        """ Forward text directly """
        src_tokens = src_tokens.to(self._device_encoder)
        attention_mask = src_tokens.ne(self.config.pad_token_id)

        # Get representations of source tokens
        encoder_out, _, _ = forward_encoder(
            self=self.encoder,
            src_tokens=src_tokens,
            attention_mask=attention_mask
        )  # (1 X 27 X 1024)

        ent_out, g_root = self.get_ent_out(node_data, graph)

        if not self.args.concept_graph:
            ent_out = None

        # Injecting reference embedding and citation embedding as well as global node of concept graph
        encoder_out, src_tokens, attention_mask = self.enrich_encoder_out(
            src_tokens=src_tokens,
            attention_mask=attention_mask,
            encoder_out=encoder_out,
            ref_emb=ref_emb,
            citation_emb=citation_emb,
            global_node=g_root,
            entity_nodes=ent_out
        )
        # Remove ent_out
        if self.args.prepend_concept:
            ent_out = None

        # decoder cached states
        x, _ = self.get_decoder_out(src_tokens, prev_output_tokens, encoder_out,
                                    attention_mask, ent_out=ent_out)

        lm_logits = F.linear(x, self.model.shared.weight, bias=self.interface.final_logits_bias)

        return lm_logits

    def enrich_encoder_out(self, src_tokens, attention_mask, encoder_out, ref_emb, citation_emb, global_node,
                           entity_nodes):
        """ Use reference embedding and citation embedding to refine encoder out """
        # common for prepend
        prepend_tokens = torch.tensor([self.config.bos_token_id]).unsqueeze(1).to(src_tokens.device)
        prepend_mask = torch.tensor([True]).unsqueeze(1).to(attention_mask.device)

        if self.args.concept_graph_global:
            # only prepend to the encoder out
            global_node = global_node.unsqueeze(1).to(encoder_out.device)
            encoder_out = torch.cat([global_node, encoder_out], dim=1)
            src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)

        if self.args.prepend_concept:
            entity_num = entity_nodes.shape[1]
            entity_nodes = entity_nodes.to(encoder_out.device)
            entity_prepend_tokens = torch.tensor([self.config.bos_token_id] * entity_num) \
                .unsqueeze(0).to(src_tokens.device)
            entity_prepend_mask = torch.tensor([True] * entity_num).unsqueeze(0).to(attention_mask.device)
            encoder_out = torch.cat([entity_nodes, encoder_out], dim=1)
            src_tokens = torch.cat([entity_prepend_tokens, src_tokens], dim=1)
            attention_mask = torch.cat([entity_prepend_mask, attention_mask], dim=1)

        # If we use reference embedding information
        if self.args.ref_graph:
            if self.args.citation_graph and self.args.prepend:
                # ref embedding + cit embedding
                ref_emb = ref_emb.to(self.ref_cit_prepend.weight.device)
                citation_emb = citation_emb.to(self.ref_cit_prepend.weight.device)
                together_emb = torch.cat([ref_emb, citation_emb], dim=1)
                together_transformed = self.ref_cit_prepend(together_emb).unsqueeze(1)
                encoder_out = torch.cat([together_transformed, encoder_out], dim=1)
                src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
                attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
            elif self.args.prepend:
                # transform the embedding and concate it with encoder_out
                ref_emb = ref_emb.to(self.ref_prepend.weight.device)
                ref_emb_transformed = self.ref_prepend(ref_emb).unsqueeze(1)

                # prepend the reference graph for encoder_out
                encoder_out = torch.cat([ref_emb_transformed, encoder_out], dim=1)
                # prepend the reference graph for src_tokens
                src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
                # prepend the attention mask
                attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
            else:
                # concate the embedding with encoder out, then pass a dimension reduction matrix
                ref_emb = ref_emb.unsqueeze(1)
                ref_emb = ref_emb.expand(1, encoder_out.shape[1], ref_emb.shape[-1])
                ref_emb = ref_emb.to(encoder_out.device)
                encoder_out = torch.cat([ref_emb, encoder_out], dim=-1)

        # If we use citation embedding information
        if self.args.citation_graph:
            if self.args.ref_graph and self.args.prepend:
                pass
            elif self.args.prepend:
                citation_emb = citation_emb.to(self.cit_prepend.weight.device)
                citation_emb_transformed = self.cit_prepend(citation_emb).unsqueeze(1)

                encoder_out = torch.cat([citation_emb_transformed, encoder_out], dim=1)
                src_tokens = torch.cat([prepend_tokens, src_tokens], dim=1)
                attention_mask = torch.cat([prepend_mask, attention_mask], dim=1)
            else:
                citation_emb = citation_emb.unsqueeze(1)
                citation_emb = citation_emb.expand(1, encoder_out.shape[1], citation_emb.shape[-1])
                citation_emb = citation_emb.to(encoder_out.device)
                encoder_out = torch.cat([citation_emb, encoder_out], dim=-1)

        # Use different dimension reduction matrix
        if self.args.ref_graph and not self.args.citation_graph and not self.args.prepend:
            encoder_out = self.ref_redc(encoder_out)
        if not self.args.ref_graph and self.args.citation_graph and not self.args.prepend:
            encoder_out = self.cit_redc(encoder_out)
        if self.args.ref_graph and self.args.citation_graph and not self.args.prepend:
            encoder_out = self.ref_cit_redc(encoder_out)
        return encoder_out, src_tokens, attention_mask

    def get_ent_out(self, node_data, graph):
        """ Get entity representations by going through GraphTransformer """
        ent_out, g_root = None, None
        if self.args.concept_graph or self.args.concept_graph_global:
            # Get initial node representations
            raw_node_enc, _, _ = forward_encoder(
                self=self.encoder,
                src_tokens=node_data,
                attention_mask=node_data.ne(self.config.pad_token_id),
                low=True
            )
            # Take the representation at EOS to be the node representation
            eos_mask = node_data.eq(self.config.eos_token_id)
            node_enc = raw_node_enc[eos_mask, :].view(raw_node_enc.size(0), -1,
                                                      raw_node_enc.size(-1))[:, -1, :]  # (18 X 1024)

            # Decrease the dimension
            node_enc = self.gtrans_dec(node_enc)

            # Go through Graph Transformer
            graph = graph.to(self._device_encoder)
            ent_out, g_root = self.gtrans(graph, node_enc)

            # Increase the dimension
            g_root = self.gtrans_inc(g_root)
            ent_out = self.gtrans_inc(ent_out)
            ent_out = ent_out.unsqueeze(0)  # (1 x 18 x 1024)
        return ent_out, g_root

    def get_decoder_out(self, src_tokens, prev_output_tokens, encoder_out,
                        encoder_attention_mask, ent_out, decoder_cached_states=None,
                        output_attentions=False):
        """ Given encoder outputs, decoder input, get decoder outputs """

        use_cache = True if self.decoder.mode == 'infer' else False

        if not use_cache:
            _, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                config=self.config,
                input_ids=src_tokens,
                decoder_input_ids=prev_output_tokens,
                causal_mask_dtype=self.model.shared.weight.dtype,
            )  # decoder_padding_mask: None, causal_mask: (194 X 194)
        else:
            decoder_padding_mask, causal_mask = None, None

        encoder_ent_mask = None
        if ent_out is not None:
            encoder_ent_mask = torch.tensor([True] * ent_out.shape[1]).expand(ent_out.shape[0],
                                                                              ent_out.shape[1])  # (1 X 18)

        x, next_cache, _, _ = forward_decoder(
            self=self.decoder,
            tgt_tokens=prev_output_tokens,
            encoder_hidden_states=encoder_out,
            ent_out=ent_out,
            encoder_padding_mask=encoder_attention_mask,
            encoder_ent_mask=encoder_ent_mask,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
        )

        return x, next_cache

    def generate(self, data, max_length, min_length,
                 num_beams, length_penalty, no_repeat_ngram_size):
        output = generate(
            self=self,
            data=data,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return output

    @property
    def config(self):
        return self.interface.model.config

    @property
    def model(self):
        return self.interface.model

    @property
    def encoder(self):
        return self.interface.model.encoder

    @property
    def decoder(self):
        return self.interface.model.decoder

    @property
    def tokenizer(self):
        return self._tokenizer


def forward_embedding(self, tokens):
    """ Embed the tokens """
    inputs_embeds = self.embed_tokens(tokens.to(self.embed_tokens.weight.device)) \
                    * self.embed_scale

    embed_pos = self.embed_positions(tokens.to(self.embed_positions.weight.device))

    inputs_embeds = move_device(inputs_embeds, embed_pos.device)
    x = inputs_embeds + embed_pos

    x = move_device(x, self.layernorm_embedding.weight.device)
    x = self.layernorm_embedding(x)

    x = F.dropout(x, p=self.dropout, training=self.training)
    return x


def forward_encoder(self, src_tokens, attention_mask=None, output_attentions=False,
                    output_hidden_states=False, low=False):
    """
    Args:
        self: In the model, self is self.encoder
        src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        low: if True, then only use the first three layers to encode
    Returns:
        Tuple comprised of:
            - **x** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_states** (List[Tensor]): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *output_hidden_states:* is True.
            - **all_attentions** (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
    """

    # check attention mask and invert
    if attention_mask is not None:
        attention_mask = invert_mask(attention_mask)

    # B x T x C -> T x B x C
    x = forward_embedding(self=self, tokens=src_tokens)
    x = x.transpose(0, 1)

    encoder_states, all_attentions = [], []
    for idx, encoder_layer in enumerate(self.layers):
        # if only use the first six layers
        if low and idx == len(self.layers) // 2:
            break
        # first half and second half of encoder not on the same device
        current_device = encoder_layer.fc1.weight.device
        x = move_device(x, current_device)
        attention_mask = move_device(attention_mask, current_device)
        if output_hidden_states:
            encoder_states.append(x)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            attn = None
        else:
            x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

        if output_attentions:
            all_attentions.append(attn)

    if self.layer_norm:
        x = move_device(x, self.layer_norm.weight.device)
        x = self.layer_norm(x)
    if output_hidden_states:
        encoder_states.append(x)

    # T x B x C -> B x T x C
    encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
    x = x.transpose(0, 1)

    return x, encoder_states, all_attentions


def forward_decoder(self, tgt_tokens, encoder_hidden_states, ent_out, encoder_padding_mask,
                    encoder_ent_mask, decoder_padding_mask, decoder_causal_mask,
                    decoder_cached_states=None, output_attentions=False,
                    output_hidden_states=False, **unused):
    """
    Includes several features from "Jointly Learning to Align and
    Translate with Transformer Models" (Garg et al., EMNLP 2019).

    Args:
        self: In the model, self is self.decoder
        tgt_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_hidden_states/ent_out: output from the encoder, used for
            encoder-side attention
        encoder_padding_mask/encoder_ent_mask: for ignoring pad tokens
        decoder_cached_states (dict or None): dictionary used for storing state during generation

    Returns:
        tuple:
            - the decoder's features of shape `(batch, tgt_len, embed_dim)`
            - hidden states
            - attentions
    """
    input_ids = tgt_tokens
    use_cache = (self.mode == 'infer')

    # check attention mask and invert
    if encoder_padding_mask is not None:
        encoder_padding_mask = invert_mask(encoder_padding_mask)
    if encoder_ent_mask is not None:
        encoder_ent_mask = invert_mask(encoder_ent_mask)

    # embed positions
    positions = self.embed_positions(input_ids.to(self.embed_positions.weight.device), use_cache=use_cache)
    if use_cache:
        input_ids = input_ids[:, -1:]
        positions = positions[:, -1:]  # happens after we embed them
        # assert input_ids.ne(self.padding_idx).any()

    x = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device)) * self.embed_scale
    x = x.to(positions.device)
    x += positions
    x = self.layernorm_embedding(x.to(self.layernorm_embedding.weight.device))
    x = F.dropout(x, p=self.dropout, training=self.training)

    # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
    x = x.transpose(0, 1)
    encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

    # decoder layers
    all_hidden_states = ()
    all_cross_attns = ()
    next_decoder_cache = []

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        current_device = decoder_layer.fc1.weight.device

        x = move_device(x, current_device)
        encoder_padding_mask = move_device(encoder_padding_mask, current_device)
        decoder_padding_mask = move_device(decoder_padding_mask, current_device)
        encoder_hidden_states = move_device(encoder_hidden_states, current_device)
        decoder_causal_mask = move_device(decoder_causal_mask, current_device)

        if output_hidden_states:
            all_hidden_states += (x,)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):
            continue

        layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

        # size of layer_cross_attn: (bsz, num_heads, tgt_len, src_len)
        x, layer_cross_attn, layer_past = forward_decoder_layer(
            self=decoder_layer,
            x=x,
            encoder_hidden_states=encoder_hidden_states,
            ent_out=ent_out,
            encoder_attn_mask=encoder_padding_mask,
            encoder_ent_mask=encoder_ent_mask,
            decoder_padding_mask=decoder_padding_mask,
            layer_state=layer_state,
            causal_mask=decoder_causal_mask,
            output_attentions=output_attentions,
        )

        if use_cache:
            next_decoder_cache.append(layer_past.copy())

        if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
            x = move_device(x, self.layer_norm.weight.device)
            x = self.layer_norm(x)
        if output_attentions:
            all_cross_attns += (layer_cross_attn,)

    # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
    all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
    x = x.transpose(0, 1)

    if use_cache:
        next_cache = next_decoder_cache
    else:
        next_cache = None

    return x, next_cache, all_hidden_states, list(all_cross_attns)


def forward_decoder_layer(self, x, encoder_hidden_states, ent_out, encoder_attn_mask=None,
                          encoder_ent_mask=None, layer_state=None, causal_mask=None,
                          decoder_padding_mask=None, output_attentions=False):
    """ Output cross attention weights """
    if layer_state is None:
        layer_state = {}

    # Self Attention
    residual = x
    if self.normalize_before:
        x = self.self_attn_layer_norm(x)

    x, self_attn_weights = self.self_attn(
        query=x,
        key=x,
        layer_state=layer_state,  # adds keys to layer state
        key_padding_mask=decoder_padding_mask,
        attn_mask=causal_mask,
        output_attentions=output_attentions
    )

    x = F.dropout(x, p=self.dropout, training=self.training)
    x = residual + x
    if not self.normalize_before:
        x = self.self_attn_layer_norm(x)

    if self.knowledge_first:
        # Cross attention for knowledge
        if ent_out is not None:
            x = attend_to_ent(self, x, ent_out, encoder_ent_mask, layer_state, output_attentions)

    # Cross attention for text
    residual = x
    assert self.encoder_attn.cache_key != self.self_attn.cache_key
    if self.normalize_before:
        x = self.encoder_attn_layer_norm(x)

    x, cross_attn_weights = self.encoder_attn(
        query=x,
        key=encoder_hidden_states,
        key_padding_mask=encoder_attn_mask,
        layer_state=layer_state,  # mutates layer state
        output_attentions=output_attentions
    )
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = residual + x
    if not self.normalize_before:
        x = self.encoder_attn_layer_norm(x)

    if not self.knowledge_first:
        # Cross attention for knowledge
        # Pop out some cache in {}
        if ent_out is not None:
            x = attend_to_ent(self, x, ent_out, encoder_ent_mask, layer_state, output_attentions)

    # Fully Connected
    residual = x
    if self.normalize_before:
        x = self.final_layer_norm(x)
    x = self.activation_fn(self.fc1(x))
    x = F.dropout(x, p=self.activation_dropout, training=self.training)
    x = self.fc2(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = residual + x
    if not self.normalize_before:
        x = self.final_layer_norm(x)
    return (
        x,
        cross_attn_weights,
        layer_state,
    )  # Return cross attention weights to see where the model attends


def attend_to_ent(self, x, ent_out, encoder_ent_mask, layer_state, output_attentions):
    """ Cross attention to entities in decoder layer
        self: decoder layer
    """
    residual = x
    assert self.ent_attn.cache_key != self.self_attn.cache_key
    assert self.ent_attn.cache_key != self.encoder_attn.cache_key
    if self.normalize_before:
        x = self.ent_layer_norm(x)

    ent_out = ent_out.to(x.device)
    encoder_ent_mask = encoder_ent_mask.to(x.device)
    x, _ = self.ent_attn(
        query=x,
        key=ent_out,
        key_padding_mask=encoder_ent_mask,
        layer_state=layer_state,
        output_attentions=output_attentions
    )
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = residual + x
    if not self.normalize_before:
        x = self.ent_layer_norm(x)
    return x


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def move_device(tensor, device):
    if tensor is None:
        return None
    else:
        tensor = tensor.to(device)
        return tensor


def _prepare_bart_decoder_inputs(config, input_ids, decoder_input_ids=None,
                                 decoder_padding_mask=None, causal_mask_dtype=torch.float32):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


# ===================================================== #
#                      Generation                       #
# ===================================================== #

@torch.no_grad()
def generate(self: GraphBARTMultiGPUWrapper, data, max_length,
             min_length, num_beams, length_penalty, no_repeat_ngram_size,
             **model_specific_kwargs):
    """
    Return token ids (LongTensor)
    """
    encoder_device = self.encoder.embed_tokens.weight.device
    input_ids = data.src_tokens
    input_ids = input_ids.to(encoder_device)
    batch_size = input_ids.shape[0]
    original_attention_mask = input_ids.ne(self.config.pad_token_id).long()

    effective_batch_size = batch_size
    effective_batch_mult = 1

    # Here the input_ids is used to get the encoder output
    original_encoder_outputs: tuple = self.encoder(input_ids, attention_mask=original_attention_mask)
    encoder_outputs = original_encoder_outputs[0]

    ent_out, g_root = self.get_ent_out(data.node_data, data.graph)

    encoder_outputs, input_ids, original_attention_mask = self.enrich_encoder_out(
        src_tokens=input_ids,
        attention_mask=original_attention_mask,
        encoder_out=encoder_outputs,
        ref_emb=data.ref_emb,
        citation_emb=data.citation_emb,
        global_node=g_root,
        entity_nodes=ent_out
    )

    if self.args.prepend_concept:
        ent_out = None

    input_ids_len = input_ids.shape[-1]

    attention_mask = original_attention_mask.unsqueeze(1).expand(
        batch_size, effective_batch_mult * num_beams, input_ids_len
    )

    attention_mask = attention_mask.contiguous().view(
        effective_batch_size * num_beams, input_ids_len
    )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    # used for masking attention to the encoder side

    # create empty decoder_input_ids
    input_ids = torch.full(
        (effective_batch_size * num_beams, 1),
        self.config.decoder_start_token_id,
        dtype=torch.long,
        device=next(self.interface.parameters()).device,
    )
    cur_len = 1

    # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and
    # num_return_sequences > 1)
    expanded_batch_idxs = (
        torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
    )

    # expand encoder_outputs
    encoder_outputs = (encoder_outputs.index_select(0, expanded_batch_idxs),
                       *original_encoder_outputs[1:])  # (1 X 14 X 1024) --> (4 X 14 X 1024)

    ent_out = ent_out.index_select(0,
                                   expanded_batch_idxs) if ent_out is not None else None  # (1 X 17 X 1024) --> (4 X 17 X 1024)

    output = _generate_beam_search(
        self=self,
        input_ids=input_ids,
        cur_len=cur_len,
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        batch_size=effective_batch_size,
        length_penalty=length_penalty,
        num_beams=num_beams,
        encoder_outputs=encoder_outputs,
        ent_out=ent_out,
        attention_mask=attention_mask,
        model_specific_kwargs=model_specific_kwargs,
    )

    # decode the generated text (and then encode in order to align predition)
    decoded_text = [self.tokenizer.decode(g, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False).strip()
                    for g in output]

    return decoded_text


def _generate_beam_search(self: GraphBARTMultiGPUWrapper, input_ids, cur_len, max_length, min_length,
                          no_repeat_ngram_size, batch_size, length_penalty, num_beams, encoder_outputs,
                          ent_out, attention_mask, model_specific_kwargs):
    """ Generate sequences for each example with beam search.
    """
    # Configuration
    early_stopping = self.config.early_stopping
    use_cache = self.config.use_cache
    bad_words_ids = self.config.bad_words_ids
    eos_token_id = self.config.eos_token_id
    pad_token_id = self.config.pad_token_id
    repetition_penalty = self.config.repetition_penalty
    vocab_size = self.config.vocab_size
    num_return_sequences = self.config.num_return_sequences

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the
    # exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # The default cache compute states
    past = (encoder_outputs, None) if encoder_outputs is not None else None
    # Our cache compute states
    decoder_cached_states = None

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        if ent_out is not None:
            x, decoder_cached_states = self.get_decoder_out(None, input_ids, encoder_outputs[0],
                                                            attention_mask, ent_out=ent_out,
                                                            decoder_cached_states=decoder_cached_states)

            lm_logits = F.linear(x, self.model.shared.weight, bias=self.interface.final_logits_bias)
            next_token_logits = lm_logits[:, -1, :]
        else:
            # this prepare inputs for generation is different than the one above
            model_inputs = self.interface.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache,
                encoder_outputs=encoder_outputs, **model_specific_kwargs
            )

            outputs = self.interface(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if to_use_cache(self, outputs, use_cache):
                past = outputs[1]

        next_token_logits = self.interface.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len, max_length=max_length
        )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        scores = postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        # We don't do sample
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size  # the beam id within this sentence
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # re-order internal states
        # TODO: Reorder decoder cache
        if ent_out is None:
            if past is not None:
                past = self.interface._reorder_cache(past, beam_idx)
        else:
            tmp_past = ((None, None), decoder_cached_states)
            decoder_cached_states = self.interface._reorder_cache(tmp_past, beam_idx)[1]

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and
    # output_num_return_sequences_per_batch
    output_batch_size = batch_size * num_return_sequences
    output_num_return_sequences_per_batch = num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.interface.parameters()).device)

    return decoded
