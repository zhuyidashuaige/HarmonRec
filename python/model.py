import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs


class HarmonRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, item2code=None, item_content=None, codebook_sizes=None, code_dims=None,
                 item2code_text=None, item2code_audio=None, item2code_visual=None,
                 codebook_sizes_text=None, codebook_sizes_audio=None, codebook_sizes_visual=None,
                 code_dims_text=None, code_dims_audio=None, code_dims_visual=None,
                 item_content_text=None, item_content_audio=None, item_content_visual=None,
                 item2users=None):
        super(HarmonRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.mm_dim = getattr(args, 'mm_dim', 128)           # Unified multimodal target dimension = 128
        args.hidden_units = self.mm_dim                      # Keep Transformer hidden dimension in sync (= 128)
        self.use_semantic = any([
            item2code is not None and codebook_sizes is not None,
            item2code_text is not None and codebook_sizes_text is not None,
            item2code_audio is not None and codebook_sizes_audio is not None,
            item2code_visual is not None and codebook_sizes_visual is not None,
        ])

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)  # collaborative ID embedding b_i
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.score_alpha = getattr(args, 'score_alpha', 0.5)  # coefficient alpha for the collaborative add-on score term

        if self.use_semantic:
            if code_dims is None:
                base = args.hidden_units // 3
                code_dims = [base, base, args.hidden_units - 2 * base]
            if code_dims_text is None:
                code_dims_text = code_dims
            if code_dims_audio is None:
                code_dims_audio = code_dims
            if code_dims_visual is None:
                code_dims_visual = code_dims

            has_text = (item2code_text is not None and codebook_sizes_text is not None) or (item2code is not None and codebook_sizes is not None)
            has_audio = (item2code_audio is not None and codebook_sizes_audio is not None)
            has_visual = (item2code_visual is not None and codebook_sizes_visual is not None)
            if has_text or has_audio or has_visual:
                self.weight_mlp_shared = torch.nn.Sequential(           # Shared MLP across all modalities: input id emb -> 3 per-layer weights
                    torch.nn.Linear(args.hidden_units, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 3)
                )
            else:
                self.weight_mlp_shared = None
            self.weight_mlp_text = self.weight_mlp_shared
            self.weight_mlp_audio = self.weight_mlp_shared
            self.weight_mlp_visual = self.weight_mlp_shared

            if item2code_text is None and item2code is not None:
                item2code_text = item2code
                codebook_sizes_text = codebook_sizes

            if item2code_text is not None and codebook_sizes_text is not None:
                self.semantic_layers_text = torch.nn.ModuleList([
                    torch.nn.Embedding(codebook_sizes_text[0]+1, code_dims_text[0], padding_idx=0),
                    torch.nn.Embedding(codebook_sizes_text[1]+1, code_dims_text[1], padding_idx=0),
                    torch.nn.Embedding(codebook_sizes_text[2]+1, code_dims_text[2], padding_idx=0),
                ])
                self.semantic_projs_text = torch.nn.ModuleList([        # Project each text semantic layer to mm_dim (128)
                    torch.nn.Linear(code_dims_text[0], self.mm_dim),
                    torch.nn.Linear(code_dims_text[1], self.mm_dim),
                    torch.nn.Linear(code_dims_text[2], self.mm_dim),
                ])
                self.register_buffer('item2code_text', item2code_text)

            if item2code_audio is not None and codebook_sizes_audio is not None:
                self.semantic_layers_audio = torch.nn.ModuleList([
                    torch.nn.Embedding(codebook_sizes_audio[0]+1, code_dims_audio[0], padding_idx=0),
                    torch.nn.Embedding(codebook_sizes_audio[1]+1, code_dims_audio[1], padding_idx=0),
                    torch.nn.Embedding(codebook_sizes_audio[2]+1, code_dims_audio[2], padding_idx=0),
                ])
                self.semantic_projs_audio = torch.nn.ModuleList([       # Project each audio semantic layer to mm_dim (128)
                    torch.nn.Linear(code_dims_audio[0], self.mm_dim),
                    torch.nn.Linear(code_dims_audio[1], self.mm_dim),
                    torch.nn.Linear(code_dims_audio[2], self.mm_dim),
                ])
                self.register_buffer('item2code_audio', item2code_audio)

            if item2code_visual is not None and codebook_sizes_visual is not None:
                self.semantic_layers_visual = torch.nn.ModuleList([
                    torch.nn.Embedding(codebook_sizes_visual[0]+1, code_dims_visual[0], padding_idx=0),
                    torch.nn.Embedding(codebook_sizes_visual[1]+1, code_dims_visual[1], padding_idx=0),
                    torch.nn.Embedding(codebook_sizes_visual[2]+1, code_dims_visual[2], padding_idx=0),
                ])
                self.semantic_projs_visual = torch.nn.ModuleList([      # Project each visual semantic layer to mm_dim (128)
                    torch.nn.Linear(code_dims_visual[0], self.mm_dim),
                    torch.nn.Linear(code_dims_visual[1], self.mm_dim),
                    torch.nn.Linear(code_dims_visual[2], self.mm_dim),
                ])
                self.register_buffer('item2code_visual', item2code_visual)

            if item_content_text is not None:
                self.register_buffer('item_content_text', item_content_text)
                in_dim_t = item_content_text.shape[1]
                self.content_proj_text = torch.nn.Linear(in_dim_t, self.mm_dim)   # Raw text content (e.g. Flan-T5-XL 2048) -> 128
            else:
                self.item_content_text = None
                self.content_proj_text = None
            if item_content_audio is not None:
                self.register_buffer('item_content_audio', item_content_audio)
                in_dim_a = item_content_audio.shape[1]
                self.content_proj_audio = torch.nn.Linear(in_dim_a, self.mm_dim)  # Raw audio content (e.g. Essentia 256) -> 128
            else:
                self.item_content_audio = None
                self.content_proj_audio = None
            if item_content_visual is not None:
                self.register_buffer('item_content_visual', item_content_visual)
                in_dim_v = item_content_visual.shape[1]
                self.content_proj_visual = torch.nn.Linear(in_dim_v, self.mm_dim) # Raw visual content (e.g. ResNet pool 256) -> 128
            else:
                self.item_content_visual = None
                self.content_proj_visual = None

        # Fusion MLP: project the concatenated [base, content] (2 * mm_dim) back to mm_dim.
        self.fusion_mlp = torch.nn.Linear(self.mm_dim * 2, self.mm_dim)

        # ID -> content-weight MLP: input ID emb (128) -> [w_t, w_a, w_v] (3 per-modality weights).
        self.id_to_content_mlp = torch.nn.Linear(self.mm_dim, 3)

        # Content fusion projection: weighted-concat of three modalities (3 * mm_dim) -> mm_dim.
        self.content_fusion_proj = torch.nn.Linear(self.mm_dim * 3, self.mm_dim)

        # Frequency-aware reweighting network:
        # input is the per-item normalized frequency q''; output is [w_id, w_con].
        self.temp = getattr(args, 'temp', 1.0)
        self.mlp_head = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.GELU(), torch.nn.Linear(64, 10))               # 1 -> 64 -> 10
        self.mlp_meta_embedding = torch.nn.Sequential(torch.nn.Linear(10, 64, bias=False), torch.nn.GELU(), torch.nn.Linear(64, 2))  # 10 -> 64 -> 2

        # compute item frequency from interactions if provided
        if item2users is not None:
            fre = torch.zeros(self.item_num+1, 1, dtype=torch.float32)
            for iid, users in enumerate(item2users):
                if iid == 0:
                    continue
                fre[iid, 0] = float(len(users))
            fre = torch.log(fre + 1.0)                 # q' = log(q + 1)
            min_v = torch.min(fre[1:]) if self.item_num >= 1 else torch.tensor(0.0)
            max_v = torch.max(fre[1:]) if self.item_num >= 1 else torch.tensor(1.0)
            denom = (max_v - min_v)
            denom = denom if denom > 0 else torch.tensor(1.0)
            fre = (fre - min_v) / denom                # q'' = (q' - min) / (max - min)
            self.register_buffer('freq_norm', fre)
        else:
            self.register_buffer('freq_norm', torch.zeros(self.item_num+1, 1, dtype=torch.float32))

        self.attention_layernorms = torch.nn.ModuleList() 
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    # Sequence modeling: take the fused item embedding (via get_item_embedding) as the
    # token representation, add positional embedding + dropout, apply a lower-triangular
    # causal mask, then stack self-attention + FFN blocks, and finally LayerNorm to obtain log_feats.
    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        ids = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.get_item_embedding(ids)            # use the fused item emb (128) as the token vector
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
    # Training forward pass (in-batch negative sampling, paper Sec. III-F):
    # Given (user_ids, log_seqs, pos_seqs), treats all unique positive items in
    # the mini-batch as the candidate pool I_batch. For every valid (non-padding)
    # position, compute the hybrid score z_{p,i} against every candidate and
    # return (logits, targets) ready for nn.CrossEntropyLoss.
    #
    # logits : (N_valid, C)  where C = |I_batch| (unique positive items in batch)
    # targets: (N_valid,)    index of the ground-truth item in the candidate pool
    def forward(self, user_ids, log_seqs, pos_seqs): # for training
        log_feats = self.log2feats(log_seqs)           # (B, T, d)

        pos_seqs_t = torch.LongTensor(pos_seqs).to(self.dev)  # (B, T)
        pos_flat = pos_seqs_t.reshape(-1)                      # (B*T,)
        valid_mask = pos_flat != 0

        # All unique positive items in the batch form the candidate pool I_batch.
        # torch.unique returns a sorted tensor, required by searchsorted below.
        candidate_ids = torch.unique(pos_flat[valid_mask])     # (C,)

        # Fused item embeddings and collaborative ID embeddings for all candidates.
        cand_embs    = self.get_item_embedding(candidate_ids)  # (C, d)  f_i
        cand_id_base = self.item_emb(candidate_ids)            # (C, d)  b_i
        w_c_cand, _  = self._get_weights(candidate_ids)        # (C, 1)
        w_c_cand     = w_c_cand.squeeze(-1)                    # (C,)

        # Valid user preference vectors h_p at each non-padding position.
        B, T = pos_seqs_t.shape
        valid_mask_2d = (pos_seqs_t != 0)                      # (B, T)
        user_feats    = log_feats[valid_mask_2d]               # (N_valid, d)
        pos_valid     = pos_flat[valid_mask]                   # (N_valid,)  ground-truth ids

        # Hybrid scoring: z_{p,i} = h_p^T f_i + alpha * w_id * (h_p^T b_i)
        logits    = user_feats @ cand_embs.T                   # (N_valid, C)
        id_logits = user_feats @ cand_id_base.T                # (N_valid, C)
        logits    = logits + self.score_alpha * w_c_cand.unsqueeze(0) * id_logits  # (N_valid, C)

        # Target: index of each ground-truth item inside the sorted candidate_ids.
        targets = torch.searchsorted(candidate_ids, pos_valid) # (N_valid,)

        return logits, targets

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        ids = torch.LongTensor(item_indices).to(self.dev)
        item_embs = self.get_item_embedding(ids)       # f_i (fused 128-d item embedding)
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # u dot f_i
        id_base = self.item_emb(ids)
        w_c, _ = self._get_weights(ids)
        id_logits = id_base.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        wc_vec = w_c.squeeze(-1)
        logits = logits + self.score_alpha * wc_vec * id_logits          # + alpha * (w_c * u dot b_i)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

    def _get_weights(self, item_ids):
        # Compute the two scalar weights [w_id, w_con]:
        #   - Input is the per-item normalized frequency q'' (pre-computed in __init__
        #     and registered as a buffer).
        #   - Two-layer MLP followed by a temperature-controlled Softmax(tau).
        q = self.freq_norm[item_ids]                   # normalized frequency q''
        f_logits = self.mlp_meta_embedding(self.mlp_head(q))  # two-layer MLP
        weights2 = torch.softmax(f_logits / self.temp, dim=-1)  # Softmax(tau)
        return (
            weights2[..., 0].unsqueeze(-1),
            weights2[..., 1].unsqueeze(-1),
        )

    def get_item_embedding(self, item_ids):
        # 1. Collaborative component (ID embedding).
        base = self.item_emb(item_ids)                 # collaborative component b_i (128-d)
        mask = (item_ids == 0)

        # 2. Derive per-modality content weights [w_t, w_a, w_v] from the ID embedding,
        #    normalized by a Softmax.
        content_weights = torch.softmax(self.id_to_content_mlp(base), dim=-1)
        w_t = content_weights[..., 0].unsqueeze(-1)
        w_a = content_weights[..., 1].unsqueeze(-1)
        w_v = content_weights[..., 2].unsqueeze(-1)

        # semantic embeddings per modality
        s_text = torch.zeros_like(base)
        s_audio = torch.zeros_like(base)
        s_visual = torch.zeros_like(base)

        if hasattr(self, 'item2code_text') and self.weight_mlp_text is not None:
            codes = self.item2code_text[item_ids]
            weights = torch.softmax(self.weight_mlp_text(base), dim=-1)
            c1 = codes[..., 0]
            c2 = codes[..., 1]
            c3 = codes[..., 2]
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            w3 = weights[..., 2].unsqueeze(-1)
            e1 = self.semantic_layers_text[0](c1) * w1
            e2 = self.semantic_layers_text[1](c2) * w2
            e3 = self.semantic_layers_text[2](c3) * w3
            p1 = self.semantic_projs_text[0](e1)
            p2 = self.semantic_projs_text[1](e2)
            p3 = self.semantic_projs_text[2](e3)
            s_text = p1 + p2 + p3

        if hasattr(self, 'item2code_audio') and self.weight_mlp_audio is not None:
            codes = self.item2code_audio[item_ids]
            weights = torch.softmax(self.weight_mlp_audio(base), dim=-1)
            c1 = codes[..., 0]
            c2 = codes[..., 1]
            c3 = codes[..., 2]
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            w3 = weights[..., 2].unsqueeze(-1)
            e1 = self.semantic_layers_audio[0](c1) * w1
            e2 = self.semantic_layers_audio[1](c2) * w2
            e3 = self.semantic_layers_audio[2](c3) * w3
            p1 = self.semantic_projs_audio[0](e1)
            p2 = self.semantic_projs_audio[1](e2)
            p3 = self.semantic_projs_audio[2](e3)
            s_audio = p1 + p2 + p3

        if hasattr(self, 'item2code_visual') and self.weight_mlp_visual is not None:
            codes = self.item2code_visual[item_ids]
            weights = torch.softmax(self.weight_mlp_visual(base), dim=-1)
            c1 = codes[..., 0]
            c2 = codes[..., 1]
            c3 = codes[..., 2]
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            w3 = weights[..., 2].unsqueeze(-1)
            e1 = self.semantic_layers_visual[0](c1) * w1
            e2 = self.semantic_layers_visual[1](c2) * w2
            e3 = self.semantic_layers_visual[2](c3) * w3
            p1 = self.semantic_projs_visual[0](e1)
            p2 = self.semantic_projs_visual[1](e2)
            p3 = self.semantic_projs_visual[2](e3)
            s_visual = p1 + p2 + p3

        # 3. Weighted fusion across modalities into a single content embedding.
        # 3.1 Apply weights
        t_sem_w = s_text * w_t
        a_sem_w = s_audio * w_a
        v_sem_w = s_visual * w_v

        # 3.2 Handle masking
        t_sem_w = torch.where(mask.unsqueeze(-1), torch.zeros_like(t_sem_w), t_sem_w)
        a_sem_w = torch.where(mask.unsqueeze(-1), torch.zeros_like(a_sem_w), a_sem_w)
        v_sem_w = torch.where(mask.unsqueeze(-1), torch.zeros_like(v_sem_w), v_sem_w)

        # 3.3 Concat and Project to create Content Emb (128)
        content_raw = torch.cat([t_sem_w, a_sem_w, v_sem_w], dim=-1)
        content_emb = self.content_fusion_proj(content_raw)

        # 4. Frequency-aware final weights [w_id, w_con].
        w_id, w_con = self._get_weights(item_ids)

        # 5. Weighted concat of [id_emb, content_emb] and a linear projection back to mm_dim.
        base_w = base * w_id
        content_emb_w = content_emb * w_con
        
        base_w = torch.where(mask.unsqueeze(-1), torch.zeros_like(base_w), base_w)
        content_emb_w = torch.where(mask.unsqueeze(-1), torch.zeros_like(content_emb_w), content_emb_w)

        final_concat = torch.cat([base_w, content_emb_w], dim=-1)
        fused = self.fusion_mlp(final_concat)
        
        return fused
    # Helper to expose intermediate per-modality components for visualization / analysis only.
    def get_item_components(self, item_ids, weighted=True, concat=False):
        base = self.item_emb(item_ids)
        mask = (item_ids == 0)
        s_text = torch.zeros_like(base)
        s_audio = torch.zeros_like(base)
        s_visual = torch.zeros_like(base)
        if hasattr(self, 'item2code_text') and self.weight_mlp_text is not None:
            codes = self.item2code_text[item_ids]
            weights = torch.softmax(self.weight_mlp_text(base), dim=-1)
            c1 = codes[..., 0]; c2 = codes[..., 1]; c3 = codes[..., 2]
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            w3 = weights[..., 2].unsqueeze(-1)
            e1 = self.semantic_layers_text[0](c1) * w1
            e2 = self.semantic_layers_text[1](c2) * w2
            e3 = self.semantic_layers_text[2](c3) * w3
            p1 = self.semantic_projs_text[0](e1)
            p2 = self.semantic_projs_text[1](e2)
            p3 = self.semantic_projs_text[2](e3)
            s_text = p1 + p2 + p3
        if hasattr(self, 'item2code_audio') and self.weight_mlp_audio is not None:
            codes = self.item2code_audio[item_ids]
            weights = torch.softmax(self.weight_mlp_audio(base), dim=-1)
            c1 = codes[..., 0]; c2 = codes[..., 1]; c3 = codes[..., 2]
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            w3 = weights[..., 2].unsqueeze(-1)
            e1 = self.semantic_layers_audio[0](c1) * w1
            e2 = self.semantic_layers_audio[1](c2) * w2
            e3 = self.semantic_layers_audio[2](c3) * w3
            p1 = self.semantic_projs_audio[0](e1)
            p2 = self.semantic_projs_audio[1](e2)
            p3 = self.semantic_projs_audio[2](e3)
            s_audio = p1 + p2 + p3
        if hasattr(self, 'item2code_visual') and self.weight_mlp_visual is not None:
            codes = self.item2code_visual[item_ids]
            weights = torch.softmax(self.weight_mlp_visual(base), dim=-1)
            c1 = codes[..., 0]; c2 = codes[..., 1]; c3 = codes[..., 2]
            w1 = weights[..., 0].unsqueeze(-1)
            w2 = weights[..., 1].unsqueeze(-1)
            w3 = weights[..., 2].unsqueeze(-1)
            e1 = self.semantic_layers_visual[0](c1) * w1
            e2 = self.semantic_layers_visual[1](c2) * w2
            e3 = self.semantic_layers_visual[2](c3) * w3
            p1 = self.semantic_projs_visual[0](e1)
            p2 = self.semantic_projs_visual[1](e2)
            p3 = self.semantic_projs_visual[2](e3)
            s_visual = p1 + p2 + p3
        c_text = torch.zeros_like(base)
        c_audio = torch.zeros_like(base)
        c_visual = torch.zeros_like(base)
        if self.item_content_text is not None and self.content_proj_text is not None:
            raw = self.item_content_text[item_ids]
            c_text = self.content_proj_text(raw)
        if self.item_content_audio is not None and self.content_proj_audio is not None:
            raw = self.item_content_audio[item_ids]
            c_audio = self.content_proj_audio(raw)
        if self.item_content_visual is not None and self.content_proj_visual is not None:
            raw = self.item_content_visual[item_ids]
            c_visual = self.content_proj_visual(raw)
        w_c, _ = self._get_weights(item_ids)
        # Dummy weights for compatibility in this unused function
        w_t = torch.zeros_like(w_c)
        w_a = torch.zeros_like(w_c)
        w_v = torch.zeros_like(w_c)
        if weighted:
            base = base * w_c
            s_text = s_text * w_t
            c_text = c_text * w_t
            s_audio = s_audio * w_a
            c_audio = c_audio * w_a
            s_visual = s_visual * w_v
            c_visual = c_visual * w_v
        base = torch.where(mask.unsqueeze(-1), torch.zeros_like(base), base)
        s_text = torch.where(mask.unsqueeze(-1), torch.zeros_like(s_text), s_text)
        c_text = torch.where(mask.unsqueeze(-1), torch.zeros_like(c_text), c_text)
        s_audio = torch.where(mask.unsqueeze(-1), torch.zeros_like(s_audio), s_audio)
        c_audio = torch.where(mask.unsqueeze(-1), torch.zeros_like(c_audio), c_audio)
        s_visual = torch.where(mask.unsqueeze(-1), torch.zeros_like(s_visual), s_visual)
        c_visual = torch.where(mask.unsqueeze(-1), torch.zeros_like(c_visual), c_visual)
        if concat:
            return torch.cat([base, c_text, s_text, c_audio, s_audio, c_visual, s_visual], dim=-1)
        return {
            'base': base,
            'text_sem': s_text,
            'text_cont': c_text,
            'audio_sem': s_audio,
            'audio_cont': c_audio,
            'visual_sem': s_visual,
            'visual_cont': c_visual,
        }
