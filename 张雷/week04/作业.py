import torch
import torch.nn as nn
import math


class BertSelfAttention(nn.Module):
    """
    多头自注意力层。将 hidden_size 均分到 num_attention_heads 个头，
    每个头独立计算缩放点积注意力，最后拼接回 hidden_size。
    """
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        # Q、K、V 投影矩阵
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # 注意力输出投影矩阵
        self.output = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        将 [batch, seq_len, hidden_size] 拆成多头形式 [batch, num_heads, seq_len, head_size]
        每个 batch 样本的 hidden_size 维度被切分成 num_heads 份，便于并行计算各头注意力。
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        # 缩放点积注意力: softmax(Q @ K^T / sqrt(d_k)) @ V
        # 除以 sqrt(d_k) 防止点积过大导致 softmax 梯度消失
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # mask 位置加上一个很大的负数，经过 softmax 后权重趋近于 0
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)

        # 恢复为 [batch, seq_len, hidden_size]：先调回 batch-first，再拼接所有头
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.all_head_size)

        return self.output(context)


class BertFeedForward(nn.Module):
    """前馈网络: linear(4h) -> GELU -> linear(h) -> dropout"""
    def __init__(self, hidden_size=768, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def gelu(self, x):
        """GELU 激活函数的 tanh 近似形式，与原始 BERT 一致"""
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))

    def forward(self, x):
        x = self.gelu(self.dense1(x))
        x = self.dropout(self.dense2(x))
        return x


class BertTransformerLayer(nn.Module):
    """
    单层 BERT Transformer（Post-LN 结构）:
      x = LayerNorm(x + Dropout(Attention(x)))
      x = LayerNorm(x + Dropout(FFN(x)))
    残差连接解决深层网络退化问题，LayerNorm 放在残差之后（post-norm）稳定训练。
    """
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.feed_forward = BertFeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        attention_output = self.attention(x, attention_mask)
        x = self.attention_layer_norm(x + self.dropout(attention_output))

        ffn_output = self.feed_forward(x)
        x = self.ffn_layer_norm(x + self.dropout(ffn_output))

        return x


class BertEmbeddings(nn.Module):
    """
    三种嵌入求和: word + position + token_type，后接 LayerNorm + Dropout。
    position embedding 从 0 开始编号，token_type 默认全 0（单句场景）。
    """
    def __init__(self, vocab_size=21128, hidden_size=768, max_position_embeddings=512,
                 type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = self.layer_norm(word_emb + pos_emb + type_emb)
        embeddings = self.dropout(embeddings)
        return embeddings


class DiyBertModel(nn.Module):
    """简易 BERT 模型：embedding -> N 层 transformer -> pooler"""
    def __init__(self, vocab_size=21128, hidden_size=768, num_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings,
                                         dropout=dropout)
        self.layers = nn.ModuleList([
            BertTransformerLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 将 [batch, seq_len] 的 0/1 mask 扩展为 [batch, 1, 1, seq_len]，
        # 并转为 additive mask: 1 -> 0 (保留), 0 -> -10000 (屏蔽)
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :].float()
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        x = self.embeddings(input_ids, token_type_ids)

        for layer in self.layers:
            x = layer(x, extended_attention_mask)

        # 取 [CLS] 位置（第一个 token）的输出做池化
        pooled_output = self.pooler(x[:, 0])
        return x, pooled_output


if __name__ == "__main__":
    model = DiyBertModel(
        vocab_size=21128,
        hidden_size=768,
        num_layers=1,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    input_ids = torch.randint(0, 21128, (2, 10))
    attention_mask = torch.ones(2, 10)

    model.eval()
    with torch.no_grad():
        sequence_output, pooled_output = model(input_ids, attention_mask=attention_mask)

    print(f"输入 shape: {input_ids.shape}")          # [2, 10]
    print(f"序列输出 shape: {sequence_output.shape}")  # [2, 10, 768]
    print(f"池化输出 shape: {pooled_output.shape}")    # [2, 768]
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
