import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Lớp Embeddings: Biến đổi các token ID (số nguyên) thành các vector (số thực).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Tạo ma trận Embedding kích thước (vocab_size x d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (Batch_Size, Seq_Len)
        # Theo paper gốc: Nhân embedding với căn bậc 2 của d_model để ổn định variance
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Mã hóa vị trí: Giúp model biết vị trí của từ trong câu (vì Attention không quan tâm thứ tự).
    Sử dụng công thức Sin/Cos cố định.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Tạo ma trận chứa positional encodings: (Seq_Len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Tạo vector vị trí: [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Tính toán số mũ cho mẫu số (div_term)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Áp dụng Sin cho vị trí chẵn
        pe[:, 0::2] = torch.sin(position * div_term)
        # Áp dụng Cos cho vị trí lẻ
        pe[:, 1::2] = torch.cos(position * div_term)

        # Thêm chiều batch để dễ tính toán: (1, Seq_Len, d_model)
        pe = pe.unsqueeze(0)

        # Đăng ký buffer để nó được lưu cùng model state_dict nhưng không phải là tham số train (parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch_Size, Seq_Len, d_model)
        # Cộng embedding ban đầu với positional encoding (lấy đúng độ dài câu hiện tại)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Lớp chuẩn hóa (Layer Norm): Giúp ổn định quá trình train.
    Công thức: (x - mean) / std * gamma + beta
    """
    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        # Tham số học được: Gamma (nhân) và Beta (cộng)
        self.alpha = nn.Parameter(torch.ones(features)) # Gamma
        self.bias = nn.Parameter(torch.zeros(features)) # Beta

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Mạng nơ-ron truyền thẳng.
    Gồm 2 lớp Linear và 1 hàm kích hoạt ReLU (hoặc GELU).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.nn.functional.gelu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
    Cơ chế Attention Đa đầu.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model # Dimension input/output
        self.h = h # Số lượng đầu (heads)
        
        # Đảm bảo d_model chia hết cho số đầu
        assert d_model % h == 0, "d_model phải chia hết cho h"
        
        self.d_k = d_model // h # Dimension của mỗi đầu
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo output layer
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # query, key, value shape: (Batch, h, Seq_Len, d_k)
        d_k = query.shape[-1]
        
        # 1. Tính Dot Product giữa Query và Key (Transposed)
        # (Batch, h, Seq_Len, d_k) @ (Batch, h, d_k, Seq_Len) -> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. Áp dụng Mask (nếu có). Thay giá trị tại vị trí mask=0 thành âm vô cùng (-1e9)
        # Để khi qua Softmax nó sẽ bằng 0.
        if mask is not None:
            # Mask thường có shape (Batch, 1, 1, Seq_Len) hoặc (Batch, 1, Seq_Len, Seq_Len)
            # mask == 0 nghĩa là vị trí cần che
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        # 3. Softmax để ra xác suất attention (tổng bằng 1)
        attention_weights = attention_scores.softmax(dim=-1)
        
        # 4. Dropout (tùy chọn)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
            
        # 5. Nhân với Value
        # (Batch, h, Seq_Len, Seq_Len) @ (Batch, h, Seq_Len, d_k) -> (Batch, h, Seq_Len, d_k)
        return (attention_weights @ value), attention_weights

    def forward(self, q, k, v, mask, past_key_value=None):
        # q, k, v ban đầu có shape (Batch, Seq_Len, d_model)
        
        # 1. Chiếu qua các Linear Layer để lấy Q, K, V
        query = self.w_q(q) # (Batch, Seq_Len_Q, d_model)
        key = self.w_k(k)   # (Batch, Seq_Len_K, d_model)
        value = self.w_v(v) # (Batch, Seq_Len_K, d_model)
        
        # 2. Chia đầu (Split Heads)
        # Biến đổi (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, h, d_k) -> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # --- KV Cache Logic ---
        present_key_value = None
        if past_key_value is not None:
            # past_key_value[0] và [1] là key, value của các bước trước
            # Shape: (Batch, h, Past_Seq_Len, d_k)
            past_key, past_value = past_key_value
            
            # Nối với hiện tại
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            
        # Lưu lại state hiện tại để trả về (nếu cần dùng cho bước sau)
        # Chỉ cache khi đang infer (thường infer thì q chie có len=1)
        # Tuy nhiên để tổng quát, ta cứ trả về full key, value
        present_key_value = (key, value)
        # ----------------------
        
        # 3. Tính Attention
        x, self.attention_weights = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # 4. Ghép đầu (Concat)
        # (Batch, h, Seq_Len, d_k) -> (Batch, Seq_Len, h, d_k) -> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # 5. Chiếu qua lớp Output
        return self.w_o(x), present_key_value

class ResidualConnection(nn.Module):
    """
    Kết nối tắt (Skip Connection) + Layer Norm.
    Kiến trúc sử dụng ở đây là PRE-LN (Norm trước khi vào Sublayer) vì nó ổn định hơn Post-LN.
    x = x + Sublayer(Norm(x))
    """
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # Áp dụng Norm trước -> Sublayer -> Dropout -> Cộng Residual
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    Một khối Encoder gồm:
    1. Multi-Head Self Attention
    2. Feed Forward
    Có Residual Connection bao quanh mỗi cái.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # 1. Self Attention
        # Dùng lambda wrapper để xử lý tuple return từ MHA
        def self_attn_layer(normed_x):
             # MHA trả về (output, cache), ta chỉ cần output cho Encoder
             out, _ = self.self_attention_block(normed_x, normed_x, normed_x, src_mask)
             return out
             
        x = self.residual_connections[0](x, self_attn_layer)
        
        # 2. Feed Forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """
    Chồng chất nhiều EncoderBlock (v.d. 6 lớp).
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features) # Lớp norm cuối cùng

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Một khối Decoder gồm:
    1. Masked Self Attention (Che tương lai)
    2. Cross Attention (Nhìn sang Encoder Output)
    3. Feed Forward
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask, past_key_values=None):
        # past_key_values: (self_attn_cache, cross_attn_cache)
        # self_attn_cache: (K, V) của Self Attn
        # cross_attn_cache: (K, V) của Cross Attn (thường computed sẵn từ encoder output)
        
        self_attn_past = None
        cross_attn_past = None
        
        if past_key_values is not None:
             self_attn_past, cross_attn_past = past_key_values
        
        # 1. Masked Self Attention
        # Do cấu trúc ResidualConnection hiện tại chỉ nhận func trả về Tensor, ta cần tách ra
        # Apply Norm trước (Pre-LN)
        norm_x = self.residual_connections[0].norm(x)
        
        # Calculate Self Attn
        attn_out, self_present = self.self_attention_block(norm_x, norm_x, norm_x, tgt_mask, past_key_value=self_attn_past)
        
        # Apply Dropout & Add Residual
        x = x + self.residual_connections[0].dropout(attn_out)
        
        # 2. Cross Attention
        norm_x = self.residual_connections[1].norm(x)
        attn_out, _ = self.cross_attention_block(norm_x, encoder_output, encoder_output, src_mask)
        x = x + self.residual_connections[1].dropout(attn_out)
        
        # 3. Feed Forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        # Trả về cache mới (chỉ self attn)
        present_key_values = (self_present, None) 
        
        return x, present_key_values

class Decoder(nn.Module):
    """
    Chồng chất nhiều DecoderBlock.
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask, past_key_values=None):
        # past_key_values: List[Tuple(self_cache, cross_cache)]
        present_key_values = []
        
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, encoder_output, src_mask, tgt_mask, past)
            present_key_values.append(present)
            
        return self.norm(x), present_key_values

class ProjectionLayer(nn.Module):
    """
    Lớp chiếu cuối cùng: Biến vector (d_model) thành xác suất (vocab_size).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, Vocab_Size)
        # Lưu ý: Chưa dùng Softmax ở đây vì hàm loss CrossEntropyLoss của PyTorch đã bao gồm LogSoftmax.
        return self.proj(x)

class Transformer(nn.Module):
    """
    Container chính chứa toàn bộ kiến trúc Transformer.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (Batch, Seq_Len) -> (Batch, Seq_Len, d_model)
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask, past_key_values=None, use_cache=False):
        # (Batch, Seq_Len) -> (Batch, Seq_Len, d_model)
        
        # --- Logic Embedding & PE ---
        x = self.tgt_embed(tgt)
        
        if past_key_values is not None:
            use_cache = True
            # Lấy chiều dài quá khứ từ layer đầu tiên (nếu có cache)
            # past_key_values[0] is (self_cache, cross_cache)
            # self_cache is (k, v)
            # k is (Batch, h, Seq, d_k)
            if past_key_values[0][0] is not None:
                past_len = past_key_values[0][0][0].size(2)
            else:
                past_len = 0
                
            # Slice PE đúng vị trí
            pe_slice = self.tgt_pos.pe[:, past_len : past_len + x.shape[1], :]
            x = x + pe_slice.to(x.device)
            x = self.tgt_pos.dropout(x)
        else:
            x = self.tgt_pos(x)
        # ----------------------------

        decoder_output, present_key_values = self.decoder(x, encoder_output, src_mask, tgt_mask, past_key_values)
        
        if not use_cache:
            return decoder_output
            
        return decoder_output, present_key_values

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Hàm Factory giúp khởi tạo Transformer dễ dàng với các tham số chuẩn.
    N: số lượng layer (blocks)
    h: số lượng heads
    """
    # 1. Kiểm tra điều kiện
    # Vì sử dụng Weight Tying, kích thước từ vựng nguồn và đích phải bằng nhau (do dùng chung Tokenizer)
    assert src_vocab_size == tgt_vocab_size, \
        f"Weight Tying requires equal vocab sizes, but got src={src_vocab_size} vs tgt={tgt_vocab_size}."
        
    # 2. Khởi tạo Embeddings
    # Positional Encoding nên được init lớn hơn max_len thực tế để tránh lỗi khi decode dài hơn dự kiến (ví dụ 5000)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # --- Áp dụng Weight Tying (Chia sẻ trọng số: 3 chiều) ---
    # a. Chia sẻ Source & Target Embedding
    tgt_embed.embedding = src_embed.embedding
    
    src_pos = PositionalEncoding(d_model, max(5000, src_seq_len), dropout)
    tgt_pos = PositionalEncoding(d_model, max(5000, tgt_seq_len), dropout)
    
    # 3. Tạo các Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
        
    # 4. Tạo các Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
        
    # 5. Lắp ráp Encoder & Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # b. Chia sẻ Embedding & Output Projection
    # Trọng số lớp chiếu (Projection) = Trọng số lớp Embedding
    projection_layer.proj.weight = src_embed.embedding.weight
    # -------------------------------------------------------------
    
    # 6. Khởi tạo Model hoàn chỉnh
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # 7. Khởi tạo tham số (Xavier Initialization)
    # Giúp model hội tụ nhanh và ổn định hơn
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer
