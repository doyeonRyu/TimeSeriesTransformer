import torch
import torch.nn as nn
import math

# 임베딩 및 위치 인코딩
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# 실수 입력을 벡터 공간으로 표현해 패턴 인식 가능하게 함
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, max_len, dropout=0.1, use_scale=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_scale = use_scale
        self.scale = math.sqrt(d_model) if use_scale else 1.0  # √d_model

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, L, _ = x.shape
        assert L <= self.pos_encoder.pe.size(1), "seq_len이 max_len을 초과"

        x = self.input_proj(x) / self.scale          # (B, L, d_model), 스케일링
        x = self.pos_encoder(x)                      # 위치 인코딩 더하기
        x = self.norm(x)                             # 안정화
        x = self.dropout(x)                          # 정규화 후 드롭아웃
        return x


class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, top_factor=5):
        """
        d_model: 임베딩 차원
        n_heads: 어텐션 헤드 수
        top_factor: 선택할 쿼리 개수를 줄이는 비율 조절 (ex. L_q // top_factor)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_factor = top_factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        """
        queries: (B, L_q, d_model)
        keys: (B, L_k, d_model)
        values: (B, L_k, d_model)
        """
        B, L_q, _ = queries.size()
        _, L_k, _ = keys.size()
        H, D = self.n_heads, self.head_dim

        # 1. Q, K, V 생성 후 멀티헤드 형태로 reshape
        q = self.q_proj(queries).view(B, L_q, H, D).transpose(1, 2)  # (B, H, L_q, D)
        k = self.k_proj(keys).view(B, L_k, H, D).transpose(1, 2)     # (B, H, L_k, D)
        v = self.v_proj(values).view(B, L_k, H, D).transpose(1, 2)   # (B, H, L_k, D)

        # 2. 쿼리-키 내적값의 L2 노름(유사도 세기)으로 중요 쿼리 선택
        #    (원 논문에서 제안된 sparse attention 선택 기준)
        q_norm = q / torch.norm(q, dim=-1, keepdim=True)  # (B, H, L_q, D)
        k_norm = k / torch.norm(k, dim=-1, keepdim=True)  # (B, H, L_k, D)
        scores_all = torch.matmul(q_norm, k_norm.transpose(-2, -1))  # (B, H, L_q, L_k)
        mean_scores = scores_all.mean(dim=-1)  # (B, H, L_q)

        # 3. top u개의 쿼리만 선택 (u ≈ L_q / top_factor)
        u = max(1, L_q // self.top_factor)
        top_indices = torch.topk(mean_scores, u, dim=-1)[1]  # (B, H, u)

        # 4. 선택된 쿼리만 어텐션 계산
        q_selected = torch.gather(q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D))
        scores = torch.matmul(q_selected, k.transpose(-2, -1)) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1)
        context_selected = torch.matmul(attn, v)  # (B, H, u, D)

        # 5. context_selected를 원래 길이로 복원 (나머지 쿼리는 0)
        context = torch.zeros(B, H, L_q, D, device=queries.device)
        context.scatter_(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D), context_selected)

        # 6. 멀티헤드 합치고 출력 투영
        out = context.transpose(1, 2).contiguous().view(B, L_q, H * D)
        return self.out_proj(out)

    
class EncoderLayer(nn.Module):
    '''
    설명:
        - Informer 스타일의 인코더 레이어.
        - ProbSparseSelfAttention을 사용하고, Pre-LN(레이어 앞쪽 정규화) 구조를 채택.
        - FFN은 (d_model → dim_feedforward → d_model)로 구성.
    입력값:
        - x: 텐서, shape = (B, L, d_model)
    출력값:
        - x: 텐서, shape = (B, L, d_model)
    '''

    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1, activation='relu'):
        super().__init__()
        # 1) 하이퍼파라미터 저장
        self.d_model = d_model                                 # d_model 차원 저장
        self.n_heads = n_heads                                 # 멀티헤드 개수 저장
        self.dim_feedforward = dim_feedforward or (4*d_model)  # FFN 내부 차원, 미지정 시 4*d_model
        self.dropout_p = dropout                               # 드롭아웃 확률
        self.activation = activation                           # 활성화 함수 유형(기본 relu)

        # 2) ProbSparse Self-Attention
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)  # ProbSparse 어텐션 모듈

        # 3) Pre-LN용 LayerNorm (어텐션 앞/FFN 앞에 각각 배치)
        self.norm1 = nn.LayerNorm(d_model)                     # 어텐션 전에 정규화
        self.norm2 = nn.LayerNorm(d_model)                     # FFN 전에 정규화

        # 4) 드롭아웃
        self.dropout1 = nn.Dropout(dropout)                    # 어텐션 출력 드롭아웃
        self.dropout2 = nn.Dropout(dropout)                    # FFN 출력 드롭아웃

        # 5) FFN: d_model → dim_feedforward → d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward),          # 첫 선형층
            nn.ReLU() if activation == 'relu' else nn.GELU(),  # 활성화 (기본 ReLU, 옵션 GELU)
            nn.Dropout(dropout),                               # 중간 드롭아웃
            nn.Linear(self.dim_feedforward, d_model)           # 복원 선형층
        )

    def forward(self, x):
        '''
        설명:
            - Pre-LN 구조로, 서브레이어(어텐션/FFN) 앞에 LayerNorm을 적용.
            - 어텐션과 FFN 각각 잔차 연결(residual connection) 포함.
        입력값:
            - x: 텐서, shape = (B, L, d_model)
        출력값:
            - x: 텐서, shape = (B, L, d_model)
        '''
        # [어텐션 블록: Pre-LN]
        x_norm = self.norm1(x)                                 # 어텐션 전에 정규화
        attn_out = self.self_attn(x_norm, x_norm, x_norm)      # ProbSparse Self-Attention 수행
        x = x + self.dropout1(attn_out)                        # 잔차 연결 + 드롭아웃

        # [FFN 블록: Pre-LN]
        x_ffn_in = self.norm2(x)                               # FFN 전에 정규화
        ff_out = self.ff(x_ffn_in)                             # FFN 통과
        x = x + self.dropout2(ff_out)                          # 잔차 연결 + 드롭아웃

        return x                                               # (B, L, d_model) 그대로 반환

class DecoderLayer(nn.Module):
    '''
    설명:
        - Informer 스타일의 디코더 레이어.
        - 서브레이어 앞에 LayerNorm을 두는 Pre-LN 구조를 사용해 학습 안정성을 개선.
        - Self-Attention(디코더 자기 자신) → Cross-Attention(인코더 출력과의 상호작용) → FFN 순으로 처리.
        - ProbSparseSelfAttention을 사용하므로 긴 시퀀스에서 계산량을 절감.
    입력값:
        - x: 텐서, shape = (B, L_dec, d_model)
        - enc_out: 텐서, shape = (B, L_enc, d_model)
    출력값:
        - x: 텐서, shape = (B, L_dec, d_model)
    '''
    def __init__(self, d_model, n_heads, dropout=0.1, dim_feedforward=None, activation='relu'):
        super().__init__()
        # 하이퍼파라미터 저장
        self.d_model = d_model                                 # 모델 차원
        self.n_heads = n_heads                                 # 어텐션 헤드 수
        self.dim_feedforward = dim_feedforward or (4*d_model)  # FFN 내부 차원(기본 4*d_model)
        self.dropout_p = dropout                               # 드롭아웃 확률
        self.activation = activation                           # 활성화 함수 종류

        # 1) Self-Attention (디코더 내부)
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)

        # 2) Cross-Attention (인코더 출력과의 어텐션)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads)

        # 3) FFN: d_model → dim_feedforward → d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward),          # 확장 선형층
            nn.ReLU() if activation == 'relu' else nn.GELU(),  # 활성화
            nn.Dropout(dropout),                               # 중간 드롭아웃
            nn.Linear(self.dim_feedforward, d_model)           # 복원 선형층
        )

        # 4) Pre-LN을 위한 LayerNorm들 (각 서브레이어 앞)
        self.norm1 = nn.LayerNorm(d_model)  # self-attn 앞
        self.norm2 = nn.LayerNorm(d_model)  # cross-attn 앞
        self.norm3 = nn.LayerNorm(d_model)  # ffn 앞

        # 5) 서브레이어 출력 드롭아웃
        self.dropout1 = nn.Dropout(dropout) # self-attn 경로
        self.dropout2 = nn.Dropout(dropout) # cross-attn 경로
        self.dropout3 = nn.Dropout(dropout) # ffn 경로

    def forward(self, x, enc_out):
        # x: (B, L_dec, d_model), enc_out: (B, L_enc, d_model)

        # [Self-Attention 블록: Pre-LN]
        x_norm = self.norm1(x)                                 # 1) 정규화로 분포 안정화
        sa_out = self.self_attn(x_norm, x_norm, x_norm)        # 2) 디코더 내부 자기-어텐션
        x = x + self.dropout1(sa_out)                          # 3) 잔차 연결 + 드롭아웃

        # [Cross-Attention 블록: Pre-LN]
        x_norm = self.norm2(x)                                 # 1) 정규화
        ca_out = self.cross_attn(x_norm, enc_out, enc_out)     # 2) 인코더 출력과의 교차 어텐션
        x = x + self.dropout2(ca_out)                          # 3) 잔차 연결 + 드롭아웃

        # [FFN 블록: Pre-LN]
        x_norm = self.norm3(x)                                 # 1) 정규화
        ff_out = self.ff(x_norm)                               # 2) 위치별 비선형 변환
        x = x + self.dropout3(ff_out)                          # 3) 잔차 연결 + 드롭아웃

        return x                                               # (B, L_dec, d_model)

'''
설명:
    - TimeSeriesTransformer와 "동일한 인터페이스/출력 규격"을 갖는 Informer 변형.
    - 입력 임베딩, 파라미터 이름/의미, 출력 형태를 그대로 유지:
        입력 x: (B, input_window, input_dim)
        출력 y: (B, output_window, 1)
    - 내부 인코더는 ProbSparseSelfAttention을 사용하는 Informer 스타일 EncoderLayer 스택을 사용.
    - 기본은 인코더-온리(MIMO 회귀): 풀링(last/mean) + 예측 헤드로 output_window를 한 번에 출력.
    - 옵션으로 use_decoder=True 설정 시, 디코더 경로(DecoderLayer 스택 + projection) 사용 가능.

입력값:
    - input_dim: 입력 특성 수(F)
    - d_model: 임베딩 차원
    - nhead: 어텐션 헤드 수
    - num_layers: 인코더 레이어 수(= e_layers)
    - dim_feedforward: FFN 내부 차원(없으면 4*d_model)
    - output_window: 예측할 시점 수
    - dropout: 드롭아웃 확률
    - max_len: 위치 인코딩 최대 길이(임베딩에서 체크)
    - pool: 대표 벡터 추출 방식("last" | "mean")
    - head_hidden: 예측 헤드 은닉 차원
    - use_decoder: 디코더 사용 여부(False가 기본)
    - d_layers: 디코더 레이어 수(use_decoder=True일 때만 사용)

출력값:
    - y_hat: 텐서, shape = (B, output_window, 1)

주의:
    - TimeSeriesEmbedding, ProbSparseSelfAttention, EncoderLayer, DecoderLayer가 이미 정의되어 있다고 가정.
    - TimeSeriesEmbedding은 기존 코드와 동일하게 √d_model로 "나눔" 스케일을 유지(네 코드와 완전 정합).
'''

class Informer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        output_window,
        dropout=0.1,
        max_len=500,
        pool="last",
        head_hidden=256,
        use_decoder=False,
        d_layers=1,
    ):
        super().__init__()
        # 1) 임베딩: 기존 TimeSeriesEmbedding을 그대로 사용해 규약/스케일 일치
        self.embedding = TimeSeriesEmbedding(input_dim, d_model, max_len, dropout=dropout, use_scale=True)  # (B,L,F)->(B,L,d)

        # 2) Informer 스타일 인코더 스택: ProbSparse 기반 EncoderLayer 사용
        self.encoder = nn.Sequential(*[
            EncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
            for _ in range(num_layers)
        ])

        # 3) 인코더 출력 정규화(안정화)
        self.enc_norm = nn.LayerNorm(d_model)

        # 4) 대표 벡터 추출 방식 저장("last" | "mean")
        self.pool = pool
        self.output_window = output_window

        # 5) 예측 헤드: 인코더-온리일 때 사용 (d_model -> head_hidden -> output_window)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, output_window)
        )

        # 6) 디코더 경로(옵션): Informer 원형을 쓰고 싶을 때 활성화
        self.use_decoder = use_decoder
        if use_decoder:
            # 디코더 입력 토큰(학습형), shape: (1, output_window, d_model)
            self.decoder_input = nn.Parameter(torch.randn(1, output_window, d_model))
            # 디코더 레이어 스택
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(d_model, nhead, dropout=dropout, dim_feedforward=dim_feedforward, activation='relu')
                for _ in range(d_layers)
            ])
            # 디코더 출력 → 1차원 타깃으로 사영
            self.projection = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        B = x.size(0)                                  # 배치 크기 추출
        h = self.embedding(x)                          # (B, L, d_model): 실수 임베딩 + 위치 인코딩 + Norm/Dropout
        h = self.encoder(h)                            # (B, L, d_model): ProbSparse 인코더 통과
        h = self.enc_norm(h)                           # (B, L, d_model): 최종 정규화

        if not self.use_decoder:
            # [인코더-온리 경로] 대표 벡터 추출
            if self.pool == "last":
                rep = h[:, -1, :]                      # (B, d_model): 마지막 타임스텝
            else:
                rep = h.mean(dim=1)                    # (B, d_model): 평균 풀링

            y = self.head(rep)                         # (B, output_window): 한 번에 멀티스텝 회귀
            y = y.unsqueeze(-1)                        # (B, output_window, 1): 출력 채널 차원 추가
            return y

        else:
            # [디코더 경로] Informer 원형 방식(autoregressive 입력 토큰 사용)
            dec = self.decoder_input.repeat(B, 1, 1)   # (B, output_window, d_model): 배치 크기에 맞춰 복제
            for layer in self.decoder_layers:
                dec = layer(dec, h)                    # 디코더 self-attn + cross-attn + FFN
            out = self.projection(dec)                 # (B, output_window, 1): 타깃 차원으로 사영
            return out
