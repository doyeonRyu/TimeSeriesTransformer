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

class TimeSeriesTransformer(nn.Module):
    '''
    설명:
        - 인코더 전용 Transformer로 과거 시계열을 인코딩하고,
          대표 벡터를 통해 다중 스텝(output_window) 전력 값을 직접 예측한다.
        - 입력 특성 수(input_dim)는 1로 고정하지 않고 가변(예: 4: idx/temp/rain/elec)으로 처리한다.

    입력값:
        - x: 텐서, shape = (batch, input_window, input_dim)

    출력값:
        - y_hat: 텐서, shape = (batch, output_window, 1)
    '''
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
        pool="last",         # "last" 또는 "mean" 선택
        head_hidden=256,     # 예측 헤드 내부 은닉 차원
    ):
        super(TimeSeriesTransformer, self).__init__()

        # 1) 임베딩: 실수 시계열 → d_model 차원 + 위치 인코딩
        self.embedding = TimeSeriesEmbedding(input_dim, d_model, max_len, dropout=dropout, use_scale=True)

        # 2) Transformer Encoder 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # batch_first=True로 (B, L, d) 유지
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 인코더 출력 안정화를 위한 최종 LayerNorm
        self.enc_norm = nn.LayerNorm(d_model)

        # 3) 풀링 방식 저장
        self.pool = pool # "last" 또는 "mean"

        # 4) 예측 헤드: d_model → output_window*1
        self.output_window = output_window
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),  # d_model → hidden
            nn.ReLU(),                        # 비선형 활성화
            nn.Dropout(dropout),              # 과적합 완화
            nn.Linear(head_hidden, output_window)  # hidden → output_window
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) 입력
        h = self.embedding(x)          # (B, L, d_model)  임베딩 + 위치 인코딩
        h = self.encoder(h)            # (B, L, d_model)  인코더 통과
        h = self.enc_norm(h)           # (B, L, d_model)  최종 정규화

        # 대표 벡터 추출: 마지막 타임스텝
        if self.pool == "last":
            rep = h[:, -1, :]          # (B, d_model)
        else:
            rep = h.mean(dim=1)        # (B, d_model)

        # 예측 헤드 통과
        y = self.head(rep)             # (B, output_window)

        # 채널(출력 변수) 차원 추가 → (B, output_window, 1)
        y = y.unsqueeze(-1)
        return y
