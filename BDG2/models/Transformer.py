import torch
import torch.nn as nn
import math

# 임베딩 및 위치 인코딩
class PositionalEncoding(nn.Module):
    """
    Class: PositionalEncoding
        - Transformer의 위치 인코딩 구현: 위치 정보를 임베딩에 추가
        - 시계열 데이터에 위치 정보를 sin/cos 함수로 주기적으로 인코딩
        - TimeSeriesEmbedding에서 사용됨
    Parameters:
        - d_model (int): 임베딩 차원 수 
        - max_len (int): 최대 시퀀스 길이 (기본값: 500)
    foward parameters:
        - x (torch.Tensor): 입력 텐서, shape = (batch, seq_len, d_model)
    Return values:
        - x (torch.Tensor): 위치 인코딩이 추가된 텐서, shape = 동일 (batch, seq_len, d_model)
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        # (max_len, d_model) 크기의 0으로 채워진 텐서 생성
        pe = torch.zeros(max_len, d_model)

        # position: 0부터 max_len-1까지의 위치 인덱스, shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # div_term: 주기 조절을 위한 분모항
        # d_model의 절반 개수만큼 생성, 지수 함수로 스케일링
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 짝수 인덱스(0, 2, 4, ...)에 대해 sin 함수 적용 
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스(1, 3, 5, ...)에 대해 cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe shape; (1, max_len, d_model)
        # register_buffer -> 학습 파라미터로는 취급하지 않지만, 모델 저장/로드 시 저장되는 버퍼로 등록
        # 쓰임새: 위치 인코딩을 모델의 일부로 취급하여 GPU/CPU 간 이동 가능
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 위치 인코딩 중에서 seq_len 만큼만 잘라서 입력에 더함
        return x + self.pe[:, :x.size(1)]

class TimeSeriesEmbedding(nn.Module):
    """
    Class: TimeSeriesEmbedding
        - 시계열 실수 입력을 Transformer가 처리할 수 있는 벡터 임베딩 공간(d_model 차원)으로 변환하는 임베딩 모듈.
    Parameters:
        - input_dim (int): 입력 시계열의 특성 수 (timestamp_idx/temp/rain/elec)
        - d_model (int): Transformer 모델의 임베딩 차원
        - max_len (int): 최대 시퀀스 길이
        - dropout (float): 드롭아웃 비율 (기본값: 0.1)
        - use_scale (bool): 스케일링 여부, True면 √d_model로 나누기
    foward parameters:
        - x (torch.Tensor): 입력 텐서, shape = (batch, seq_len, input_dim)
    Return values:
        - x (torch.Tensor): 임베딩된 텐서, shape = (batch, seq_len, d_model)
        - 위치 인코딩 + LayerNorm + Dropout이 적용 완료된 임베딩 
    """
    def __init__(self, input_dim, d_model, max_len, dropout=0.1, use_scale=True):
        super().__init__()
        # input_dim -> d_model 차원으로 선형 변환
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩 모듈 (PositionalEncoding) 초기화
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        # 각 시퀀스 위치의 벡터 분포를 평균 0, 분산 1로 정규화
        # 시퀀스의 각 시점별 벡터를 기준으로 정규화함
        self.norm = nn.LayerNorm(d_model)
        # 과적합 방지를 위한 드롭아웃
        # 일부 뉴런을 랜덤하게 0으로 설정하여 학습 안정화
        self.dropout = nn.Dropout(dropout)
        # 스케일링 옵션
        self.use_scale = use_scale
        self.scale = math.sqrt(d_model) if use_scale else 1.0  # √d_model
        # Transformer 논문에서 제안: 입력을 √d_model로 나누어 안정적인 학습 유도

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, L, _ = x.shape # B: batch size, L: seq_len

        # 시퀀스 길이 L이 위치 인코딩의 max_len보다 길면 에러 발생
        # seq_len(L): 한 입력 시퀀스의 시간 길이 (= input_window)
        assert L <= self.pos_encoder.pe.size(1), "seq_len이 max_len을 초과함."

        # 1) 입력을 d_model 차원으로 투영 & 스케일링
        x = self.input_proj(x) / self.scale # shape: (B, L, d_model)
       
        # 2) 위치 인코딩 더하기 
        x = self.pos_encoder(x) # shape: (B, L, d_model)
        
        # 3)  Layer Normalization으로 안정화
        x = self.norm(x) # shape: (B,L, d_model)
        
        # 4) 드롭아웃 적용 
        x = self.dropout(x) # shape: (B, L, d_model)
        
        # 5) 완성된 시계열 임베딩 반환             
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Class: TimsSeriesTransformer
        - Transformer encoder 기반 시계열 데이터 예측 모델
        - 실수 시계열 입력(F)을 d_model 차원으로 임베딩(위치 인코딩 포함)한 뒤 
        Transformer Encoder를 통과시켜, 풀링된 대표 벡터로 다단계 에측을 수행하는 인코더 기반 모델
        - 디코더 없이, 마지막/평균 풀링된 인코더 출력 -> MLP 헤드로 여러 스텝을 한 번에 회귀(오차 누적 방지)
    Parameters:
        - input_dim (int): 입력 피처 수 F (timestamp_idx/temp/rain/elec)
        - d_model (int): Transformer 모델의 임베딩 차원
        - nhead (int): 멀티헤드 어텐션의 헤드 수
        - num_layers (int): Transformer 인코더 레이어 수
        - dim_feedforward (int): 인코더 내부 FFN 차원
        - output_window (int): 예측할 출력 윈도우 크기 (24시간)
        - dropout (float): 드롭아웃 비율 (기본값: 0.1)
        - max_len (int): 최대 시퀀스 길이 (기본값: 500)
        - pool (str): 풀링 방식 ("last" 또는 "mean") (기본값: "last")
        - head_hidden (int): 예측 헤드 내부 은닉 차원 (기본값: 256)
    foward parameters:
        - x (torch.Tensor): 입력 텐서, shape = (batch, seq_len, input_dim)
    Return values:
        - y (torch.Tensor): 예측 결과, shape = (batch, output_window, 1)
        - 미래 스텝별 단일 타깃 회귀
    Note:
        - 인코더만 사용하므로, 미래 정보 누출 방지를 위해 causal mask를 추가
        - 여러 타깃 채널 예측 시: 
            - 최종 linear 출력 차원을 output_window*target_dim으로 바꾸고 reshape 필요 
    """
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
    ):
        super(TimeSeriesTransformer, self).__init__()

        # 1) 임베딩: 실수 시계열 → d_model 차원 + 위치 인코딩
        self.embedding = TimeSeriesEmbedding(input_dim, d_model, max_len, dropout=dropout, use_scale=True)
        ## 입력 [B, L, F] → Linear로 [B, L, d_model], 
        ## PositionalEncoding으로 위치 정보 추가
        ## LayerNorm으로 안정화, Dropout으로 과적합 방지

        # 2) Transformer Encoder 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # 각 토큰(시점) 벡터의 차원
            nhead=nhead, # 멀티 헤드 수 
            dim_feedforward=dim_feedforward, # FFN 내부 차원
            dropout=dropout,
            batch_first=True # 입력/출력 텐서 shape을 [B, L, d] 유지
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 스택된 인코더 레이어: self-attention + FFN 블록을 num_layers 만큼 반복

        # 인코더 출력 안정화를 위한 최종 LayerNorm
        self.enc_norm = nn.LayerNorm(d_model)

        # 3) 풀링 방식 저장
        self.pool = pool # "last"(마지막 시점 벡터) 또는 "mean"(시간 평균)

        # 4) 예측 헤드: [B, d_model] → [B, output_window*1]
        self.output_window = output_window
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden), # d_model → hidden
            nn.ReLU(), # 비선형 활성화
            nn.Dropout(dropout), # 과적합 완화
            nn.Linear(head_hidden, output_window) # hidden → output_window (단계별 예측)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function: forward
            - 시계열 배치를 임베딩 -> 인코더 통과 -> 풀링 -> 미래 output_window 스텝 회귀
        Parameters:
            - x (torch.Tensor): 입력 텐서, shape = (batch(B), seq_len(L), input_dim(F))
        Return values:
            - y (torch.Tensor): 예측 결과, shape = (batch(B), output_window, 1)
        """
        # x: (B, L, F) 입력
        h = self.embedding(x) # (B, L, d_model)  임베딩 + 위치 인코딩 + Norm/Dropout
        h = self.encoder(h) # (B, L, d_model)  인코더 통과 (현재 마스크 미적용)
        h = self.enc_norm(h) # (B, L, d_model)  최종 정규화로 안정화

        # 대표 벡터 추출
        if self.pool == "last":
            rep = h[:, -1, :] # (B, d_model) 마지막 시점 벡터 사용
        else:
            rep = h.mean(dim=1) # (B, d_model) 시간 평균 풀링

        # 예측 헤드 통과
        y = self.head(rep) # (B, output_window)

        # 채널(출력 변수) 차원 추가 → (B, output_window, 1)
        y = y.unsqueeze(-1)
        return y
