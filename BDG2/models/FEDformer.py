# 기본 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAverage(nn.Module):
    '''
    설명:
        - 시계열 길이 축(axis=1, seq_len) 방향으로 이동 평균을 계산하여
          저주파 성분(Trend)을 추출하는 모듈.
        - 입력/출력 형태는 (B, L, d_model)을 유지한다.
    입력값:
        - x: 텐서, shape = (batch, seq_len, d_model)
    출력값:
        - avg: 텐서, shape = (batch, seq_len, d_model), 이동 평균 결과(Trend)
    '''
    def __init__(self, kernel_size: int):
        super(MovingAverage, self).__init__()
        # 커널 크기 저장
        self.kernel_size = int(kernel_size)                   # 정수형 보장
        assert self.kernel_size >= 1, "kernel_size는 1 이상이어야 합니다."
        # 짝수/홀수 모두 길이 보존되도록 좌우 패딩을 분리 계산
        self.pad_left = (self.kernel_size - 1) // 2          # 왼쪽 패딩
        self.pad_right = self.kernel_size - 1 - self.pad_left# 오른쪽 패딩

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape                                     # 배치, 길이, 차원 분해
        x = x.permute(0, 2, 1)                                # (B, D, L)로 변환 → 1D 풀링 입력 형태

        # 좌우 패딩(테두리 값을 복제하여 왜곡 최소화)
        x_padded = F.pad(x, (self.pad_left, self.pad_right), mode='replicate')  # (B, D, L + pad_left + pad_right)

        # 커널 크기=kernel_size, 보폭=1로 평균 풀링 → 길이 유지되도록 설계
        avg = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)    # (B, D, L)

        avg = avg.permute(0, 2, 1).contiguous()                 # (B, L, D)로 복원
        return avg

# 시계열 분해 모듈 (Trend + Seasonal) # Autoformer와 동일
class SeriesDecomposition(nn.Module):
    '''
    설명:
        - 시계열 x를 이동평균으로 추출한 Trend와, 나머지 성분(Seasonal)으로 분해한다.
        - 입력/출력 텐서의 형태는 (batch, seq_len, d_model)을 유지한다.

    입력값:
        - x: 텐서, shape=(B, L, D)
            시계열 임베딩(또는 히든 상태)

    출력값:
        - seasonal: 텐서, shape=(B, L, D)
            계절/잔차 성분(= x - trend)
        - trend: 텐서, shape=(B, L, D)
            이동 평균으로 추출한 추세 성분
    '''
    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size)  # 이동 평균 모듈(길이 보존)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D) 입력 텐서
        trend = self.moving_avg(x)      # (B, L, D) 이동 평균으로 추세 추출
        seasonal = x - trend            # (B, L, D) 원 신호에서 추세를 뺀 잔차/계절 성분
        return seasonal, trend          # (B, L, D), (B, L, D)

class FourierBlock(nn.Module):
    '''
    설명:
        - 입력 x의 주파수 스펙트럼(rFFT)에서 에너지가 큰 상위 mode_select_num개 모드만 남기고
          나머지는 0으로 마스킹한 뒤, irFFT로 시간영역 (B, L, D)로 복원한다.
        - FEDformer의 "frequency-domain filtering"을 단순화한 형태.

    입력값:
        - x: 텐서, shape=(B, L, D)
            시간영역 히든 시퀀스

    출력값:
        - y: 텐서, shape=(B, L, D)
            상위 주파수 모드만 반영된 시간영역 시퀀스
    '''
    def __init__(self, mode_select_num: int):
        super().__init__()
        self.mode_select_num = mode_select_num  # 선택할 주파수 모드 개수

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape                                      # 배치, 길이, 차원 읽기
        X = torch.fft.rfft(x, n=L, dim=1)                      # (B, Lf, D) 복소수 스펙트럼 (Lf = L//2 + 1)

        # 모드 에너지 계산: 채널 평균으로 주파수별 크기 추정
        mag = X.abs().mean(dim=-1)                             # (B, Lf)

        # 상위 k개 주파수 인덱스 선택
        Lf = X.size(1)                                         # 양수 주파수 개수
        k = min(self.mode_select_num, Lf)                      # 과도 선택 방지
        top_idx = torch.topk(mag, k=k, dim=1).indices          # (B, k)

        # 배치별 마스크 생성(선택한 모드만 True)
        mask = torch.zeros(B, Lf, device=x.device, dtype=torch.bool)  # (B, Lf)
        mask.scatter_(1, top_idx, True)                        # 배치별로 top-k 위치 True
        mask = mask.unsqueeze(-1).expand(-1, -1, D)            # (B, Lf, D)로 브로드캐스트

        # 마스킹 적용: 선택 모드만 남기고 나머지는 0
        X_filtered = torch.where(mask, X, torch.zeros_like(X)) # (B, Lf, D), 복소 dtype 유지

        # 시간영역으로 복원 (길이 L 유지)
        y = torch.fft.irfft(X_filtered, n=L, dim=1)            # (B, L, D)

        return y

class FEDformer(nn.Module):
    '''
    설명:
        - 입력 시계열을 선형 임베딩(input_dim → d_model) 후 이동평균 기반 분해(Seasonal/Trend)를 수행.
        - Seasonal 성분에 대해 FourierBlock(상위 주파수 모드 선택 후 irFFT 복원)을 적용한다.
        - 시간축 평균 풀링으로 대표 벡터를 만들고, MIMO 헤드로 한 번에 output_window 길이 예측을 출력한다.
        - 최종 출력 형태는 (B, output_window, 1)이다.

    입력값:
        - x: 텐서, shape=(B, L, input_dim)

    출력값:
        - y_hat: 텐서, shape=(B, output_window, 1)
    '''
    def __init__(self,
                input_dim, 
                d_model, 
                output_window,
                mode_select_num=16, 
                kernel_size=25, 
                dropout=0.1, 
                pool="mean"
                ):
        super().__init__()
        self.pool = pool                                  # "mean" 또는 "last" 풀링 방식 저장
        self.output_window = output_window                      # 예측할 출력 길이
        # 1) 입력 투영: (.., input_dim) → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2) 시계열 분해 모듈(이동평균 기반): (B, L, d_model) → seasonal, trend
        self.decomp = SeriesDecomposition(kernel_size)

        # 3) Fourier 블록: seasonal에 대해 상위 모드만 남기고 시간영역 복원 → (B, L, d_model)
        self.fourier = FourierBlock(mode_select_num)

        # 4) 안정화용 정규화/드롭아웃(선택)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # 5) MIMO 예측 헤드: d_model → output_window
        self.decoder = nn.Linear(d_model, output_window)

    def forward(self, x):
        # x: (B, L, input_dim)
        h = self.input_proj(x)                     # (B,L,D)
        seasonal, trend = self.decomp(h)           # (B,L,D), (B,L,D)

        seasonal_f = self.fourier(seasonal)        # (B,L,D)  상위 모드만 반영
        h = seasonal_f + trend                     # 베이스라인 복원 (seasonal을 FourierBlock에 넣은 뒤 trend를 다시 더하고 풀링함)
        h = self.norm(h)
        h = self.drop(h)

        rep = h.mean(dim=1) if self.pool == "mean" else h[:, -1, :]  # (B,D)
        out = self.decoder(rep).unsqueeze(-1)       # (B,H,1)
        return out