# 기본 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F

# Autoformer와 동일
class MovingAverage(nn.Module):
    """
    Class: MovingAverage
        - 시계열 길이를 보존하면서 이동 평균을 계산하는 모듈
        - Autoformer의 시계열 분해에서 trend(추세) 성분을 추출할 때 사용
        - 보통 trend = MovingAverage(x), seasonal = x - trend 형태로 사용
    Parameters:
        - kernel_size(int): 이동 평균 window(창) 길이. 클수록 추세가 부드러워짐.
    forward parameters:
        - x: 텐서, shape [B, L, D]
            - B: 배치 크기, L: 시간(시퀀스) 길이, D: 피처 차원
    forward returns:
        - torch.Tensor, shape [B, L, D]
            - 길이가 보존된 이동평균 결과(추세 근사)
    """
    def __init__(self, kernel_size: int):
        super(MovingAverage, self).__init__()
        # 커널 크기 저장 (이동 평균 창 길이)
        self.kernel_size = int(kernel_size) # 정수형 보장
        assert self.kernel_size >= 1, "kernel_size는 1 이상이어야 함."
        
        # 길이 보존 패딩 계산: 짝/홀 모두 L을 유지하도록 왼/오 패딩을 나눠서 부여 
        # e.g.) k=5면 pad_left=2, pad_right=2 / k=4면 pad_left=1, pad_right=2
        self.pad_left = (self.kernel_size - 1) // 2 # 왼쪽 패딩
        self.pad_right = self.kernel_size - 1 - self.pad_left # 오른쪽 패딩

    def forward(self, x):
        # x: (B, L, D) 배치, 길이, 차원 
        B, L, D = x.shape # 배치, 길이, 차원 분해

        # avg_pool1d: (B, C, L) 형태 입력 필요 -> 차원 바꿔줌
        # D: C 역할, 길이 축 L로 유지 # (B, D, L)
        x = x.permute(0, 2, 1)

        # 좌우 패딩을 부여해 길이를 보존.
        # mode='replicate': 가장자리 값을 복제해서 채우므로 경계 왜곡을 줄임
        # 결과 길이: L + pad_left + pad_right # (B, D, L + pad_left + pad_right)
        x_padded = F.pad(x, (self.pad_left, self.pad_right), mode='replicate')  

        # 평균 풀링으로 이동평균 수행
        # 커널 = kernel_size, stride=1 -> 길이 보존 
        # 출력 길이 공식: (L + pad_left + pad_right - kernel)1 + 1 = L
        avg = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1) 

        # 다시 (B, L, D)로 복원 
        avg = avg.permute(0, 2, 1).contiguous()
        return avg

# Autoformer와 동일
# 시계열 분해 모듈 (Trend + Seasonal)
class SeriesDecomposition(nn.Module):
    """
    Class: SeroesDecomposition
        - 시계열 데이터를 추세(trend)와 잔차/계절(seasonal)로 분해
        - trend: MovingAverage(x), seasonal: x - trend
    Parameters:
        - kernel_size(int): 이동 평균 커널 크기 (trend 추출용)
    forward parameters:
        - x: 텐서, shape [B, L, D]
            - B: 배치 크기, L: 시퀀스 길이, D: 피처 차원
    forward returns:
        - seasonal (torch.Tensor): shape [B, L, D], 빠른 변동/주기/노이즈 성분
        - trend (torch.Tensor): shape [B, L, D], 느린 추세 성분
    """
    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size) # 이동 평균 모듈(길이 보존)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D) 입력 텐서
        trend = self.moving_avg(x) # (B, L, D) 이동 평균으로 추세 추출
        seasonal = x - trend # (B, L, D) 원 신호에서 추세를 뺀 잔차/계절 성분
        return seasonal, trend # (B, L, D), (B, L, D)

class FourierBlock(nn.Module):
    """
    Class: FourierBlock
        - 입력 시계열을 실수 FFT(rFFT)로 변환하여 주파수 성분의 크기를 추정한 뒤,
        - 배치별로 에너지가 큰 상위 k개 주파수 모드만 남기고 나머지는 0으로 마스킹
        - 선택된 모드만을 가지고 역 FFT(irFFT)를 적용하여 시간 영역으로 복원
    Parameters:
        - mode_select_num (int): 각 배치에 대해 선택할 상위 주파수 모드의 개수
    forward parameters:
        - x (torch.Tensor): 입력 텐서, shape = (B, L, D)
            - B: 배치 크기, L: 시퀀스 길이, D: 피처 차원
    forward returns:
        - y (torch.Tensor): 복원된 텐서, shape = (B, L, D)
            - 입력 시계열의 상위 모드만 남긴 후 역 FFT로 복원된 결과

    """
    def __init__(self, mode_select_num: int):
        super().__init__()

        # 보존할 상위 주파수 모드 개수 저장
        self.mode_select_num = mode_select_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape # B: 배치 크기, L: 시퀀스 길이, D: 피처 차원 추출

        # 1) 시간 -> 주파수: 실수 FFT(rFFT) 수행 (양의 주파수만, 길이 L 유지)
        X = torch.fft.rfft(x, n=L, dim=1) # x: (B, L, D) -> X: (B, Lf, D), Lf = L//2 + 1

        # 2) 주파수 에너지 추정: 채널 평균으로 주파수별 크기 계산
        mag = X.abs().mean(dim=-1) # (B, Lf) 각 주파수의 평균 크기

        # 3) 상위 k개 주파수 인덱스 선택
        Lf = X.size(1) # rFFT 길이 (양의 주파수 개수)
        k = min(self.mode_select_num, Lf) # 실제 선택 개수 k를 Lf로 제한
        top_idx = torch.topk(mag, k=k, dim=1).indices # (B, k) 상위 k개 주파수 인덱스

        # 4) 배치별 마스크 생성(선택한 모드만 True)
        mask = torch.zeros(B, Lf, device=x.device, dtype=torch.bool) # 마스크 초기화
        mask.scatter_(1, top_idx, True) # 각 배치 행에 상위 k개의 위치를 True로 설정
        mask = mask.unsqueeze(-1).expand(-1, -1, D) # 채널 차원 D로 브로드캐스트 -> (B, Lf, D)

        # 5) 마스킹 적용: 선택 모드만 남기고 나머지는 0
        X_filtered = torch.where(mask, X, torch.zeros_like(X)) # (B, Lf, D) 선택된 모드만 남김

        # 6) 주파수 -> 시간: 역 rFFT로 복원 (길이 L 유지)
        y = torch.fft.irfft(X_filtered, n=L, dim=1) # y: (B, L, D) 복원된 시간 영역 시계열

        return y # 선택 모드 기반 복원 결과 반환

class FEDformer(nn.Module):
    """
    Class: FEDformer
        - 입력을 d_model로 임베딩하고 SeriesDecomposition을 통해 seasonal과 trend로 분해
        - seasonal에 대해 FourierBlock을 적용해 장주기, 주요 진동만 남김
        - trend와 다시 합친 후 정규화/드롭아우스올 안정화하고, 풀링으로 시퀀스 표현 요약
        - 최종 선형층이 d_model -> output_window 다중-스텝 예측 수행
        - 인코더, 디코더 스택과 AutoCorrelation 모듈을 사용하지 않은 경량화 버전
    Parameters:
        - input_dim (int): 입력 피처 차원
        - d_model (int): 임베딩 차원
        - output_window (int): 예측할 출력 시퀀스 길이
        - mode_select_num (int): FourierBlock에서 선택할 상위 주파수 모드 개수
        - kernel_size (int): 이동 평균 커널 크기 (trend 추출용)
        - dropout (float): 드롭아웃 비율
        - pool (str): 시퀀스 표현 요약 방식, "mean" 또는 "last" 중 선택
    forward parameters:
        - x (torch.Tensor): 입력 텐서, shape = (batch, seq_len, input_dim)
            - batch: 배치 크기, seq_len: 시퀀스 길이, input_dim: 입력 피처 차원
    forward returns:
        - out (torch.Tensor): 예측 결과, shape = (batch, output_window, 1)
            - 다중-스텝 예측 결과, 각 스텝별 단일 타깃 회귀
    """
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

        # 풀링 방식 설정
        self.pool = pool 
        # 예측할 스텝 수 저장
        self.output_window = output_window  

        # 1) 입력 투영: (.., input_dim) → d_model
        self.input_proj = nn.Linear(input_dim, d_model) # 원시 입력을 히든 차원으로 매핑

        # 2) 시계열 분해 모듈(이동평균 기반): (B, L, d_model) → seasonal, trend
        self.decomp = SeriesDecomposition(kernel_size)

        # 3) Fourier 블록: seasonal에 대해 상위 모드만 남기고 시간영역 복원 → (B, L, d_model)
        self.fourier = FourierBlock(mode_select_num) # rFFT -> 상위-k 모드 선택 -> irFFT

        # 4) 안정화용 정규화/드롭아웃
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # 5) MIMO 예측 헤드: d_model → output_window
        self.decoder = nn.Linear(d_model, output_window) # 다중 스텝 동시 예측

    def forward(self, x):
        # x: (B, L, input_dim)
        h = self.input_proj(x) # (B, L, d_model) 입력을 d_model 차원으로 임베딩
        seasonal, trend = self.decomp(h) # seasonal, trend = (B, L, d_model) 각각 분해

        seasonal_f = self.fourier(seasonal) # seasonal에 주파수 필터 적용 (B, L, d_model)
        h = seasonal_f + trend # 계절 + 추세 재결합 (B, L, d_model)
        h = self.norm(h) # 정규화으로 안정화
        h = self.drop(h) # 드롭아웃으로 일반화

        # 시퀀스 풀링: mean 또는 마지막 시점 선택
        rep = h.mean(dim=1) if self.pool == "mean" else h[:, -1, :]

        # (B, d_model) → (B, output_window, 1)
        out = self.decoder(rep).unsqueeze(-1)
        return out # 다중-스텝 예측 반환