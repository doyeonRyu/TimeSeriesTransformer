import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 시계열 분해 모듈 (Trend + Seasonal)
class SeriesDecomposition(nn.Module):
    """
    Class: SeriesDecomposition
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

class AutoCorrelation(nn.Module):
    """
    Class: AutoCorrelation
        - 시계열의 지연 구조를 주파수 영역(FFT)에서 빠르게 찾아내고, 
          가장 중요한 지연들(top-k)을 골라 values를 해당 지연만큼 순환이동 (roll)하여 
          가중합으로 출력 시퀀스를 재구성하는 모듈
        - Transformer의 self-attention 대신, 
          Q와 K의 순환 상관을 써서 주기적 패턴(계절성)의 지연을 직접적으로 선택
    Parameters:
        - top_k(int): 선택할 상위 지연 개수 (기본값: 5)
    forward parameters:
        - queries (torch.Tensor): shape = [B, L, D]. 쿼리 텐서
        - keys (torch.Tensor): shape = [B, L, D]. 키 텐서
        - values (torch.Tensor): shape = [B, L, D]. 값 텐서
    forward returns:
        - out (torch.Tensor): shape = [B, L, D]. 
            선택된 지연들로 순환이동한 values의 가중합 결과
    """
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k  # 선택할 상위 지연 개수 저장 

    @staticmethod # 정적 메서드로 의
    def _batch_roll(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Function: _batch_roll
            - 배치별로 서로 다른 쉬프트 값을 사용해 시간축을 순환 이동(roll)하는 함수
        Parameters:
            - x (torch.Tensor): 입력 텐서, shape = [B, L, D]
                - B: 배치 크기, L: 시퀀스 길이, D: 피처 차원
            - shift (torch.Tensor): 각 배치별로 적용할 쉬프트 값, shape = [B,]
                - 각 배치마다 다른 지연을 적용할 수 있음
        Returns:
            - rolled (torch.Tensor): 순환 이동된 텐서, shape =[B, L, D]
        """
        # x: (B, L, D), shift: (B,) 각 배치마다 다른 쉬프트를 한 번에 적용
        B, L, D = x.shape # 배치, 길이, 차원 추출
        idx_base = torch.arange(L, device=x.device).unsqueeze(0) # (1, L) 기준 인덱스
        # 음수 방향으로 이동하려면 (t - s) 모듈로 L 사용
        gather_idx = (idx_base - shift.unsqueeze(1)) % L # (B, L) 각 배치별 인덱스
        gather_idx = gather_idx.unsqueeze(-1).expand(B, L, D) # (B, L, D) 차원 맞춤
        return torch.gather(x, dim=1, index=gather_idx) # (B, L, D) 배치별 롤링 결과

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Function: forward
            - Q, K의 순한 상관을 FFT로 계산하여 지연 점수를 얻고, 상위 top-k 지연만큼 V를 배치별로 롤링 후 
              softmax 가중합으로 출력 시퀀스 생성
        Parameters:
            - queries (torch.Tensor): shape = [B, L, D]. 쿼리 텐서
            - keys (torch.Tensor): shape = [B, L, D]. 키 텐서
            - values (torch.Tensor): shape = [B, L, D]. 값 텐서
        Returns:
            - out (torch.Tensor): shape = [B, L, D]. 
                선택된 지연들로 순환이동한 values의 가중합 결과
        """
        # 입력: (B, L, D) 규약
        B, L, D = queries.shape # 배치, 길이, 차원 읽기

        # 1) 주파수 영역에서 순환 상관 계산
        #   rFFT: 실수 입력을 절반 스펙트럼(Lf=L//2+1)으로 변환하여 효율적 계산
        Q = torch.fft.rfft(queries, n=L, dim=1) # (B, Lf, D) Q의 rFFT
        K = torch.fft.rfft(keys,    n=L, dim=1) # (B, Lf, D) K의 rFFT
        # 교차 스펙트럼 R = Q * conj(K) = 주파수 영역 곱 → 시간 영역 역변환하면 순환 상관
        R = Q * torch.conj(K) # (B, Lf, D)
        corr = torch.fft.irfft(R, n=L, dim=1) # (B, L, D) 지연별 상관 점수(평균)

        # 2) 채널 평균으로 지연 중요도 산출(다른 집계도 가능: max, L2 등)
        scores = corr.mean(dim=-1) # (B, L)

        # 3) 상위 top_k 지연 선택
        k = min(self.top_k, L) # 선택할 지연 개수 제한 (길이 한계 내에서)
        topw, topi = torch.topk(scores, k=k, dim=-1) # topw: (B, k) 점수, topi: (B, k) 지연 인덱스 

        # 4) 안정적 가중치: softmax로 정규화
        weights = torch.softmax(topw, dim=-1) # (B, k) 지연 가중치 

        # 5) 선택된 각 지연에 대해 values를 배치별로 순환 이동 후 가중합
        out = torch.zeros_like(values) # (B, L, D) 누적 버퍼 초기화
        for i in range(k): # k개 지연 반복(벡터화 + 짧은 루프)
            shift_i = topi[:, i] # (B,) 배치별 i번째 지연 
            rolled = self._batch_roll(values, shift=shift_i) # (B, L, D) 배치별 롤. 해당 지연만큼 순환 이동
            w_i = weights[:, i].view(B, 1, 1) # (B,1,1) 브로드캐스트용 가중치 
            out = out + rolled * w_i # (B, L, D) 누적

        return out # (B, L, D)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, moving_avg_kernel, dropout=0.1):
        """
        Class: EncoderLayer
            - 입력 시계열 x를 SeriesDecompotition을 seasonal과 trend로 분해
            - seasonal에 AutoCorrelation을 적용하고 잔차 연결 + 정규화
            - 다시 한 번 분해후 위치별 FFN을 적용, 잔차 연결 + 정규화
            - 레이어 내에서 생성된 trend들을 누적 합산하여 밖으로 전달하고, seasonal은 다음 레이어로 전달
        Parameters:
            - d_model (int): 임베딩 차원 수
            - moving_avg_kernel (int): 이동 평균 커널 크기 (trend 추출용)
            - dropout (float): 드롭아웃 비율 (기본값: 0.1)
        forward parameters:
            - x (torch.Tensor): 입력 텐서, shape = [B, L, D]
                - B: 배치 크기, L: 시퀀스 길이, D: 피처 차원
        forward returns:
            - seasonal_out (torch.Tensor): shape = [B, L, D]
                - 계절 성분(다음 레이어 입력용)
            - trend_comp (torch.Tensor): shape = [B, L, D]
                - 이 레이어까지 누적된 추세 성분(복원용)
        """
        super(EncoderLayer, self).__init__()
        # 1) 1차 시계열 분해: x -> (seasonal1, trend1)
        self.decomp1 = SeriesDecomposition(moving_avg_kernel) 

        # 2) AutoCorrelation: self-attention 대체(지연 선택 - 가중합)
        self.auto_corr = AutoCorrelation()

        # 공통 dropout / 정규화 
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model) # AutoCorrelation 블록 뒤 정규화 

        # 3) 위치별 FFN (d_model -> d_model로 단순화; dim_feedforward로 확장 가능)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), # 선형 변환 1 (특징 혼합)
            nn.GELU(), # 비선형 활성화 (transformer에서는 relu 사용)
            nn.Dropout(dropout), # 과적화 방지 드롭아웃
            nn.Linear(d_model, d_model) # 선형 변환 2 (차원 복원)
        )

        # 4) 2차 시계열 분해: AutoCorrelation + 정규화 이후의 신호를 다시 분해 
        self.decomp2 = SeriesDecomposition(moving_avg_kernel)
        self.layer_norm2 = nn.LayerNorm(d_model)# FFN 블록 뒤 정규화

    def forward(self, x):
        # x: (B, L, D) 입력
        # 블록 A: Auto Correlation으로 의존성 집계
        
        # 1) 1차 분해 -> seasonal1, trend1 (둘 다 [B, L, D])
        seasonal1, trend1 = self.decomp1(x) 
        # 2) seasonal1에 AutoCorrelation 적용(Q=K=V=seasonal1)
        ac_out = self.auto_corr(seasonal1, seasonal1, seasonal1)
        # 3) 잔차 연결(residual): seasonal1 + AutoCorrelation 결과
        x = seasonal1 + self.dropout(ac_out) 
        # 4) 정규화로 안정화 (shape 유지 [B, L, D])
        x = self.layer_norm1(x)

        # 블록 B: FFN으로 위치별 비선형 변환

        # 5) 2차 분해 -> seasonal2, trend2
        seasonal2, trend2 = self.decomp2(x)
        # 6) FFN 통과 (위치별 MLP)
        ffn_out = self.ffn(seasonal2)
        # 7) 잔차 연결: seasonal2 + FFN 결과
        seasonal_out = seasonal2 + self.dropout(ffn_out)
        # 8) 정규화 (shape 유지 [B, L, D])
        seasonal_out = self.layer_norm2(seasonal_out)

        # 트렌드 누적 

        # 9) 레이어 내 생성된 trend 합산 (누적 개념)
        trend_comp = trend1 + trend2 
        # 10) (다음 레이어 입력용 seasonal_out, 밖으로 전달할 trend_comp 반환)
        return seasonal_out, trend_comp  
    
class Encoder(nn.Module):
    """
    Class: Encoder
        - 여러 개의 EncoderLayer를 쌓아 계절성(seasonal)을 점진적으로 변환하고,
            각 레이어에서 추출된 추세(trend)를 누적 합산으로 모음
        - 최종적으로 (최종 seasonal_out, 누적 trend_sum) 반환
    Parameters:
        - d_model (int): 임베딩 차원 수
        - moving_avg_kernel (int): 이동 평균 커널 크기 (trend 추출용)
        - num_layers (int): EncoderLayer의 개수
        - dropout (float): 드롭아웃 비율 (기본값: 0.1)
    forward parameters:
        - x (torch.Tensor): 입력 텐서, shape = [B, L, D]
            - B: 배치 크기, L: 시퀀스 길이, D: 피처 차원
    forward returns:
        - x (torch.Tensor): shape = [B, L, D]
            - 최종 계절 성분(마지막 레이어 출력)
        - trend_sum (torch.Tensor): shape = [B, L, D]
            - 모든 레이어에서 누적된 추세 성분
    """
    def __init__(self, d_model, moving_avg_kernel, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        # 여러 개의 EncoderLayer 스택
        # 각 EncoderLayer는 (seasonal_out, trend_comp) 쌍을 반환
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, moving_avg_kernel, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (B, L, D) 입력 텐서
        # B, L, D = x.shape # 배치 크기, 시퀀스 길이, 차원
        trend_list = []  # 각 레이어에서 추출된 trend를 모아둘 리스트 

        # 각 EncoderLayer를 순차적으로 통과
        for layer in self.layers:
            x, trend = layer(x) # x: (B, L, D) seasonal 갱신, trend: (B, L, D)
            trend_list.append(trend) # trend 추가

        # 각 레이어별 trend를 모두 합산 -> 누적 추세
        # trend_list: 길이 num_layers, 각 원소 (B, L, D) 형태
        trend_sum = torch.stack(trend_list, dim=0).sum(dim=0) # (num_layers, B, L, D) -> sum -> (B, L, D)

        # 반환:
        # x: 마지막 레이어의 seasonal_out 출력 (다음 모듈로 전달)
        # trend_sum: 모든 레이어에서 누적된 추세 성분 (나중에 seasonal과 결합)
        return x, trend_sum

class DecoderLayer(nn.Module):
    """
    Class: DecoderLayer
        - 디코더 내부에서 입력 시계열을 이동평균 기반 분해하고
        - AutoCorrelation을 통해 패턴을 추출하여 피처 갱신
    Parameters:
        - d_model (int): 임베딩 차원 수
        - moving_avg_kernel (int): 이동 평균 커널 크기 (trend 추출용)
        - dropout (float): 드롭아웃 비율 (기본값: 0.1)
    forward parameters:
        - x (torch.Tensor): 입력 텐서, shape = [B, L_dec, D]
            - B: 배치 크기, L_dec: 시퀀스 길이, D: 피처 차원
        - cross (torch.Tensor): 인코더 출력, shape = [B, L_enc, D]
            - L_enc: 인코더 시퀀스 길이
    forward returns:
        - seasonal_out (torch.Tensor): shape = [B, L_dec, D]
            - 계절 성분(다음 레이어 입력용)
        - trend_out (torch.Tensor): shape = [B, L_dec, D]
            - 이 레이어까지 누적된 추세 성분(복원용)
    """
    def __init__(self, d_model, moving_avg_kernel, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 1) 디코더 내부용 분해 + 자기상관
        #   자기상관을 통해 시계열의 계절성 패턴을 추출
        self.self_decomp = SeriesDecomposition(moving_avg_kernel) 
        self.self_auto_corr = AutoCorrelation()
        self.layer_norm1 = nn.LayerNorm(d_model) # 잔차 연결 후 안정화

        # 2) 인코더-디코더 교차 자기상관
        #   cross_auto_corr: 디코더 쿼리 vs 인코더 키/값으로 상관
        self.cross_auto_corr = AutoCorrelation()
        self.layer_norm2 = nn.LayerNorm(d_model) 

        # 3) 위치별 FFN
        #   각 타입스탭/토큰을 독립적으로 변환
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), # 1차 선형 변환
            nn.GELU(), # 비선형 활성화
            nn.Dropout(dropout), # 과적화 방지 드롭아웃
            nn.Linear(d_model, d_model) # 2차 선형 변환(차원 복귀)
        )

        # 추가 분해 모듈
        #   FFN 이후 혹은 블록 마지막에 다시 분해하여 trend 성분을 추출/누적 시 사용
        self.decomp = SeriesDecomposition(moving_avg_kernel)
        
        # 최종 정규화 및 공통 드롭아웃
        self.layer_norm3 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, cross):
        # x: (B, L_dec, D), cross: (B, L_enc, D)
        B, L_dec, D = x.size() # 배치, 디코더 길이, 차원 추출

        # 1) 디코더 자기 경로: 분해 → 자기상관 → 잔차 + 정규화
        seasonal_1, _ = self.self_decomp(x) # 계절성만 자기상관 대상으로 사용 
        x_ac = self.self_auto_corr(seasonal_1, seasonal_1, seasonal_1) # Q=K=V=seasonal_1
        x = self.layer_norm1(seasonal_1 + self.dropout(x_ac)) # 잔차 연결 후 정규화

        # 2) 교차 경로 길이 정렬: cross를 디코더 길이에 맞춘다
        if cross.size(1) != L_dec: # 길이가 다르면 
            if cross.size(1) > L_dec: # 인코더 길이가 더 길면
                cross = cross[:, :L_dec, :] # 앞에서 잘라서 맞추기
            else:
                pad_len = L_dec - cross.size(1) # 모자란 길이 계산
                pad = x.new_zeros(B, pad_len, D) # 제로 패딩 생성
                cross = torch.cat([cross, pad], dim=1) # 뒤에 제로 패딩 추가

        # 3) 교차 자기상관: x를 쿼리, cross를 키/값으로 → 잔차 + 정규화
        x_ac = self.cross_auto_corr(x, cross, cross) # Q=x, K=V=cross
        x = self.layer_norm2(x + self.dropout(x_ac)) # 잔차 연결 후 정규화

        # 4) FFN 경로: 위치별 변환 → 잔차 + 정규화
        y = self.ffn(x) # 각 위치 독립 변환
        y = self.layer_norm3(x + self.dropout(y)) # 잔차 연결 후 정규화

        # 5) 최종 분해: 다음 레이어용 seasonal과 trend 분리
        seasonal_2, trend_2 = self.decomp(y) # (계절성, 추세) 재분해

        return seasonal_2, trend_2

class Decoder(nn.Module):
    """
    Class: Decoder
        - 여러 개의 DecoderLayer를 순차로 통과하면서 디코더의 계절성분을 갱신하고,
            각 레이어에서 추출된 추세(trend)를 누적 합산하여 최종 출력
        - 마지막에 seasonal + trend를 더한 은닉표현을 값 공간으로 투영하여 최종 출력 생성
    Parameters:
        - d_model (int): 임베딩 차원 수
        - moving_avg_kernel (int): 이동 평균 커널 크기 (trend 추출용)
        - num_layers (int): DecoderLayer의 개수
        - dropout (float): 드롭아웃 비율 (기본값: 0.1)
        - input_dim (int): 입력 피처 차원 (디코더 입력용)
        - output_dim (int): 최종 출력 차원 (디코더 출력용, 기본값: 1)
    forward parameters:
        - seasonal_init (torch.Tensor): shape = [B, L_dec, D]
            - 디코더 입력용 초기 계절 성분
        - trend_init (torch.Tensor): shape = [B, L_dec, D]
            - 디코더 입력용 초기 추세 성분
        - cross (torch.Tensor): shape = [B, L_enc, D]
            - 인코더 출력(교차 경로)
    forward returns:
        - out (torch.Tensor): shape = [B, L_dec, output_dim]
    """
    def __init__(self,
                d_model, # d_model 차원
                moving_avg_kernel, # 이동 평균 커널 크기
                num_layers=1, # 디코더 레이어 수
                dropout=0.1, # 드롭아웃 비율
                input_dim=4, # 입력 피처 차원
                output_dim=1 # 최종 출력 차원
        ):
        super(Decoder, self).__init__()
        # 디코더 레이어 스택: DecoderLayer를 num_layers만큼 쌓음
        # 각 레이어는 (seasonal_out, trend_out) 쌍을 반환
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, moving_avg_kernel, dropout)
            for _ in range(num_layers)
        ])
        # 최종 투영: d_model -> output_dim
        self.projection = nn.Linear(d_model, output_dim)
        self.inpu_dim=input_dim

    def forward(self, seasonal_init, trend_init, cross):
        # seasonal_init: (B, L_dec, D)
        # trend_init: (B, L_dec, D)
        # cross: (B, L_enc, D)

        # 디코더 내부 누적 추세와 계절 시작점 설정
        trend = trend_init # 누적 추세 trend_init에서 시작
        x = seasonal_init # 계절 성분 seasonal_init에서 시작

        # 디코더 레이어를 순차 통과
        for layer in self.layers:
            # 각 레이어에서 갱신된 seasonal_out과 trend_out을 반환
            x, trend_layer = layer(x, cross)  # x: (B, L_dec, D), trend_layer: (B, L_dec, D)
            # 레이어별 추세 성분 누적
            trend = trend + trend_layer # 추세 누적 합산

        # 최종 은닉 표현: 계절 성분 + 누적 추세
        out_hidden = x + trend # (B, L_dec, D)
        # 값 공간으로 투영: d_model -> output_dim
        out = self.projection(out_hidden) # (B, L_dec, output_dim)
        return out # 최종 예측 반환

class Autoformer(nn.Module):
    """
    Class: Autoformer
        - Autoformer 모델 구현
        - 입력을 d_model 차원으로 임베딩하고,
        - 인코더가 계절성을 정제하고 추세 누적
        - 디코더가 예측 구간 길이 만큼 초기 seasonal/trend 시퀀스를 받아 인코더 출력과 교차 자기 상관으로 결합하여 
        - d_model 차원 히든 상태를 출력
        - d_model -> 1 최종 출력으로 변환
    Parameters:
        - input_len (int): 입력 시계열 길이
        - pred_len (int): 예측 시계열 길이
        - d_model (int): 임베딩 차원 수
        - moving_avg (int): 이동 평균 커널 크기 (trend 추출용)
        - enc_layers (int): 인코더 레이어 수
        - dec_layers (int): 디코더 레이어 수
        - dropout (float): 드롭아웃 비율 (기본값: 0.1)
        - input_dim (int): 입력 피처 차원 (기본값: 4)
    forward parameters:
        - x_enc (torch.Tensor): shape = [B, input_len, input_dim]
            - 인코더 입력 시계열, B: 배치 크기, input_len: 입력 길이, input_dim: 피처 차원
        - x_mark_enc (torch.Tensor): shape = [B, input_len, 2]
            - 인코더 입력 시계열의 시간 마크 (선택적(포함 가능), 예: 시간 정보)
        - x_dec (torch.Tensor): shape = [B, pred_len, input_dim]
            - 디코더 입력 시계열, B: 배치 크기, pred_len: 예측 길이, input_dim: 피처 차원
        - x_mark_dec (torch.Tensor): shape = [B, pred_len, 2]
            - 디코더 입력 시계열의 시간 마크 (선택적(포함 가능), 예: 시간 정보)
    forward returns:
        - y_hat (torch.Tensor): shape = [B, pred_len, 1]
            - 최종 예측 결과, B: 배치 크기, pred_len: 예측 길이, 1: 단일 출력 차원
    """
    def __init__(self,
                input_len,
                pred_len, 
                d_model=64, 
                moving_avg=25,
                enc_layers=2, 
                dec_layers=1, 
                dropout=0.1, 
                input_dim=4
            ):
        super(Autoformer, self).__init__()

        # 하이퍼파라미터 저장
        self.input_len = input_len # 인코더 입력 길이
        self.pred_len = pred_len # 디코더 예측 길이
        self.d_model = d_model # 임베딩 차원
        self.input_dim = input_dim # 입력 피처 차원

        # 1) 입력 임베딩: input_dim → d_model
        self.enc_embedding = nn.Linear(input_dim, d_model) # 원시 입력을 모델 히든 차원으로 투영

        # 2) 최종 출력 사영: d_model → 1
        self.output_layer = nn.Linear(d_model, 1) # 디코더 히든을 스칼라 예측으로 변환

        # 3) 인코더/디코더 구성
        self.encoder = Encoder(
            d_model=d_model,
            moving_avg_kernel=moving_avg, # 이동 평균 커널 크기
            num_layers=enc_layers,
            dropout=dropout
        )
        # Decoder가 d_model을 그대로 출력하도록 output_dim=d_model로 설정
        self.decoder = Decoder(
            d_model=d_model,
            moving_avg_kernel=moving_avg,
            num_layers=dec_layers,
            dropout=dropout,
            output_dim=d_model # 디코더 출력 히든 차원 = d_model
        )

        # 4) 디코더 입력 초기화용 분해기(시작 seasonal/trend 준비)
        #  디코더 입력을 seasonal과 trend로 분해하여 초기값으로 사용
        self.decomp = SeriesDecomposition(kernel_size=moving_avg)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: (B, input_len, input_dim)
        B = x_enc.size(0) # 배치 크기

        # 1. Linear embedding (input_dim → d_model)
        x_enc = self.enc_embedding(x_enc) # (B, input_len, d_model)

        # 2. 인코더 입력 분해
        seasonal_init, trend_init = self.decomp(x_enc) # 각각 (B, input_len, d_model)

        # 3. 인코더 전파
        enc_out, trend_enc = self.encoder(x_enc)
        #   enc_out: seasonal 정제 결과, trend_enc: trend 누적 합산

        # 4. 디코더 입력 초기화
        #   seasonal_input: 제로 텐서를 임베딩해 디코더 시작 seasonal 입력 생성
        seasonal_input = torch.zeros(B, self.pred_len, self.input_dim, device=x_enc.device) # (B, pred_len, input_dim)
        seasonal_input = self.enc_embedding(seasonal_input) # (B, pred_len, d_model)

        #    trend_input: 인코더 입력 추세의 마지막 시점 값을 디코더 길이만큼 반복
        trend_input = trend_init[:, -1:, :].repeat(1, self.pred_len, 1) # (B, pred_len, d_model)

        # 5. 디코더 전파 (디코더는 d_model 차원 히든을 출력)
        dec_hidden = self.decoder(seasonal_input, trend_input, enc_out) # (B, pred_len, d_model)

        # 6. 최종 변환: d_model → 1
        y_hat = self.output_layer(dec_hidden) # (B, pred_len, 1) 최종 예측 결과
        return y_hat 
