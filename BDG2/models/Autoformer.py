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

# 시계열 분해 모듈 (Trend + Seasonal)
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

class AutoCorrelation(nn.Module):
    '''
    설명:
        - Autoformer의 핵심인 자기상관 기반 시간 지연 집계(TDA)를 구현한다.
        - queries, keys의 원소별 상관도를 주파수 영역(FFT)에서 계산한 뒤
          시간축으로 역변환하여 각 지연(shift)별 상관 점수를 얻는다.
        - 상위 top_k 지연만 선택하여 values를 해당 지연만큼 순환 이동(roll)시킨 후
          가중합하여 출력 컨텍스트를 만든다.

    입력값:
        - queries: 텐서, shape=(B, L, D)
        - keys:    텐서, shape=(B, L, D)
        - values:  텐서, shape=(B, L, D)

    출력값:
        - out: 텐서, shape=(B, L, D)
            top_k 지연을 가중합한 컨텍스트
    '''
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k  # 선택할 상위 지연 개수

    @staticmethod
    def _batch_roll(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), shift: (B,)  각 배치마다 다른 쉬프트를 한 번에 적용
        B, L, D = x.shape                                          # 배치, 길이, 차원 추출
        idx_base = torch.arange(L, device=x.device).unsqueeze(0)   # (1, L)  기준 인덱스
        # 음수 방향으로 이동하려면 (t - s) 모듈로 L 사용
        gather_idx = (idx_base - shift.unsqueeze(1)) % L           # (B, L)  각 배치별 인덱스
        gather_idx = gather_idx.unsqueeze(-1).expand(B, L, D)      # (B, L, D) 차원 맞춤
        return torch.gather(x, dim=1, index=gather_idx)            # (B, L, D)  배치별 롤링 결과

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # 입력: (B, L, D) 규약
        B, L, D = queries.shape                                     # 텐서 크기 읽기

        # 1) 주파수 영역에서 순환 상관(circular correlation) 계산
        Q = torch.fft.rfft(queries, n=L, dim=1)                     # (B, Lf, D)  rFFT (Lf = L//2+1)
        K = torch.fft.rfft(keys,    n=L, dim=1)                     # (B, Lf, D)
        # Q * conj(K) = 주파수 영역 곱 → 시간 영역 역변환하면 순환 상관
        R = Q * torch.conj(K)                                       # (B, Lf, D)
        corr = torch.fft.irfft(R, n=L, dim=1)                       # (B, L, D)  지연별 상관 점수

        # 2) 채널 평균으로 지연 중요도 산출(다른 집계도 가능: max, L2 등)
        scores = corr.mean(dim=-1)                                  # (B, L)

        # 3) 상위 top_k 지연 선택
        k = min(self.top_k, L)                                      # 선택할 지연 개수 제한
        topw, topi = torch.topk(scores, k=k, dim=-1)                # topw: (B, k), topi: (B, k)

        # 4) 안정적 가중치: softmax로 정규화
        weights = torch.softmax(topw, dim=-1)                       # (B, k)

        # 5) 선택된 각 지연에 대해 values를 배치별로 순환 이동 후 가중합
        out = torch.zeros_like(values)                              # (B, L, D) 누적 버퍼
        for i in range(k):                                          # k개 지연 반복(벡터화 + 짧은 루프)
            shift_i = topi[:, i]                                    # (B,)
            rolled = self._batch_roll(values, shift=shift_i)        # (B, L, D) 배치별 롤
            w_i = weights[:, i].view(B, 1, 1)                       # (B,1,1) 브로드캐스트용
            out = out + rolled * w_i                                # (B, L, D) 누적

        return out                                                  # (B, L, D)
    
class EncoderLayer(nn.Module):
    '''
    설명:
        - Autoformer 인코더 레이어.
        - 입력을 1차 분해(Seasonal/Trend) → Seasonal에 Auto-Correlation(TDA) 적용 → 잔차+정규화
          → 2차 분해 → Seasonal에 FFN 적용 → 잔차+정규화.
        - 레이어에서 추출된 trend 성분(trend1 + trend2)을 함께 반환해 상위 인코더에서 누적 합산한다.

    입력값:
        - x: 텐서, shape=(B, L, d_model)

    출력값:
        - seasonal_out: 텐서, shape=(B, L, d_model)
        - trend_comp:   텐서, shape=(B, L, d_model)  # 해당 레이어에서 추출된 trend 성분
    '''
    def __init__(self, d_model, moving_avg_kernel, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.decomp1 = SeriesDecomposition(moving_avg_kernel)   # 1차 분해 모듈
        self.auto_corr = AutoCorrelation()                      # 자기상관(TDA) 모듈
        self.dropout = nn.Dropout(dropout)                      # 드롭아웃
        self.layer_norm1 = nn.LayerNorm(d_model)                # 1번째 정규화

        self.ffn = nn.Sequential(                               # 위치별 FFN
            nn.Linear(d_model, d_model),                        # 선형 변환 1
            nn.GELU(),                                          # GELU 활성화
            nn.Dropout(dropout),                                # 드롭아웃
            nn.Linear(d_model, d_model)                         # 선형 변환 2
        )

        self.decomp2 = SeriesDecomposition(moving_avg_kernel)   # 2차 분해 모듈
        self.layer_norm2 = nn.LayerNorm(d_model)                # 2번째 정규화

    def forward(self, x):
        # x: (B, L, D) 입력
        seasonal1, trend1 = self.decomp1(x)                     # 1) 1차 분해 → seasonal1, trend1
        ac_out = self.auto_corr(seasonal1, seasonal1, seasonal1)# 2) seasonal1에 TDA 적용
        x = seasonal1 + self.dropout(ac_out)                    # 3) 잔차 연결
        x = self.layer_norm1(x)                                 # 4) 정규화

        seasonal2, trend2 = self.decomp2(x)                     # 5) 2차 분해 → seasonal2, trend2
        ffn_out = self.ffn(seasonal2)                           # 6) FFN 통과
        seasonal_out = seasonal2 + self.dropout(ffn_out)        # 7) 잔차 연결
        seasonal_out = self.layer_norm2(seasonal_out)           # 8) 정규화

        trend_comp = trend1 + trend2                            # 9) 레이어별 trend 성분 합산
        return seasonal_out, trend_comp                         # 10) seasonal과 trend 반환

class Encoder(nn.Module):
    '''
    설명:
        - Autoformer 인코더 컨테이너.
        - 여러 EncoderLayer를 스택하며, 각 레이어에서 추출된 trend 성분을 누적 합산한다.
        - 최종적으로 seasonal 출력과 누적된 trend 합을 함께 반환한다(디코더에서 사용 가능).

    입력값:
        - x: 텐서, shape=(B, L, d_model)

    출력값:
        - seasonal_out: 텐서, shape=(B, L, d_model)
        - trend_sum:    텐서, shape=(B, L, d_model)
    '''
    def __init__(self, d_model, moving_avg_kernel, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([                           # 레이어 스택 생성
            EncoderLayer(d_model, moving_avg_kernel, dropout)   # EncoderLayer 인스턴스
            for _ in range(num_layers)                          # num_layers 만큼 반복
        ])

    def forward(self, x):
        trend_list = []                                         # 1) 레이어별 trend 저장 리스트

        for layer in self.layers:                               # 2) 각 레이어 순차 적용
            x, trend = layer(x)                                 #    layer가 (seasonal, trend) 반환
            trend_list.append(trend)                            #    trend 성분 누적

        trend_sum = torch.stack(trend_list, dim=0).sum(dim=0)   # 3) trend들을 합산 (B, L, D)
        return x, trend_sum                                     # 4) 최종 seasonal, trend 합 반환

class DecoderLayer(nn.Module):
    '''
    설명:
        - Autoformer 디코더 레이어.
        - 디코더 자기상관(TDA) → 교차 자기상관(인코더-디코더) → FFN → 분해 순으로 처리.
        - 출력은 seasonal(다음 블록 입력)과 trend(누적/복원용)로 나눈다.

    입력값:
        - x: 텐서, shape = (B, L_dec, d_model)
             디코더 입력 시퀀스(임베딩/히든)
        - cross: 텐서, shape = (B, L_enc, d_model)
             인코더 출력 시퀀스(히든)

    출력값:
        - seasonal_2: 텐서, shape = (B, L_dec, d_model)
        - trend_2:    텐서, shape = (B, L_dec, d_model)
    '''
    def __init__(self, d_model, moving_avg_kernel, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 1) 디코더 내부용 분해 + 자기상관
        self.self_decomp = SeriesDecomposition(moving_avg_kernel)   # 자기 경로 분해
        self.self_auto_corr = AutoCorrelation()                     # 자기상관(TDA)
        self.layer_norm1 = nn.LayerNorm(d_model)                    # 잔차 후 정규화

        # 2) 인코더-디코더 교차 자기상관
        self.cross_auto_corr = AutoCorrelation()                    # 교차 TDA
        self.layer_norm2 = nn.LayerNorm(d_model)                    # 잔차 후 정규화

        # 3) 위치별 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),                            # 선형
            nn.GELU(),                                              # 활성화
            nn.Dropout(dropout),                                    # 드롭아웃
            nn.Linear(d_model, d_model)                             # 선형
        )
        self.decomp = SeriesDecomposition(moving_avg_kernel)        # FFN 이후 분해
        self.layer_norm3 = nn.LayerNorm(d_model)                    # 잔차 후 정규화
        self.dropout = nn.Dropout(dropout)                          # 공용 드롭아웃

    def forward(self, x, cross):
        # x: (B, L_dec, D), cross: (B, L_enc, D)
        B, L_dec, D = x.size()                                      # 배치/길이/차원 추출

        # 1) 디코더 자기 경로: 분해 → 자기상관 → 잔차 + 정규화
        seasonal_1, _ = self.self_decomp(x)                         # (B, L_dec, D) 분해
        x_ac = self.self_auto_corr(seasonal_1, seasonal_1, seasonal_1)  # TDA
        x = self.layer_norm1(seasonal_1 + self.dropout(x_ac))       # residual + LN

        # 2) 교차 경로 길이 정렬: cross를 디코더 길이에 맞춘다
        if cross.size(1) != L_dec:                                  # 길이 다르면
            if cross.size(1) > L_dec:                               # 더 길면 자르기
                cross = cross[:, :L_dec, :]
            else:                                                   # 더 짧으면 제로 패딩
                pad_len = L_dec - cross.size(1)
                pad = x.new_zeros(B, pad_len, D)                    # 디바이스/dtype 일치
                cross = torch.cat([cross, pad], dim=1)              # (B, L_dec, D)

        # 3) 교차 자기상관: x를 쿼리, cross를 키/값으로 → 잔차 + 정규화
        x_ac = self.cross_auto_corr(x, cross, cross)                # (B, L_dec, D)
        x = self.layer_norm2(x + self.dropout(x_ac))                # residual + LN

        # 4) FFN 경로: 위치별 변환 → 잔차 + 정규화
        y = self.ffn(x)                                             # (B, L_dec, D)
        y = self.layer_norm3(x + self.dropout(y))                   # residual + LN

        # 5) 최종 분해: 다음 레이어용 seasonal과 trend 분리
        seasonal_2, trend_2 = self.decomp(y)                        # (B, L_dec, D), (B, L_dec, D)

        return seasonal_2, trend_2

class Decoder(nn.Module):
    '''
    설명:
        - Autoformer 디코더 컨테이너.
        - 여러 DecoderLayer를 통과하며 seasonal을 갱신하고 trend를 누적 합산한다.
        - 마지막에 seasonal + trend를 합쳐 최종 출력 차원(output_dim)으로 사영한다.

    입력값:
        - seasonal_init: (B, L_dec, d_model)  디코더 시작용 계절 성분(시작 토큰/제로 등)
        - trend_init:    (B, L_dec, d_model)  인코더 등에서 전달된 초기 추세(또는 제로)
        - cross:         (B, L_enc, d_model)  인코더 출력(교차 상관에 사용)

    출력값:
        - out: (B, L_dec, output_dim)        최종 예측(단변량이면 output_dim=1)
    '''
    def __init__(self,
                d_model, 
                moving_avg_kernel, 
                num_layers=1, 
                dropout=0.1, 
                input_dim=4, 
                output_dim=1):
        super(Decoder, self).__init__()
        # 디코더 레이어 스택
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, moving_avg_kernel, dropout)
            for _ in range(num_layers)
        ])
        # 최종 투영: d_model -> output_dim
        self.projection = nn.Linear(d_model, output_dim)
        self.inpu_dim=input_dim

    def forward(self, seasonal_init, trend_init, cross):
        # seasonal_init: (B, L_dec, D), trend_init: (B, L_dec, D), cross: (B, L_enc, D)
        trend = trend_init                     # 추세 성분 누적 시작
        x = seasonal_init                      # 계절 성분 경로 시작

        # 디코더 레이어를 순차 통과
        for layer in self.layers:
            x, trend_layer = layer(x, cross)   # x: (B,L_dec,D), trend_layer: (B,L_dec,D)
            trend = trend + trend_layer        # 추세 성분 누적

        # 최종 출력: seasonal + trend를 합쳐 값 공간으로 사영
        out_hidden = x + trend                 # (B, L_dec, D)
        out = self.projection(out_hidden)      # (B, L_dec, output_dim)
        return out

class Autoformer(nn.Module):
    '''
    설명:
        - Autoformer 메인 모듈.
        - 인코더: 시계열을 분해(Seasonal/Trend)하며 Auto-Correlation 기반 인코딩 수행.
        - 디코더: 디코더 입력 seasonal/encoder trend 정보를 활용해 예측 시퀀스를 생성.
        - 최종 출력은 (B, pred_len, 1) 형태.

    입력값:
        - x_enc: 텐서, shape=(B, input_len, input_dim)
        - x_mark_enc: 사용 안 함(자리 유지용)
        - x_dec: 사용 안 함(자리 유지용)
        - x_mark_dec: 사용 안 함(자리 유지용)

    출력값:
        - y_hat: 텐서, shape=(B, pred_len, 1)
    '''
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
        self.input_len = input_len              # 인코더 입력 길이
        self.pred_len = pred_len                # 디코더/예측 길이
        self.d_model = d_model                  # 히든 차원
        self.input_dim = input_dim              # 입력 차원 (전력, 온도, 강수 등)

        # 1) 입력 임베딩: input_dim → d_model
        self.enc_embedding = nn.Linear(input_dim, d_model)    # (B,L,input_dim)->(B,L,d_model)

        # 2) 최종 출력 사영: d_model → 1
        self.output_layer = nn.Linear(d_model, 1)             # (B,L,d_model)->(B,L,1)

        # 3) 인코더/디코더 구성
        self.encoder = Encoder(
            d_model=d_model,
            moving_avg_kernel=moving_avg,
            num_layers=enc_layers,
            dropout=dropout
        )
        # Decoder가 d_model을 그대로 내도록 output_dim=d_model로 설정
        self.decoder = Decoder(
            d_model=d_model,
            moving_avg_kernel=moving_avg,
            num_layers=dec_layers,
            dropout=dropout,
            output_dim=d_model
        )

        # 4) 디코더 입력 초기화용 분해기(시작 seasonal/trend 준비)
        self.decomp = SeriesDecomposition(kernel_size=moving_avg)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: (B, input_len, input_dim)
        B = x_enc.size(0)                                   # 배치 크기

        # 1. Linear embedding (input_dim → d_model)
        x_enc = self.enc_embedding(x_enc)                   # (B, input_len, d_model)

        # 2. 인코더 입력 분해
        seasonal_init, trend_init = self.decomp(x_enc)      # 각각 (B, input_len, d_model)

        # 3. 인코더 전파
        enc_out, trend_enc = self.encoder(x_enc)            # enc_out: (B, input_len, d_model), trend_enc: (B, input_len, d_model)

        # 4. 디코더 입력 초기화
        seasonal_input = torch.zeros(B, self.pred_len, self.input_dim, device=x_enc.device)
        seasonal_input = self.enc_embedding(seasonal_input)  # → (B, pred_len, d_model)

        #    trend_input: 인코더 입력 추세의 마지막 시점 값을 디코더 길이만큼 반복
        trend_input = trend_init[:, -1:, :].repeat(1, self.pred_len, 1)           # (B, L_dec, d_model)

        # 5. 디코더 전파 (디코더는 d_model 차원 히든을 출력)
        dec_hidden = self.decoder(seasonal_input, trend_input, enc_out)           # (B, L_dec, d_model)

        # 6. 최종 사영: d_model → 1
        y_hat = self.output_layer(dec_hidden)                                     # (B, L_dec, 1)
        return y_hat
