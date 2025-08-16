import numpy as np
import pandas as pd
import torch 
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath, input_window, output_window):
    """
    Fuction: load_and_preprocess
        1. CSV 파일 불러오기 
        2. 정렬, 결측치 보정
        3. Train, Validation, Test 분할
        4. MinMax 정규화
        5. 슬라이딩 윈도우 생성
        6. Tensor로 변환
    Parameters:
        filepath (str): CSV 파일 경로
        input_window (int): 입력 윈도우 크기
        output_window (int): 출력 윈도우 크기
    Return values:
        X_train_tensor (torch.Tensor): 학습용 입력 데이터
        y_train_tensor (torch.Tensor): 학습용 출력 데이터
        X_val_tensor (torch.Tensor): 검증용 입력 데이터
        y_val_tensor (torch.Tensor): 검증용 출력 데이터
        X_test_tensor (torch.Tensor): 테스트용 입력 데이터
        y_test_tensor (torch.Tensor): 테스트용 출력 데이터
        x_scaler (MinMaxScaler): 입력 데이터 스케일러 (추후 변환용)
        y_scaler (MinMaxScaler): 출력 데이터 스케일러 (추후 변환용)
    """
    # 1. csv 파일 불러오기  
    ## filepath: f"C:/Users/ryudo/OneDrive - gachon.ac.kr/AiCE2/석사논문/Transformer/BDG2/data/toronto_data_{year}.csv"
    df = pd.read_csv(filepath)

    # 2. 정렬, 결측치 보정, MinMax 정규화
    ## timestamp 순으로 정렬
    df["timestamp"] = pd.to_datetime(df["timestamp"]) # 문자열 -> datetime 변환
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first") # 중복 제거 및 정렬 

    ## timestamp 열을 0부터 시작하는 정수로 변환
    ## 항상 1시간 간격으로 시간축 맞춤
    full_index = pd.date_range(start=df["timestamp"].min(),
                            end=df["timestamp"].max(),
                            freq="h")
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"

    # timestamp_idx 부여
    df["timestamp_idx"] = np.arange(len(df))

    cols = ["timestamp_idx", "electricity", "Mean Temp (°C)", "Total Rain (mm)"] # x
    target_col = "electricity" # y

    # 결측치 보정 (선형 보간 -> 앞/뒤 채우기)
    df[["Mean Temp (°C)", "Total Rain (mm)", "electricity"]] = (
        df[["Mean Temp (°C)", "Total Rain (mm)", "electricity"]]
        .interpolate(method="time", limit_direction="both") # 선형 보간
        .ffill() # 앞쪽 채우기
        .bfill() # 뒤쪽 채우기
    )

    # 3. Train, Validation, Test 분할
    L = len(df) # 전체 데이터 길이
    train_end = int(L * 0.8) # 80%: train
    val_end   = int(L * 0.9) # 10%: validation, 나머지 10%: test
``
    x_raw = df[cols].values.astype(np.float32)            # [timestamp_idx, elec, temp, rain]
    y_raw = df[[target_col]].values.astype(np.float32)    # [elec]

    x_train_raw, x_val_raw, x_test_raw = x_raw[:train_end], x_raw[train_end:val_end], x_raw[val_end:]
    y_train_raw, y_val_raw, y_test_raw = y_raw[:train_end], y_raw[train_end:val_end], y_raw[val_end:]

    # 4. MinMax 정규화: train에만 fit, valid/test는 transform
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = x_scaler.fit_transform(x_train_raw) # train 
    x_val   = x_scaler.transform(x_val_raw)
    x_test  = x_scaler.transform(x_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw) # train
    y_val   = y_scaler.transform(y_val_raw)
    y_test  = y_scaler.transform(y_test_raw)

    # 5. 슬라이딩 윈도우 생성 
    def sliding_windows(x_scaled, y_scaled, input_window, output_window):
        """
        Function: sliding_windows
            - 슬라이딩 윈도우 생성 과정
        Parameters:
            - x_scaled (np.ndarray): 이미 정규화된 입력 데이터
            - y_scaled (np.ndarray): 이미 정규화된 출력 데이터
            - input_window (int): 입력 윈도우 크기
            - output_window (int): 출력 윈도우 크기
        Return values:
            - X_list (np.ndarray): 입력 윈도우 데이터
            - Y_list (np.ndarray): 출력 윈도우 데이터
        """
        T = x_scaled.shape[0] # 전체 시간 스텝 수 (1시간 단위 1년치 데이터라면 8760)
        X_list, Y_list = [], []
        for i in range(0, T - input_window - output_window + 1):
            X_win = x_scaled[i : i + input_window, :]                               # (input_window, 4)
            Y_win = y_scaled[i + input_window : i + input_window + output_window, :] # (output_window, 1)
            X_list.append(X_win)
            Y_list.append(Y_win)
        if len(X_list) == 0:
            return (np.empty((0, input_window, x_scaled.shape[1]), dtype=np.float32),
                    np.empty((0, output_window, 1), dtype=np.float32))
        return np.stack(X_list, 0).astype(np.float32), np.stack(Y_list, 0).astype(np.float32)
        # X_list: (N, input_window, 4)
        # Y_list: (N, output_window, 1)
        # N: 윈도우 개수 (for문 수 만큼)

    # 슬라이딩 윈도우 적용
    X_tr_np, y_tr_np = sliding_windows(x_train, y_train, input_window, output_window)
    X_va_np, y_va_np = sliding_windows(x_val,   y_val,   input_window, output_window)
    X_te_np, y_te_np = sliding_windows(x_test,  y_test,  input_window, output_window)

    # 6. Tensor로 변환 (dtype 변환: numpy -> torch.from_numpy)
    X_train_tensor = torch.from_numpy(X_tr_np).to("cuda")  # (N_tr, 168, 4)
    y_train_tensor = torch.from_numpy(y_tr_np).to("cuda")  # (N_tr, 24, 1)

    X_val_tensor   = torch.from_numpy(X_va_np).to("cuda")
    y_val_tensor   = torch.from_numpy(y_va_np).to("cuda")

    X_test_tensor  = torch.from_numpy(X_te_np).to("cuda")
    y_test_tensor  = torch.from_numpy(y_te_np).to("cuda")

    print("Train X:", X_train_tensor.shape, "y:", y_train_tensor.shape)
    print("Valid  X:", X_val_tensor.shape,   "y:", y_val_tensor.shape)
    print("Test  X:", X_test_tensor.shape,   "y:", y_test_tensor.shape)

    return (
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor,
        x_scaler, y_scaler
    )

