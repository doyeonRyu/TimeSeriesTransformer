import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

def plot_data_by_month(filepath, building_name, year, X_train, X_val):
    # 데이터 불러오기
    df = pd.read_csv(filepath)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.sort_values('Date/Time').reset_index(drop=True)

    # train/valid/test 경계 인덱스 계산
    train_end = X_train.shape[0]
    valid_end = train_end + X_val.shape[0]

    # 시각화
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 왼쪽 y축: electricity (파랑)
    elec_line, = ax1.plot(df['Date/Time'], df['electricity'],
                          color='tab:blue', label='Electricity', linewidth=0.8)
    ax1.set_ylabel('Electricity', color='tab:blue')

    # 오른쪽 y축: Temp (빨강) / Rain (노랑)
    ax2 = ax1.twinx()
    temp_line, = ax2.plot(df['Date/Time'], df['Mean Temp (°C)'],
                          color='tab:red', label='Mean Temp (°C)', linewidth=0.8)
    rain_line, = ax2.plot(df['Date/Time'], df['Total Rain (mm)'],
                          color='tab:orange', label='Total Rain (mm)', linewidth=0.8)
    ax2.set_ylabel('Temperature / Rain', color='tab:red')

    # 경계선 표시 (---)
    split_line1 = ax1.axvline(df['Date/Time'].iloc[train_end],
                              color='gray', linestyle='--', label='Train/Valid Split', alpha=0.8)
    split_line2 = ax1.axvline(df['Date/Time'].iloc[valid_end],
                              color='black', linestyle='--', label='Valid/Test Split', alpha=0.8)

    # x축 라벨은 month만 표시
    ax1.xaxis.set_major_locator(MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter('%m'))
    ax1.set_xlabel('Month')


    # 범례 생성: 모든 라인을 합쳐서 표시
    lines = [elec_line, temp_line, rain_line, split_line1, split_line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # 제목 표시
    title_str = f"{building_name} Electricity / Temperature / Rain Data in {year}"
    plt.title(title_str)
    ax1.grid(True, which='both', axis='y', alpha=0.8)     # Electricity 축만 grid 표시
    plt.tight_layout()
    plt.show()

    # 저장
    fig.savefig(f"plots/png/{building_name}_{year}_data_plot.png", dpi=300)
    print(f"[저장 완료] plots/png/{building_name}_{year}_data_plot.png")

def plot_forecast(
    model,
    preds, trues,
    input_window, output_window,
    sample_index=0,
    X_test=None, x_scaler=None
    ):
    # 1) 입력 전력 시퀀스 준비
    elec_feature_index = 1  # 전력 컬럼 인덱스 (0: timestamp, 1: electricity, 2: temp, 3: rain)
    x_in = X_test[sample_index].cpu().numpy() if hasattr(X_test, "cpu") else X_test[sample_index]
    x_in_orig_all = x_scaler.inverse_transform(x_in)       # (input_window, F)
    input_seq = x_in_orig_all[:, elec_feature_index]       # 전력 컬럼만 1D로

    # 2) 출력(타깃/예측) 시퀀스 준비
    pred_seq = preds[sample_index].reshape(-1)   # (output_window,)
    true_seq = trues[sample_index].reshape(-1)   # (output_window,)

    L_in, L_out = input_window, output_window
    t_in  = np.arange(L_in)
    t_out = np.arange(L_in, L_in + L_out)

    # 3) 플롯
    plt.figure(figsize=(12, 5))
    # 입력 구간(파랑)
    plt.plot(t_in, input_seq, label="Input Electricity", linewidth=1.2, color="tab:blue")
    # 출력 구간: 실제(검정) vs 예측(빨강 점선)
    plt.plot(t_out, true_seq, label="Target (True)", linewidth=1.2, color="black")
    plt.plot(t_out, pred_seq, label="Prediction", linewidth=1.6, color="tab:red")

    # 경계선
    plt.axvline(L_in - 0.5, color="gray", linestyle="--", alpha=0.8, label="Input/Output Split")

    plt.title(f"{model.__class__.__name__} — Sample {sample_index} (Input {L_in} → Output {L_out})")
    plt.xlabel("Time step (relative)")
    plt.ylabel("Electricity")
    plt.legend(loc="upper left")
    plt.grid(True, which="both", axis="y", alpha=0.7)
    plt.tight_layout()

    save_path = f"plots/png/{model.__class__.__name__}_sample_{sample_index}_forecast.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[{model.__class__.__name__} 모델 결과 저장 완료] {save_path}")
