from utils.metrics import mae, rmse, mape
import numpy as np
import torch

def test_model(
    model, 
    X_test, 
    y_test,
    X_scaler,
    y_scaler
    ):

    # 평가 모드
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(X_test)):
            # 입력 시퀀스
            input_seq = X_test[i].unsqueeze(0)  # (1, input_window, input_dim)
            target_seq = y_test[i].squeeze(0).cpu().numpy()  # (output_window, 1)

            # 모델 예측
            output = model(input_seq)  # (1, output_window, 1)
            output = output.squeeze(0).cpu().numpy()  # (output_window, 1)

            # 저장
            predictions.append(output)
            targets.append(target_seq)

    # numpy 변환
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 정규화 복원
    predictions_orig = y_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    targets_orig = y_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

    # 성능 지표 계산
    mae_score = mae(targets_orig, predictions_orig)
    rmse_score = rmse(targets_orig, predictions_orig)
    mape_score = mape(targets_orig, predictions_orig)

    return predictions_orig, targets_orig