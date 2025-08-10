import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val, 
    X_test, 
    y_test,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    num_epochs=30
    ):

    # 손실/옵티마이저/스케줄러
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Dataloader 설정
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, drop_last=False
    )

    # 학습루프 
    best_val = float("inf")
    best_path = f"results/{model.__class__.__name__}-BDG2.pt"


    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(model.device) if hasattr(model, "cuba") else xb
            yb = yb.to(model.device) if hasattr(model, "cuba") else yb

            optimizer.zero_grad()
            output = model(xb)                  # (B, output_window, 1)
            loss = criterion(output, yb)        # MAE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 그래디언트 클립: 역전파 기울기 폭주 방지
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        # ---- valid ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(model.device) if hasattr(model, "cuba") else xb
                yb = yb.to(model.device) if hasattr(model, "cuba") else yb
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item()

        val_loss = val_loss_sum / len(val_loader)

        # 경로 생성
        os.makedirs(os.path.dirname(best_path), exist_ok=True)

        # 체크포인트 저장
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict()}, best_path)
            print(f"[최적 {model.__class__.__name__}모델 저장 완료] {best_path}")

        print(f"Epoch {epoch:03d} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} (best {best_val:.4f})")

        