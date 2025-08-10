import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    PatchTST 스타일 패치 임베딩
    - channel_independent=True: 각 변수(채널)를 독립적으로 패치하고 공유 백본에 넣는 CI 모드
      → proj: Linear(patch_size -> d_model)
      → 출력: (B*D, N, d_model)  # B: 배치, D: 채널 수(input_dim), N: 패치 개수
    - channel_independent=False: 채널을 패치 안에서 함께 펼쳐서 투영(간이형)
      → proj: Linear(patch_size*input_dim -> d_model)
      → 출력: (B, N, d_model)

    num_patches = 1 + (input_window - patch_size) // stride
    """
    def __init__(self, input_window, patch_size, input_dim, d_model,
                 stride=None, channel_independent=True, dropout=0.0):
        super().__init__()
        self.input_window = input_window
        self.patch_size = patch_size
        self.stride = stride or patch_size          # 기본은 비겹침
        self.channel_independent = channel_independent

        assert self.patch_size <= input_window, "patch_size가 input_window보다 클 수 없습니다."
        assert (input_window - patch_size) % self.stride == 0, "stride가 안 나눠떨어집니다."
        self.num_patches = 1 + (input_window - patch_size) // self.stride

        if channel_independent:
            # 각 채널(변수)별로 시간패치만 펼침
            self.proj = nn.Linear(patch_size, d_model)
        else:
            # 패치 내 시간×채널을 한꺼번에 펼침(간이형)
            self.proj = nn.Linear(patch_size * input_dim, d_model)

        # 패치 위치 임베딩(학습형)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, D)  # L=input_window, D=input_dim
        return:
          - CI=True  -> (B*D, N, d_model)
          - CI=False -> (B,   N, d_model)
        """
        B, L, D = x.shape
        assert L == self.input_window, "입력 길이가 input_window와 달라요."

        # (B, N, D, P) : N=num_patches, P=patch_size
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)

        if self.channel_independent:
            # (B, D, N, P) -> (B*D, N, P)
            patches = patches.permute(0, 2, 1, 3).contiguous().view(B*D, self.num_patches, self.patch_size)
            z = self.proj(patches) + self.pos_emb          # (B*D, N, d_model)
            z = self.dropout(z)
            return z
        else:
            # (B, N, D*P)
            patches = patches.permute(0, 1, 2, 3).contiguous().view(B, self.num_patches, D*self.patch_size)
            z = self.proj(patches) + self.pos_emb          # (B, N, d_model)
            z = self.dropout(z)
            return z

class PatchTST(nn.Module):
    def __init__(self, input_window, patch_size, input_dim, d_model, output_window,
                 num_layers=2, nhead=4, dropout=0.1,
                 channel_independent=False, target_channel_idx=0, stride=None):
        super().__init__()
        self.channel_independent = channel_independent
        self.target_channel_idx = target_channel_idx
        self.output_window = output_window
        self.input_dim = input_dim

        # CI/stride 지원하는 PatchEmbedding을 쓰는 경우
        self.embedding = PatchEmbedding(
            input_window=input_window,
            patch_size=patch_size,
            input_dim=input_dim,
            d_model=d_model,
            stride=stride or patch_size,
            channel_independent=channel_independent,
            dropout=dropout
        )

        # 패치 개수 (비겹침이면 input_window//patch_size)
        self.num_patches = self.embedding.num_patches

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_window)

    def forward(self, x):
        # x: (B, L, D_in)
        B, _, D_in = x.shape
        z = self.embedding(x)             # CI=False: (B,N,d) / CI=True: (B*D_in,N,d)
        z = z + self.pos_emb
        z = self.emb_norm(z)
        z = self.emb_drop(z)

        z = self.encoder(z)               # same shape

        rep = z.mean(dim=1)               # CI=False: (B,d) / CI=True: (B*D_in,d)
        out = self.decoder(rep).unsqueeze(-1)  # CI=False: (B,H,1) / CI=True: (B*D_in,H,1)

        if self.channel_independent:
            # (B*D_in,H,1) -> (B,D_in,H,1) -> 대상 채널만 선택 -> (B,H,1)
            out = out.view(B, D_in, self.output_window, 1)
            out = out[:, self.target_channel_idx, :, :]

        return out                         # (B,H,1)
