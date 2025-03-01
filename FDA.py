
batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
C, H, W = batch["img"].shape[1], batch["img"].shape[2], batch["img"].shape[3]
learnable_h = H
learnable_w = np.floor(W / 2).astype(int) + 1
if not hasattr(self.model, 'freq_filter'):
    self.model.freq_filter = nn.Parameter(torch.ones(C, learnable_h, learnable_w)).to(self.device)
new_H = 2 ** np.ceil(np.log2(H)).astype(int)
new_W = 2 ** np.ceil(np.log2(W)).astype(int)
resized_img = nn.functional.interpolate(batch["img"], size=(new_H, new_W), mode="bilinear", align_corners=False)
feature_fft = torch.fft.rfftn(resized_img, dim=(-2, -1))
feature_fft = feature_fft + 1e-8  # 加上一个极小值以避免除零错误
feature_amp = torch.abs(feature_fft)  # 幅度谱
feature_pha = torch.angle(feature_fft)  # 相位谱
low_freq_ratio = 0.5
cutoff_h = int(new_H * low_freq_ratio)
cutoff_w = int((new_W // 2 + 1) * low_freq_ratio)
mask = torch.ones((new_H, new_W // 2 + 1), device=self.device)
mask[:cutoff_h, :cutoff_w] = 0
resized_filter = nn.functional.interpolate(
    self.model.freq_filter.unsqueeze(0),
    size=(new_H, new_W // 2 + 1),
    mode="bilinear",
    align_corners=False
).squeeze(0)
resized_filter = resized_filter.unsqueeze(0)
mask_expanded = mask.unsqueeze(0).unsqueeze(0)
feature_amp_filtered = feature_amp * (1 - mask_expanded) + (feature_amp * resized_filter) * mask_expanded
feature_fft_invariant = feature_amp_filtered * torch.exp(1j * feature_pha)
resized_img = torch.fft.irfftn(feature_fft_invariant, dim=(-2, -1)).real
batch["img"] = nn.functional.interpolate(resized_img, size=(H, W), mode="bilinear", align_corners=False)
batch["img"] /= 255