# Mô hình khuếch tán (Diffusion)

> Mô hình khuếch tán học cách **khử nhiễu dần dần**: bắt đầu từ nhiễu ngẫu nhiên thuần túy và từng bước biến nó thành một mẫu dữ liệu sắc nét. Đây là nền tảng của Stable Diffusion, DALL·E và nhiều hệ sinh ảnh hiện đại.

## Hai quá trình

### Quá trình thuận (Forward) — thêm nhiễu

Dần dần phá hủy dữ liệu bằng cách cộng nhiễu Gauss qua $T$ bước, theo một lịch nhiễu $\beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left( x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \; \beta_t I \right)$$

Sau đủ nhiều bước, $x_T$ gần như là nhiễu chuẩn $\mathcal{N}(0, I)$. Một tính chất tiện lợi: có thể nhảy thẳng tới bước $t$ bất kỳ với $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

### Quá trình nghịch (Reverse) — khử nhiễu

Mô hình học cách đảo ngược, từng bước đưa nhiễu về dữ liệu:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left( x_{t-1}; \mu_\theta(x_t, t), \; \Sigma_\theta(x_t, t) \right)$$

## Mục tiêu huấn luyện

Thay vì dự đoán trực tiếp $x_{t-1}$, mô hình **dự đoán lượng nhiễu** $\epsilon$ đã thêm vào. Hàm mất mát đơn giản hóa thành:

$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ \big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2 \right]$$

Mạng $\epsilon_\theta$ (thường là kiến trúc **U-Net**) học đoán nhiễu tại mỗi mức $t$.

```python
t = torch.randint(0, T, (batch,))
eps = torch.randn_like(x0)
x_t = sqrt_acp[t] * x0 + sqrt_one_minus_acp[t] * eps
loss = mse(model(x_t, t), eps)
```

## Lấy mẫu

Bắt đầu từ $x_T \sim \mathcal{N}(0, I)$, lặp lại bước khử nhiễu từ $t = T$ về $1$ để thu được mẫu $x_0$. DDPM gốc cần hàng trăm tới hàng nghìn bước; các bộ giải như **DDIM** rút ngắn còn vài chục bước.

## Sinh có điều kiện

- **Classifier-free guidance** — huấn luyện đồng thời có và không điều kiện, rồi nội suy để điều khiển mức độ bám theo prompt.
- **Latent Diffusion** — chạy khuếch tán trong không gian ẩn nén (như Stable Diffusion) để giảm chi phí tính toán mạnh.

## Ưu và nhược

- **Ưu:** chất lượng và đa dạng mẫu rất cao, huấn luyện **ổn định** hơn GAN.
- **Nhược:** lấy mẫu **chậm** do cần nhiều bước; chi phí suy luận lớn.

> Diffusion hiện là họ mô hình dẫn đầu cho sinh ảnh, và đang lan sang âm thanh, video và cả sinh phân tử.
