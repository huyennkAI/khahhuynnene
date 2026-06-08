# Normalizing Flows

> Normalizing Flow biến một phân phối đơn giản (như Gauss) thành phân phối dữ liệu phức tạp qua một chuỗi **hàm khả nghịch**. Nhờ công thức đổi biến, nó cho likelihood **chính xác** mà vẫn lấy mẫu một bước.

## Trực giác ra đời

Autoregressive cho likelihood chính xác nhưng lấy mẫu chậm vì tuần tự. VAE lấy mẫu nhanh nhưng chỉ cho *chặn dưới* likelihood. Câu hỏi: **có cách nào vừa lấy mẫu nhanh, vừa tính likelihood chính xác?**

Ý tưởng then chốt: nếu phép biến đổi từ nhiễu sang dữ liệu là **khả nghịch** và ta biết nó "kéo giãn/nén" không gian bao nhiêu tại mỗi điểm, thì có thể truy ngược mật độ một cách chính xác bằng giải tích — công thức đổi biến.

## Khắc phục điều gì

- **So với VAE:** thay chặn dưới ELBO bằng likelihood **chính xác**, vì phép ánh xạ khả nghịch nên không cần biến ẩn xấp xỉ.
- **So với Autoregressive:** lấy mẫu chỉ một lần truyền thẳng, không tuần tự.
- Đổi lại, phải **hi sinh tính tự do của kiến trúc** để giữ tính khả nghịch.

## Lý thuyết: công thức đổi biến

Cho biến ẩn $z \sim p_z(z)$ (thường $\mathcal{N}(0, I)$) và một hàm khả nghịch khả vi $x = f(z)$. Khi đó mật độ của $x$ là:

$$p_x(x) = p_z\big(f^{-1}(x)\big) \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right|$$

Lấy log để dùng cho huấn luyện:

$$\log p_x(x) = \log p_z\big(f^{-1}(x)\big) + \log \left| \det J_{f^{-1}}(x) \right|$$

### Chứng minh (trường hợp một chiều)

Gọi $z = f^{-1}(x)$. Bảo toàn xác suất trên một khoảng vi phân: xác suất khối phải bằng nhau dù đổi biến,

$$p_x(x)\, |dx| = p_z(z)\, |dz|$$

Suy ra $p_x(x) = p_z(z) \left| \dfrac{dz}{dx} \right| = p_z\big(f^{-1}(x)\big) \left| \dfrac{d f^{-1}}{dx} \right|$.

Trong nhiều chiều, hệ số kéo giãn thể tích chính là **trị tuyệt đối của định thức Jacobian** $\left|\det J\right|$, vì định thức Jacobian đo tỉ lệ thay đổi thể tích của phép biến đổi. $\blacksquare$

## Hai ràng buộc thiết kế

Để khả thi, mỗi phép biến đổi $f$ phải đồng thời:

1. **Khả nghịch** — để cả huấn luyện (cần $f^{-1}$) và sinh mẫu (cần $f$) đều chạy được.
2. **Có định thức Jacobian dễ tính** — định thức tổng quát tốn $O(d^3)$, không khả thi ở số chiều lớn.

### Coupling layer (RealNVP)

Mẹo kinh điển: chia $x$ thành hai nửa $x_a, x_b$ và biến đổi affine một nửa theo nửa kia:

$$y_a = x_a, \qquad y_b = x_b \odot \exp\!\big(s(x_a)\big) + t(x_a)$$

- **Khả nghịch dễ dàng:** $x_b = (y_b - t(y_a)) \odot \exp(-s(y_a))$.
- **Jacobian tam giác:** định thức chỉ là tích các phần tử đường chéo,

$$\log \left|\det J\right| = \sum_j s(x_a)_j$$

Mạng $s, t$ có thể tùy ý phức tạp mà Jacobian vẫn rẻ.

## Xếp chồng nhiều phép biến đổi

Ghép $K$ phép khả nghịch $f = f_K \circ \dots \circ f_1$; log-det cộng dồn:

$$\log p_x(x) = \log p_z(z) + \sum_{k=1}^{K} \log \left| \det J_{f_k^{-1}} \right|$$

Càng nhiều tầng, "dòng chảy" (flow) càng nắn được phân phối phức tạp hơn — tên gọi *normalizing flow* nghĩa là dòng biến đổi đưa dữ liệu về phân phối chuẩn hóa.

## Đại diện tiêu biểu

- **NICE / RealNVP** — đặt nền với coupling layer.
- **Glow** — thêm tích chập khả nghịch $1\times 1$, sinh khuôn mặt chất lượng cao.

## Ưu điểm

- **Likelihood chính xác** và tối ưu trực tiếp (không qua chặn dưới).
- **Lấy mẫu nhanh** — một lần truyền thẳng.
- **Không gian ẩn diễn giải được**, dễ nội suy và thao tác thuộc tính.

## Nhược điểm

- **Ràng buộc khả nghịch** giới hạn khả năng biểu diễn so với mạng tự do.
- **Cùng số chiều ở mọi tầng** ($z$ và $x$ phải cùng kích thước) khiến mô hình **rất lớn và tốn bộ nhớ**.
- Chất lượng mẫu thường **kém GAN và Diffusion** ở cùng quy mô.

> Flow là lựa chọn đẹp về lý thuyết khi cần likelihood chính xác và lấy mẫu nhanh, nhưng cái giá khả nghịch khiến nó khó cạnh tranh về độ sắc nét. Các ý tưởng đổi biến của nó vẫn tái xuất trong nhiều mô hình lai hiện đại.
