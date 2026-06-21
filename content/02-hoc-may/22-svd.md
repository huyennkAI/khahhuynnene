# Phân rã giá trị suy biến (SVD)

> Phân rã giá trị suy biến (Singular Value Decomposition, SVD) là công cụ đại số tuyến tính quan trọng nhất trong học máy. Mọi ma trận thực $\mathbf{A} \in \mathbb{R}^{m \times n}$ — dù không vuông, không đầy hạng — đều có thể phân rã thành:
>
> $$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$
>
> trong đó $\mathbf{U}$, $\mathbf{V}$ là ma trận trực giao và $\boldsymbol{\Sigma}$ là ma trận đường chéo với các **giá trị suy biến (singular values)** không âm xếp giảm dần. SVD là nền tảng của [PCA](pca), giải bài toán bình phương tối thiểu, nén ảnh và hệ gợi ý.

---

# 1. Phát biểu và ký hiệu

**Định lý SVD.** Với mọi $\mathbf{A} \in \mathbb{R}^{m \times n}$, tồn tại phân rã:

$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

trong đó:

* $\mathbf{U} \in \mathbb{R}^{m \times m}$ — ma trận **trực giao (orthogonal)**: $\mathbf{U}^\top \mathbf{U} = \mathbf{I}_m$. Các cột $\mathbf{u}_k$ gọi là **vector suy biến trái (left singular vectors)**.
* $\mathbf{V} \in \mathbb{R}^{n \times n}$ — ma trận trực giao: $\mathbf{V}^\top \mathbf{V} = \mathbf{I}_n$. Các cột $\mathbf{v}_k$ gọi là **vector suy biến phải (right singular vectors)**.
* $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ — ma trận "đường chéo" (chỉ các phần tử $\Sigma_{kk} = \sigma_k$ khác $0$), với $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r \ge 0 = \cdots$ và $r = \text{rank}(\mathbf{A})$. Các $\sigma_k$ gọi là **giá trị suy biến (singular values)**.

**Dạng ngoại tích.** SVD cũng viết được như tổng các ma trận hạng một:

$$\mathbf{A} = \sum_{k=1}^{r} \sigma_k \mathbf{u}_k \mathbf{v}_k^\top$$

Đây là cách nhìn quan trọng nhất cho ứng dụng xấp xỉ hạng thấp.

---

# 2. Chứng minh SVD tồn tại

Ta chứng minh bằng quy nạp, dựa trên một lập luận hình học đẹp.

**Bước cơ sở.** Xét $\mathbf{A} \ne \mathbf{0}$. Hàm $f(\mathbf{v}) = \lVert \mathbf{A}\mathbf{v} \rVert$ liên tục trên hình cầu đơn vị compact $\{\lVert \mathbf{v} \rVert = 1\}$, nên đạt cực đại tại một điểm $\mathbf{v}_1$. Đặt $\sigma_1 = \lVert \mathbf{A}\mathbf{v}_1 \rVert > 0$ và $\mathbf{u}_1 = \mathbf{A}\mathbf{v}_1 / \sigma_1$.

Với $\mathbf{v} \perp \mathbf{v}_1$, ta chứng minh $\mathbf{A}\mathbf{v} \perp \mathbf{u}_1$. Xét $g(t) = \lVert \mathbf{A}(\mathbf{v}_1 \cos t + \mathbf{v} \sin t) \rVert^2$, đây là bình phương chuẩn của $\mathbf{A}$ tác động lên vector đơn vị. Vì $g(t)$ đạt cực đại tại $t = 0$, đạo hàm tại $t = 0$ bằng $0$: $g'(0) = 2\sigma_1 \langle \mathbf{u}_1, \mathbf{A}\mathbf{v} \rangle = 0$, nên $\mathbf{A}\mathbf{v} \perp \mathbf{u}_1$.

**Bước quy nạp.** Vậy $\mathbf{A}$ ánh xạ $\mathbf{v}_1^\perp$ vào $\mathbf{u}_1^\perp$. Xây hệ trực chuẩn cho $\mathbf{u}_1^\perp$ và $\mathbf{v}_1^\perp$, biểu diễn $\mathbf{A}$ trong hệ đó cho ma trận nhỏ hơn $(m-1)\times(n-1)$. Áp dụng quy nạp để thu được toàn bộ SVD. $\blacksquare$

---

# 3. Quan hệ với trị riêng của $\mathbf{A}^\top \mathbf{A}$ và $\mathbf{A}\mathbf{A}^\top$

Đây là cầu nối quan trọng nối SVD với trị riêng:

$$\mathbf{A}^\top \mathbf{A} = (\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top)^\top(\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top) = \mathbf{V}\boldsymbol{\Sigma}^\top\mathbf{U}^\top\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \mathbf{V}(\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma})\mathbf{V}^\top$$

Vì $\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1^2, \dots, \sigma_r^2, 0, \dots)$, đây là **phân tích phổ** của $\mathbf{A}^\top\mathbf{A}$: các **trị riêng** của $\mathbf{A}^\top\mathbf{A}$ là $\sigma_k^2$, và **vector riêng** là các cột của $\mathbf{V}$.

Tương tự $\mathbf{A}\mathbf{A}^\top = \mathbf{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top)\mathbf{U}^\top$, với cột của $\mathbf{U}$ là vector riêng.

> **Hệ quả quan trọng.** SVD của $\mathbf{X}$ (ma trận dữ liệu, $N \times d$) cho **thành phần chính** của [PCA](pca) ngay lập tức: vector riêng của $\mathbf{X}^\top\mathbf{X} = (N-1)\mathbf{S}$ chính là các cột của $\mathbf{V}$, và $\lambda_k = \sigma_k^2 / (N-1)$. Đây là lý do tất cả thư viện PCA đều dùng SVD chứ không dùng tính trị riêng trực tiếp.

---

# 4. Xấp xỉ hạng thấp: Định lý Eckart–Young

Đây là kết quả đẹp nhất và ứng dụng nhiều nhất của SVD.

**Định lý Eckart–Young–Mirsky.** Trong số tất cả các ma trận $\hat{\mathbf{A}}$ có $\text{rank}(\hat{\mathbf{A}}) \le k$, xấp xỉ hạng $k$ tốt nhất của $\mathbf{A}$ (theo chuẩn Frobenius và chuẩn phổ) là:

$$\mathbf{A}_k = \sum_{j=1}^{k} \sigma_j \mathbf{u}_j \mathbf{v}_j^\top = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$$

trong đó $\mathbf{U}_k, \mathbf{V}_k$ là $k$ cột đầu của $\mathbf{U}, \mathbf{V}$ và $\boldsymbol{\Sigma}_k$ là $k \times k$ ma trận đường chéo đầu tiên.

**Sai số xấp xỉ Frobenius:**

$$\lVert \mathbf{A} - \mathbf{A}_k \rVert_F^2 = \sum_{j=k+1}^{r} \sigma_j^2$$

**Phác chứng minh** (phần quan trọng). Ta cần chỉ ra không có ma trận hạng $\le k$ nào có sai số Frobenius nhỏ hơn. Giả sử $\hat{\mathbf{A}}$ hạng $\le k$, thì $\ker(\hat{\mathbf{A}})$ có số chiều $\ge n - k$. Không gian $\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_{k+1})$ có chiều $k+1$. Hai không gian này giao nhau tại ít nhất một vector đơn vị $\mathbf{v}$. Với $\mathbf{v}$ đó: $\hat{\mathbf{A}}\mathbf{v} = \mathbf{0}$ nhưng $\lVert \mathbf{A}\mathbf{v}\rVert^2 \ge \sigma_{k+1}^2$ (vì $\mathbf{v}$ là tổ hợp của $\mathbf{v}_1, \dots, \mathbf{v}_{k+1}$ và $\mathbf{A}$ khuếch đại chúng theo $\sigma_j$). Vậy $\lVert \mathbf{A} - \hat{\mathbf{A}}\rVert \ge \sigma_{k+1}$. $\blacksquare$

---

# 5. Thin SVD và Truncated SVD

**Thin (Compact) SVD.** Khi $m \gg n$ (hoặc $n \gg m$), phần lớn $\mathbf{U}$ chứa các vector ứng với giá trị suy biến bằng $0$ — không góp phần vào $\mathbf{A}$. Thin SVD chỉ giữ $r = \text{rank}(\mathbf{A})$ vector:

$$\mathbf{A} = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top, \quad \mathbf{U}_r \in \mathbb{R}^{m \times r}, \; \mathbf{V}_r \in \mathbb{R}^{n \times r}$$

**Truncated SVD.** Chỉ tính $k < r$ giá trị suy biến lớn nhất — đủ cho xấp xỉ hạng $k$. Với ma trận lớn, các thuật toán như **Lanczos** hay **randomized SVD** làm điều này hiệu quả hơn nhiều so với tính toàn bộ SVD.

```python
import numpy as np
from sklearn.utils.extmath import randomized_svd

# Randomized SVD - nhanh hơn nhiều với ma trận lớn
U, sigma, Vt = randomized_svd(A, n_components=50, random_state=42)

# Tái tạo xấp xỉ hạng 50
A_approx = U * sigma @ Vt
```

---

# 6. Ứng dụng 1: Giải bình phương tối thiểu

Cho hệ $\mathbf{A}\mathbf{x} = \mathbf{b}$ (thường $m > n$, quá xác định). Từ [hồi quy tuyến tính](linear-regression), nghiệm bình phương tối thiểu là $\hat{\mathbf{x}} = (\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top \mathbf{b} = \mathbf{A}^\dagger \mathbf{b}$, với $\mathbf{A}^\dagger$ là **giả nghịch đảo Moore–Penrose**.

Dùng SVD: $\mathbf{A}^\dagger = \mathbf{V}\boldsymbol{\Sigma}^\dagger \mathbf{U}^\top$, trong đó $\boldsymbol{\Sigma}^\dagger$ thay $\sigma_k \ne 0$ bởi $1/\sigma_k$ và giữ $0$ nguyên. Khi $\mathbf{A}^\top\mathbf{A}$ **gần suy biến** (một số $\sigma_k$ rất nhỏ), nghịch đảo trực tiếp **khuếch đại nhiễu**; dùng **truncated SVD** (bỏ các $\sigma_k$ nhỏ hơn ngưỡng) là một dạng điều chuẩn — tương tự Ridge regression nhưng hoạt động trên các giá trị suy biến.

---

# 7. Ứng dụng 2: Nén ảnh

Một ảnh xám $m \times n$ pixel là ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$. SVD cho $\mathbf{A} = \sum_k \sigma_k \mathbf{u}_k \mathbf{v}_k^\top$. Giữ lại $k$ số hạng đầu:

$$\mathbf{A}_k = \sum_{j=1}^{k} \sigma_j \mathbf{u}_j \mathbf{v}_j^\top$$

**Tỉ lệ nén.** Lưu toàn bộ ảnh cần $mn$ số; lưu $\mathbf{A}_k$ cần $k(m + n + 1)$ số (các $\mathbf{u}_j$, $\mathbf{v}_j$ và $\sigma_j$). Nén khi $k(m+n) < mn$, tức $k < mn/(m+n)$.

Chất lượng xấp xỉ kiểm soát được qua **tỉ lệ năng lượng giữ lại**: $\frac{\sum_{j=1}^k \sigma_j^2}{\sum_{j=1}^r \sigma_j^2}$. Chỉ $k = 50$ thành phần đầu của ảnh $512 \times 512$ thường cho ảnh trông khá tốt (>$99\%$ năng lượng).

---

# 8. Ứng dụng 3: Latent Semantic Analysis (LSA)

Trong xử lý ngôn ngữ tự nhiên, xây dựng ma trận **tần suất từ-tài liệu (term-document matrix)** $\mathbf{A}_{ij}$ = tần suất TF-IDF của từ $i$ trong tài liệu $j$ (xem [feature engineering](feature-engineering)). Ma trận này thường thưa ($m \approx 10^4$ từ, $n \approx 10^5$ tài liệu).

**Latent Semantic Analysis (LSA)** áp truncated SVD để tìm $k \approx 100$–$300$ chủ đề ẩn:

$$\mathbf{A} \approx \mathbf{A}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$$

Mỗi cột của $\mathbf{V}_k$ là biểu diễn tài liệu trong không gian $k$ chủ đề; mỗi cột của $\mathbf{U}_k$ là biểu diễn từ. Hai tài liệu "gần nhau" trong không gian chủ đề ẩn dù không chia sẻ từ — LSA bắt được **từ đồng nghĩa** và **ngữ cảnh** mà bag-of-words không nắm được.

---

# 9. Ứng dụng 4: Hệ gợi ý — nền tảng cho Matrix Factorization

Đây là ứng dụng cốt lõi sẽ kết nối sang bài [Matrix Factorization](matrix-factorization). Ma trận **đánh giá người dùng–sản phẩm (user-item rating matrix)** $\mathbf{R} \in \mathbb{R}^{m \times n}$ thường rất **thưa** (hầu hết ô trống vì người dùng chỉ đánh giá một phần nhỏ sản phẩm).

Ý tưởng: xấp xỉ $\mathbf{R} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$. Các hàng của $\mathbf{U}_k$ là **biểu diễn người dùng** trong không gian $k$ "sở thích ẩn"; các cột của $\mathbf{V}_k^\top$ là **biểu diễn sản phẩm**. Điểm dự đoán cho người dùng $i$ và sản phẩm $j$ là phần tử $(i,j)$ của $\mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$.

Nhưng SVD cổ điển đòi hỏi ma trận **đầy đủ** (không có ô trống). Thực tế phải dùng các biến thể ước lượng SVD trên dữ liệu thiếu — đó là [Matrix Factorization](matrix-factorization) với hàm mất mát tối ưu chỉ trên các ô đã quan sát.

---

# 10. Một số tính chất quan trọng

**Chuẩn phổ.** $\lVert \mathbf{A} \rVert_2 = \sigma_1$ — giá trị suy biến lớn nhất. Điều này vì $\max_{\lVert \mathbf{v}\rVert=1}\lVert \mathbf{A}\mathbf{v}\rVert = \sigma_1$ (đạt tại $\mathbf{v}_1$).

**Chuẩn Frobenius.** $\lVert \mathbf{A} \rVert_F = \sqrt{\sum_{k} \sigma_k^2}$.

**Hạng.** $\text{rank}(\mathbf{A}) = $ số giá trị suy biến khác $0$.

**Điều kiện số (condition number).** $\kappa(\mathbf{A}) = \sigma_1 / \sigma_r$ — đo mức độ "bất ổn định" của ma trận. $\kappa$ lớn ⇒ hệ $\mathbf{A}\mathbf{x}=\mathbf{b}$ nhạy cảm với nhiễu nhỏ ở $\mathbf{b}$.

**Ma trận giả nghịch đảo.**

$$\mathbf{A}^\dagger = \mathbf{V}\boldsymbol{\Sigma}^\dagger \mathbf{U}^\top, \quad \boldsymbol{\Sigma}^\dagger_{kk} = \begin{cases}1/\sigma_k & \sigma_k > 0 \\ 0 & \sigma_k = 0\end{cases}$$

---

# 11. Quan hệ giữa SVD và các phân rã khác

| Phân rã | Yêu cầu | Công thức | Ứng dụng |
|---|---|---|---|
| SVD | Mọi ma trận | $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ | PCA, bình phương tối thiểu, nén |
| Eigendecomposition | Ma trận vuông | $\mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$ | Phân tích động học |
| Spectral (PSD) | Đối xứng PSD | $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ | Hiệp phương sai, kernel |
| QR | Mọi ma trận | $\mathbf{Q}\mathbf{R}$ | Gram–Schmidt, bình phương tối thiểu |
| LU | Ma trận vuông | $\mathbf{L}\mathbf{U}$ | Giải hệ tuyến tính |

SVD là phân rã **tổng quát nhất và ổn định nhất về số học** — chi phí tính toán $O(\min(mn^2, m^2 n))$, nhưng cho hầu hết thông tin cần thiết.

---

# 12. Tổng kết

SVD phân rã **mọi ma trận** thành tích ba ma trận có cấu trúc đẹp $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, bộc lộ hình học bản chất của ánh xạ tuyến tính: các vector suy biến phải ($\mathbf{v}_k$) là các hướng đầu vào quan trọng, giá trị suy biến ($\sigma_k$) là mức khuếch đại, và vector suy biến trái ($\mathbf{u}_k$) là các hướng đầu ra tương ứng.

Ba kết quả nền tảng cần nhớ: (1) **Eckart–Young** — xấp xỉ hạng $k$ tốt nhất là $\mathbf{A}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top$; (2) **quan hệ với PCA** — cột của $\mathbf{V}$ là thành phần chính, $\lambda_k = \sigma_k^2/(N-1)$; (3) **giả nghịch đảo** — $\mathbf{A}^\dagger = \mathbf{V}\boldsymbol{\Sigma}^\dagger\mathbf{U}^\top$ giải bài toán bình phương tối thiểu ổn định.

> Bài tiếp theo — **Gợi ý dựa trên nội dung** — mở đầu phần hệ gợi ý, dùng đặc trưng của sản phẩm để xây mô hình hồi quy riêng cho từng người dùng.
