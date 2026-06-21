# Phân tích thành phần chính (PCA)

> Phân tích thành phần chính (Principal Component Analysis, PCA) tìm một **không gian con tuyến tính** để chiếu dữ liệu xuống sao cho **phương sai** của dữ liệu chiếu là **lớn nhất**. Nói cách khác: ta tìm các hướng mà dữ liệu "trải rộng" nhiều nhất, giữ lại $d$ hướng quan trọng nhất, bỏ đi phần còn lại.
>
> $$\mathbf{w}_1 = \arg\max_{\lVert \mathbf{w} \rVert = 1} \operatorname{Var}(\mathbf{w}^\top \mathbf{X}^\top) = \arg\max_{\lVert \mathbf{w} \rVert = 1} \mathbf{w}^\top \mathbf{S} \mathbf{w}$$
>
> Nghiệm là **vector riêng** của ma trận hiệp phương sai $\mathbf{S}$ — một kết quả thanh lịch nối đại số tuyến tính với thống kê.

---

# 1. Động cơ: vì sao giảm chiều?

Dữ liệu thực tế thường có hàng trăm hoặc hàng nghìn chiều (đặc trưng), nhưng cấu trúc thực sự có thể nằm trên một **đa tạp (manifold)** ít chiều hơn nhiều. Ba lý do chính cần giảm chiều:

* **Trực quan hóa** — không thể vẽ dữ liệu $d > 3$ chiều; chiếu xuống 2D hay 3D giúp ta "nhìn thấy" cấu trúc.
* **Loại nhiễu** — các chiều ít phương sai thường chứa nhiễu hơn tín hiệu; bỏ chúng đi cải thiện mô hình.
* **Tiết kiệm tính toán** — đặc trưng ít hơn thì mô hình huấn luyện nhanh hơn và ít [overfit](overfitting) hơn.

PCA là phương pháp giảm chiều **tuyến tính, không giám sát (unsupervised)** phổ biến nhất. Nó không dùng nhãn (khác với [LDA](lda)) và tìm cấu trúc thuần túy từ phân phối của dữ liệu $\mathbf{x}$.

---

# 2. Phương sai và ma trận hiệp phương sai

## 2.1. Trung tâm hóa dữ liệu

Trước tiên **trừ trung bình** để dữ liệu có tâm tại gốc tọa độ. Gọi $\mathbf{m} = \frac{1}{N}\sum_i \mathbf{x}_i$ là trung bình mẫu, đặt $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \mathbf{m}$. Từ đây ta ngầm hiểu dữ liệu đã được trung tâm hóa.

## 2.2. Ma trận hiệp phương sai mẫu

**Ma trận hiệp phương sai mẫu (sample covariance matrix)** $\mathbf{S} \in \mathbb{R}^{d \times d}$:

$$\mathbf{S} = \frac{1}{N-1} \sum_{i=1}^{N} \mathbf{x}_i \mathbf{x}_i^\top = \frac{1}{N-1} \mathbf{X}^\top \mathbf{X}$$

trong đó $\mathbf{X} \in \mathbb{R}^{N \times d}$ là ma trận dữ liệu đã trung tâm hóa (mỗi hàng là một mẫu). $\mathbf{S}$ **đối xứng** và **nửa xác định dương (positive semi-definite, PSD)**: với mọi $\mathbf{v}$, $\mathbf{v}^\top \mathbf{S} \mathbf{v} = \frac{1}{N-1}\sum_i (\mathbf{x}_i^\top \mathbf{v})^2 \ge 0$.

Thành phần $S_{jk}$ đo sự **đồng biến** giữa chiều $j$ và chiều $k$; đường chéo $S_{jj} = \operatorname{Var}(x_j)$. Ma trận này tóm tắt toàn bộ cấu trúc hình học bậc hai của đám mây điểm.

---

# 3. Thành phần chính đầu tiên

Hướng chiếu $\mathbf{w} \in \mathbb{R}^d$ (đơn vị: $\lVert \mathbf{w} \rVert = 1$) cho giá trị chiếu $z_i = \mathbf{w}^\top \mathbf{x}_i$. Phương sai của các giá trị chiếu:

$$\operatorname{Var}(\{z_i\}) = \frac{1}{N-1}\sum_i z_i^2 = \frac{1}{N-1}\sum_i (\mathbf{w}^\top \mathbf{x}_i)^2 = \mathbf{w}^\top \underbrace{\left(\frac{1}{N-1}\mathbf{X}^\top \mathbf{X}\right)}_{\mathbf{S}} \mathbf{w} = \mathbf{w}^\top \mathbf{S} \mathbf{w}$$

**Thành phần chính thứ nhất (first principal component)** là hướng cực đại hóa phương sai chiếu:

$$\mathbf{w}_1 = \arg\max_{\lVert \mathbf{w} \rVert = 1} \mathbf{w}^\top \mathbf{S} \mathbf{w}$$

Đây là bài toán **tối ưu có ràng buộc**. Dùng nhân tử Lagrange từ [tối ưu lồi](convex-optimization): tối ưu $\mathbf{w}^\top \mathbf{S} \mathbf{w} - \lambda(\lVert \mathbf{w} \rVert^2 - 1)$. Lấy đạo hàm và cho bằng $\mathbf{0}$:

$$2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0} \;\Rightarrow\; \boxed{\mathbf{S}\mathbf{w} = \lambda\mathbf{w}}$$

**$\mathbf{w}_1$ phải là vector riêng (eigenvector) của $\mathbf{S}$.** Giá trị mục tiêu tại nghiệm là $\mathbf{w}^\top \mathbf{S} \mathbf{w} = \mathbf{w}^\top \lambda \mathbf{w} = \lambda$. Vậy để **cực đại hóa** phương sai, ta chọn **vector riêng ứng với trị riêng lớn nhất** $\lambda_1$. $\blacksquare$

---

# 4. Các thành phần chính tiếp theo

Thành phần chính thứ hai $\mathbf{w}_2$ phải **trực giao** với $\mathbf{w}_1$ (để không trùng lặp thông tin đã giữ) và cực đại hóa phương sai còn lại:

$$\mathbf{w}_2 = \arg\max_{\substack{\lVert \mathbf{w} \rVert = 1 \\ \mathbf{w} \perp \mathbf{w}_1}} \mathbf{w}^\top \mathbf{S} \mathbf{w}$$

Cùng lập luận Lagrange, thêm ràng buộc $\mathbf{w}_1^\top \mathbf{w} = 0$. Có thể chứng minh (bằng cách trực tiếp hoặc qua phân tích phổ) rằng $\mathbf{w}_2$ là vector riêng ứng với **trị riêng lớn thứ hai** $\lambda_2 \le \lambda_1$.

**Kết luận tổng quát.** $k$ thành phần chính là $k$ vector riêng đầu tiên của $\mathbf{S}$, ứng với $k$ trị riêng lớn nhất $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_k > 0$. Chúng đôi một trực giao (vì $\mathbf{S}$ đối xứng và các trị riêng phân biệt). Gọi ma trận chiếu $\mathbf{W} = [\mathbf{w}_1, \dots, \mathbf{w}_d] \in \mathbb{R}^{d \times d}$; phép chiếu xuống $d$ chiều là $\mathbf{z}_i = \mathbf{W}_{:d}^\top \mathbf{x}_i \in \mathbb{R}^d$.

---

# 5. Phân tích phổ của ma trận hiệp phương sai

Ma trận $\mathbf{S}$ đối xứng PSD nên theo **định lý phân tích phổ (spectral theorem)** có phân tích:

$$\mathbf{S} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top$$

trong đó $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_d]$ là ma trận trực giao (các cột là vector riêng) và $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_d)$ với $\lambda_1 \ge \cdots \ge \lambda_d \ge 0$.

Trong hệ tọa độ mới $\mathbf{z} = \mathbf{V}^\top \mathbf{x}$, ma trận hiệp phương sai trở thành **đường chéo** $\boldsymbol{\Lambda}$: các chiều mới **không tương quan** với nhau, và phương sai theo chiều $k$ chính xác là $\lambda_k$.

> **Trực giác hình học.** PCA tìm trục chính của "đám mây điểm". Trục dài nhất (phương sai lớn nhất) là $\mathbf{v}_1$, trục dài thứ hai vuông góc với nó là $\mathbf{v}_2$, v.v. Chiếu dữ liệu lên $d$ trục đầu giữ lại $d$ chiều quan trọng nhất của hình học dữ liệu.

---

# 6. Tỉ lệ phương sai giải thích

Chọn bao nhiêu thành phần? **Tỉ lệ phương sai giải thích (explained variance ratio)** của $d$ thành phần đầu:

$$\text{EVR}(d) = \frac{\sum_{k=1}^{d} \lambda_k}{\sum_{k=1}^{D} \lambda_k}$$

Thực hành: vẽ **scree plot** (đồ thị $\lambda_k$ theo $k$) hoặc **tích lũy EVR** theo $d$. Chọn $d$ sao cho EVR đạt ngưỡng mong muốn, thường $85$–$95\%$:

```python
import numpy as np

# Sau khi đã có giá trị riêng lambda_
cumulative_evr = np.cumsum(lambdas) / lambdas.sum()
d = np.argmax(cumulative_evr >= 0.95) + 1  # số chiều giữ 95% phương sai
```

---

# 7. PCA qua SVD

Trên thực tế, **không nên** tính $\mathbf{S} = \mathbf{X}^\top \mathbf{X}/(N-1)$ rồi giải trị riêng trực tiếp — bình phương điều kiện của ma trận (condition number) sẽ tăng lên bình phương, gây mất độ chính xác số học. Thay vào đó, dùng **phân rã giá trị suy biến (SVD)** của ma trận dữ liệu:

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

trong đó $\mathbf{U} \in \mathbb{R}^{N \times N}$ và $\mathbf{V} \in \mathbb{R}^{d \times d}$ là ma trận trực giao, $\boldsymbol{\Sigma}$ là ma trận đường chéo với các **giá trị suy biến (singular values)** $\sigma_1 \ge \cdots \ge \sigma_{\min(N,d)} \ge 0$. Quan hệ với hiệp phương sai:

$$\mathbf{S} = \frac{1}{N-1}\mathbf{X}^\top \mathbf{X} = \frac{1}{N-1}\mathbf{V}\boldsymbol{\Sigma}^2 \mathbf{V}^\top$$

Nên $\mathbf{v}_k$ (cột của $\mathbf{V}$) **chính là** các thành phần chính, và $\lambda_k = \sigma_k^2/(N-1)$. Chi tiết toán học của SVD trong bài [SVD](svd).

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)          # giữ đủ thành phần cho 95% phương sai
Z = pca.fit_transform(X)              # chiều mới: (N, d_optimal)

print(pca.explained_variance_ratio_)  # tỉ lệ phương sai mỗi thành phần
print(pca.components_)                # các thành phần chính (hàng = w_k)
```

---

# 8. Tái tạo và sai số xấp xỉ

Sau khi chiếu xuống $d$ chiều, ta có thể **tái tạo** xấp xỉ:

$$\hat{\mathbf{x}}_i = \mathbf{W}_d \mathbf{z}_i + \mathbf{m} = \sum_{k=1}^{d} z_{ik} \mathbf{w}_k + \mathbf{m}$$

**Sai số tái tạo** (reconstruction error) trung bình:

$$\mathbb{E}\lVert \mathbf{x} - \hat{\mathbf{x}} \rVert^2 = \sum_{k=d+1}^{D} \lambda_k$$

Tức sai số chính xác là **tổng các trị riêng bị bỏ**. Chọn $d$ lớn hơn thì sai số nhỏ hơn, nhưng chiều biểu diễn tăng. Đây cũng là góc nhìn PCA như **autoencoder tuyến tính** tối ưu: bộ mã hóa $\mathbf{W}_d^\top$ và bộ giải mã $\mathbf{W}_d$ là nghiệm của bài toán tối thiểu sai số tái tạo trong không gian tuyến tính.

---

# 9. PCA và whitening

**Whitening** (hay **sphering**) là bước đi xa hơn: không chỉ chiếu lên các thành phần chính mà còn **chuẩn hóa phương sai** mỗi chiều về $1$. Sau khi có $\mathbf{z} = \mathbf{V}^\top \mathbf{x}$, chia mỗi chiều cho độ lệch chuẩn:

$$\tilde{z}_k = \frac{z_k}{\sqrt{\lambda_k}} = \frac{\mathbf{v}_k^\top \mathbf{x}}{\sqrt{\lambda_k}}$$

Dữ liệu whitened có hiệp phương sai bằng ma trận đơn vị $\mathbf{I}$ — "đám mây điểm" trở thành hình cầu. Bước này thường dùng như tiền xử lý cho ICA (Independent Component Analysis), hoặc để làm gradient descent hội tụ nhanh hơn khi ma trận hiệp phương sai lệch lạc (liên hệ với lý do chuẩn hóa đặc trưng trong [feature engineering](feature-engineering) và mục 8.1 của bài đó).

---

# 10. So sánh PCA và LDA

| | PCA | LDA |
|---|---|---|
| Giám sát? | Không | Có (dùng nhãn) |
| Tối đa hóa | Phương sai chiếu | Tỉ số tách lớp / tản lớp |
| Bài toán trị riêng | $\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$ | $\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$ |
| Số chiều tối đa | $\min(N-1, D)$ | $C - 1$ |
| Mục đích chính | Nén, trực quan, loại nhiễu | Phân loại, tách lớp |

Dùng PCA khi **không có nhãn** hoặc khi mục tiêu là **nén/trực quan hóa**. Dùng [LDA](lda) khi **có nhãn** và mục tiêu là **phân loại**.

---

# 11. Hạn chế của PCA

* **Chỉ tuyến tính** — không bắt được cấu trúc phi tuyến (đa tạp cong). Với cấu trúc phi tuyến dùng t-SNE, UMAP, hay kernel PCA.
* **Không giám sát** — có thể bỏ chiều quan trọng cho phân loại nhưng ít phương sai (xem mục 1 bài [LDA](lda)).
* **Nhạy với thang đo** — nếu các đặc trưng có đơn vị khác nhau, đặc trưng có thang lớn sẽ chi phối phương sai. **Bắt buộc chuẩn hóa (standardize)** trước khi áp PCA trừ khi tất cả đặc trưng cùng đơn vị.
* **Nhạy với outlier** — outlier phóng đại phương sai theo một hướng, có thể làm lệch thành phần chính. Dùng Robust PCA trong trường hợp đó.

---

# 12. Tổng kết

PCA tìm $d$ hướng tuyến tính giữ lại nhiều **phương sai** nhất bằng cách giải bài toán trị riêng của **ma trận hiệp phương sai**:

$$\mathbf{S}\mathbf{w}_k = \lambda_k \mathbf{w}_k, \qquad \lambda_1 \ge \lambda_2 \ge \cdots$$

Phương sai chiếu theo hướng $\mathbf{w}_k$ đúng bằng $\lambda_k$, nên giữ $d$ vector riêng đầu là giữ lại phần lớn nhất thông tin. Trong thực tế, tính qua **SVD** của ma trận dữ liệu thay vì trực tiếp trên $\mathbf{S}$ để đảm bảo ổn định số học. Chọn $d$ bằng **scree plot** hoặc ngưỡng **EVR** ($95\%$ là thông dụng).

Ba quy tắc vàng: **chuẩn hóa dữ liệu** trước PCA; chọn số chiều qua **tỉ lệ phương sai giải thích**; và nhớ rằng PCA **không giám sát** — nếu có nhãn và cần phân loại, hãy cân nhắc [LDA](lda).

> Bài tiếp theo — **SVD** — đào sâu vào phân rã giá trị suy biến, nền toán học của PCA, và ứng dụng của nó trong xấp xỉ ma trận hạng thấp, nén ảnh và hệ gợi ý.
