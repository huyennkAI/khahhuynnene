# Lọc cộng tác (Collaborative Filtering)

> Lọc cộng tác (Collaborative Filtering, CF) gợi ý sản phẩm dựa trên **hành vi tập thể**: không cần biết nội dung sản phẩm, chỉ cần biết ai thích gì. Trực giác: nếu bạn và người dùng B có lịch sử đánh giá rất giống nhau, thì sản phẩm nào B thích mà bạn chưa xem — bạn cũng có thể thích.
>
> Khác với [content-based filtering](content-based-rec) nhìn vào **sản phẩm**, CF nhìn vào **ma trận tương tác người dùng–sản phẩm** và khai thác cấu trúc của nó.

---

# 1. Ma trận đánh giá và bài toán điền khuyết

Toàn bộ CF xoay quanh **ma trận đánh giá (rating matrix)** $\mathbf{R} \in \mathbb{R}^{m \times n}$:

* Hàng $u$ ứng với người dùng $u$ ($m$ người dùng).
* Cột $i$ ứng với sản phẩm $i$ ($n$ sản phẩm).
* $R_{ui} = r_{ui}$ nếu người dùng $u$ đã đánh giá sản phẩm $i$, còn lại là **ô trống (missing)**.

Trong thực tế $\mathbf{R}$ cực kỳ **thưa**: một người dùng trung bình chỉ đánh giá $0.1\%$–$1\%$ số sản phẩm. **Bài toán CF là điền vào các ô trống** — dự đoán $\hat{r}_{ui}$ cho mọi cặp (người dùng, sản phẩm) chưa tương tác, để gợi ý $k$ sản phẩm có $\hat{r}_{ui}$ cao nhất.

---

# 2. Hai hướng tiếp cận: Memory-Based vs. Model-Based

Có hai họ phương pháp lọc cộng tác:

* **Memory-based (neighborhood-based)**: dùng trực tiếp các điểm đánh giá trong $\mathbf{R}$ để tính độ tương đồng và dự đoán. Không "học" tham số; mỗi dự đoán tra cứu trực tiếp.
* **Model-based**: xây một mô hình tham số học từ $\mathbf{R}$ (ví dụ [ma trận phân rã](matrix-factorization)). Sau khi huấn luyện, dự đoán chỉ cần tra bảng embedding.

Bài này tập trung vào **memory-based** — trực quan và không đòi hỏi huấn luyện. Model-based sẽ được trình bày trong [Matrix Factorization](matrix-factorization).

---

# 3. CF dựa trên người dùng (User-Based CF)

## 3.1. Ý tưởng

Để dự đoán $\hat{r}_{ui}$, tìm những người dùng **tương tự** $u$ (đã từng đánh giá sản phẩm $i$), rồi lấy điểm của họ trên sản phẩm $i$ làm cơ sở dự đoán.

## 3.2. Đo độ tương đồng người dùng

Mỗi người dùng $u$ được biểu diễn bởi vector đánh giá $\mathbf{r}_u \in \mathbb{R}^n$ (phần tử ô trống để $0$ hoặc tính riêng). Hai độ đo phổ biến:

**Độ tương đồng cosine** (chỉ trên sản phẩm cả hai cùng đánh giá):

$$\text{sim}(u, v) = \frac{\sum_{i \in \mathcal{I}_{uv}} r_{ui}\, r_{vi}}{\sqrt{\sum_{i \in \mathcal{I}_{uv}} r_{ui}^2}\,\sqrt{\sum_{i \in \mathcal{I}_{uv}} r_{vi}^2}}$$

**Tương quan Pearson** (trừ trung bình trước để loại thiên lệch chấm điểm):

$$\text{sim}(u, v) = \frac{\sum_{i \in \mathcal{I}_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i} (r_{ui} - \bar{r}_u)^2}\,\sqrt{\sum_{i} (r_{vi} - \bar{r}_v)^2}}$$

Tương quan Pearson tốt hơn vì xử lý được **thiên lệch thang điểm**: người dùng A luôn cho $4$–$5$ sao, người B luôn cho $1$–$3$ sao, nhưng cả hai thực sự thích những thứ giống nhau. Pearson chuẩn hóa điều đó.

## 3.3. Dự đoán

Gọi $\mathcal{N}_k(u)$ là $k$ người dùng tương đồng nhất với $u$ (đã đánh giá sản phẩm $i$). Điểm dự đoán:

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v)\,(r_{vi} - \bar{r}_v)}{\sum_{v \in \mathcal{N}_k(u)} |\text{sim}(u, v)|}$$

Trừ và cộng trung bình để điều chỉnh thiên lệch: ta dự đoán $u$ sẽ đánh giá sản phẩm này **cao hơn hay thấp hơn** mức bình thường của mình bao nhiêu, dựa trên mức các láng giềng đánh giá cao/thấp hơn mức của họ.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_cf_predict(R, user, item, k=10):
    """R: ma trận đánh giá (m x n), ô trống = 0"""
    sims = cosine_similarity(R)[user]           # (m,) similarity với user
    sims[user] = -1                             # loại bản thân
    # Chỉ xét láng giềng đã đánh giá item
    rated = np.where(R[:, item] > 0)[0]
    if len(rated) == 0:
        return R[user].mean()
    top_k = rated[np.argsort(sims[rated])[::-1][:k]]
    weights = sims[top_k]
    scores = R[top_k, item]
    if weights.sum() == 0:
        return R[user].mean()
    return np.dot(weights, scores) / np.abs(weights).sum()
```

---

# 4. CF dựa trên sản phẩm (Item-Based CF)

## 4.1. Ý tưởng đổi chiều

Thay vì "tìm người tương tự tôi", đổi câu hỏi thành "tìm sản phẩm tương tự sản phẩm tôi đã thích". Trực giác: sản phẩm **tương đồng** là những sản phẩm được người dùng đánh giá tương tự nhau.

## 4.2. Độ tương đồng sản phẩm

Mỗi sản phẩm $i$ biểu diễn bởi vector đánh giá $\mathbf{r}_i \in \mathbb{R}^m$ (cột của $\mathbf{R}$). Tính ma trận tương đồng sản phẩm $\mathbf{S} \in \mathbb{R}^{n \times n}$ bằng cosine:

$$S_{ij} = \frac{\mathbf{r}_i^\top \mathbf{r}_j}{\lVert \mathbf{r}_i \rVert \lVert \mathbf{r}_j \rVert}$$

chỉ trên người dùng đã đánh giá cả hai sản phẩm.

## 4.3. Dự đoán

$$\hat{r}_{ui} = \frac{\sum_{j \in \mathcal{N}_k(i) \cap \mathcal{I}_u} S_{ij}\, r_{uj}}{\sum_{j \in \mathcal{N}_k(i) \cap \mathcal{I}_u} |S_{ij}|}$$

Trong đó $\mathcal{N}_k(i)$ là $k$ sản phẩm tương đồng nhất với $i$, và $\mathcal{I}_u$ là sản phẩm người dùng $u$ đã đánh giá. Dự đoán điểm người dùng $u$ cho sản phẩm $i$ là trung bình có trọng số điểm người đó đã cho các sản phẩm tương tự.

## 4.4. Lợi thế của Item-Based CF

* **Ổn định hơn**: ma trận tương đồng sản phẩm thay đổi chậm (sản phẩm ít hơn người dùng, thêm người dùng mới không ảnh hưởng nhiều), nên có thể **tính trước (precompute)** và cache.
* **Tốc độ dự đoán nhanh hơn**: chỉ cần tra bảng $S_{ij}$ đã tính sẵn, không phải tính lại similarity mỗi lần.
* **Amazon** đã dùng item-based CF từ đầu những năm 2000 (Linden et al., 2003) và nó là nền tảng của "khách hàng mua sản phẩm này cũng mua...".

---

# 5. So sánh User-Based và Item-Based CF

| | User-Based | Item-Based |
|---|---|---|
| Độ tương đồng | Giữa người dùng | Giữa sản phẩm |
| Ổn định | Thấp (người dùng thay đổi) | Cao (sản phẩm ổn định) |
| Có thể precompute | Khó (thay đổi nhanh) | ✓ Dễ cache |
| Số lượng | Thường nhiều người dùng hơn sản phẩm | ✓ Ít sản phẩm hơn người dùng |
| Diễn giải | "Người như bạn thích X" | "Vì bạn thích Y, thử X" |
| Phù hợp khi | Ít người dùng, nhiều sản phẩm | Ít sản phẩm, nhiều người dùng |

---

# 6. Vấn đề cold-start và sparsity

## 6.1. Cold-Start

* **Người dùng mới**: không có lịch sử đánh giá → không tính được similarity. Giải pháp: hỏi một số sản phẩm mồi (onboarding), dùng thông tin nhân khẩu học, hoặc kết hợp với content-based.
* **Sản phẩm mới**: chưa ai đánh giá → không xuất hiện trong similarity matrix. Giải pháp: content-based cho sản phẩm mới, chuyển sang CF sau khi có đủ đánh giá.

## 6.2. Sparsity

Với ma trận cực thưa, hầu hết cặp người dùng/sản phẩm không có sản phẩm/người dùng chung để tính similarity → similarity = 0 cho phần lớn cặp. Một số giải pháp:

* **Giảm chiều**: chiếu dữ liệu xuống không gian ít chiều hơn (SVD, [matrix factorization](matrix-factorization)) trước khi tính similarity.
* **Implicit feedback**: thay vì đòi điểm đánh giá tường minh (explicit), dùng hành vi ngầm định — click, xem, mua — thì ma trận dày hơn nhiều.

## 6.3. Implicit Feedback

Hầu hết dữ liệu thực tế là **implicit**: không có điểm đánh giá $1$–$5$, chỉ có nhị phân "tương tác hay không" hoặc số lần tương tác. $r_{ui} \in \{0, 1\}$ (mua/không mua) hoặc số lần nghe nhạc. Xử lý implicit data cần điều chỉnh: các ô $0$ không có nghĩa "không thích" mà có thể là "chưa biết đến". Mô hình phải phân biệt được "không tương tác vì không thích" và "không tương tác vì chưa gặp".

---

# 7. Đánh giá hệ gợi ý

Đánh giá hệ gợi ý khác với đánh giá bộ phân loại thông thường. Các độ đo phổ biến:

**Độ đo dự đoán điểm:** RMSE và MAE trên tập test:

$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u,i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2}$$

**Độ đo xếp hạng (ranking):** quan trọng hơn RMSE trong thực tế — gợi ý đúng thứ tự quan trọng hơn dự đoán chính xác điểm số.

* **Precision@k**: tỉ lệ sản phẩm liên quan trong top-$k$ gợi ý.
* **Recall@k**: tỉ lệ sản phẩm liên quan được tìm thấy trong top-$k$.
* **NDCG@k** (Normalized Discounted Cumulative Gain): tính đến thứ hạng — gợi ý đúng ở vị trí cao hơn thì điểm cao hơn.

**Đánh giá offline vs. online:** offline dùng dữ liệu lịch sử; online A/B test trực tiếp đo conversion rate, click-through rate trên người dùng thật.

---

# 8. Hạn chế của Memory-Based CF

* **Chi phí tính toán** — tính ma trận similarity $m \times m$ hoặc $n \times n$ tốn $O(m^2)$ hoặc $O(n^2)$; với hàng triệu người dùng/sản phẩm, không khả thi.
* **Kém với dữ liệu thưa** — cần đủ điểm chung để tính similarity có ý nghĩa.
* **Không tổng quát hóa** — không học được pattern ẩn, chỉ dựa trên co-occurrence trực tiếp.

Những hạn chế này là động lực cho **model-based CF** — cụ thể là [matrix factorization](matrix-factorization), học biểu diễn ẩn (latent representation) của người dùng và sản phẩm qua gradient descent, cho phép xử lý dữ liệu thưa tốt hơn và mở rộng hàng triệu người dùng.

---

# 9. Tổng kết

Lọc cộng tác khai thác cấu trúc của ma trận đánh giá $\mathbf{R}$ để điền vào các ô trống:

* **User-based CF**: tìm $k$ người dùng tương đồng nhất, dự đoán dựa trên điểm của họ.
* **Item-based CF**: tìm $k$ sản phẩm tương đồng nhất với sản phẩm đã đánh giá, dùng weighted average.

Tương đồng đo bằng **cosine** hoặc **Pearson** (tốt hơn vì trừ bias thang điểm). Item-based thường ưa hơn vì ổn định và có thể precompute. Vấn đề lớn nhất là **cold-start** và **sparsity** — giải quyết bằng kết hợp content-based hoặc chuyển sang model-based CF.

> Bài tiếp theo — **Matrix Factorization** — là bước nâng cấp model-based: thay vì tính similarity trực tiếp, học biểu diễn ẩn cho người dùng và sản phẩm bằng tối ưu hóa gradient descent.
