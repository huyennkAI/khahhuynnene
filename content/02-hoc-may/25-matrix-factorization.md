# Phân rã ma trận cho gợi ý (Matrix Factorization)

> Phân rã ma trận (Matrix Factorization, MF) là cách tiếp cận **model-based** cho hệ gợi ý: thay vì tính độ tương đồng trực tiếp giữa người dùng hay sản phẩm như trong [lọc cộng tác](collaborative-filtering), ta học **biểu diễn ẩn (latent representation)** cho từng người dùng và sản phẩm dưới dạng vector thấp chiều, sao cho tích vô hướng của hai vector xấp xỉ điểm đánh giá thực tế.
>
> $$\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$
>
> $\mathbf{p}_u \in \mathbb{R}^k$ là embedding của người dùng $u$, $\mathbf{q}_i \in \mathbb{R}^k$ là embedding của sản phẩm $i$. Không gian $k$ chiều (thường $k \ll m, n$) chứa các **nhân tố ẩn** (latent factors) — có thể là thể loại, phong cách, mức độ phức tạp — mà mô hình tự học mà không cần nhãn thủ công.

---

# 1. Từ ma trận đánh giá đến phân rã hạng thấp

Nhắc lại từ [lọc cộng tác](collaborative-filtering): ma trận đánh giá $\mathbf{R} \in \mathbb{R}^{m \times n}$ rất **thưa** — hầu hết ô trống. Mục tiêu là điền vào các ô trống đó.

**Giả định cốt lõi của MF**: $\mathbf{R}$ có cấu trúc **hạng thấp** — dù có hàng triệu người dùng và sản phẩm, hành vi thực sự chỉ bị chi phối bởi $k$ nhân tố ẩn ($k$ nhỏ, thường $20$–$200$). Ví dụ: với phim, $k$ nhân tố ẩn có thể tương ứng với mức độ "hành động", "lãng mạn", "nghệ thuật"...

Ta xấp xỉ:

$$\mathbf{R} \approx \mathbf{P}\mathbf{Q}^\top$$

Trong đó:
* $\mathbf{P} \in \mathbb{R}^{m \times k}$: ma trận embedding người dùng — hàng $u$ là $\mathbf{p}_u$.
* $\mathbf{Q} \in \mathbb{R}^{n \times k}$: ma trận embedding sản phẩm — hàng $i$ là $\mathbf{q}_i$.

**Khác với SVD**: [SVD](svd) phân rã ma trận đầy đủ theo nghĩa chính xác; với $\mathbf{R}$ thưa và nhiễu, ta **tối ưu hóa trực tiếp** $\mathbf{P}$ và $\mathbf{Q}$ chỉ trên các ô đã quan sát.

---

# 2. Hàm mục tiêu

## 2.1. Bình phương tối thiểu có điều chuẩn

Học $\mathbf{P}$ và $\mathbf{Q}$ bằng cách tối thiểu lỗi bình phương trên các ô đã biết $\Omega = \{(u, i) : r_{ui} \text{ đã quan sát}\}$:

$$\mathcal{L}(\mathbf{P}, \mathbf{Q}) = \sum_{(u,i) \in \Omega} \big(r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\big)^2 + \lambda\big(\lVert \mathbf{P} \rVert_F^2 + \lVert \mathbf{Q} \rVert_F^2\big)$$

Điều chuẩn Frobenius $\lambda(\lVert \mathbf{P} \rVert_F^2 + \lVert \mathbf{Q} \rVert_F^2)$ ngăn [overfitting](overfitting) — với nhiều tham số ($(m+n) \times k$ phần tử) nhưng ít ràng buộc từ dữ liệu thưa, không có điều chuẩn thì mô hình sẽ ghi nhớ dữ liệu huấn luyện thay vì tổng quát hóa.

## 2.2. Mô hình bias

Trong thực tế, điểm đánh giá bị ảnh hưởng bởi **thiên lệch**:

* Người dùng $u$ có xu hướng cho điểm cao hoặc thấp hơn trung bình (bias $b_u$).
* Sản phẩm $i$ vốn được đánh giá cao hoặc thấp (bias $b_i$).

Mô hình đầy đủ:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

Trong đó $\mu$ là trung bình toàn cục. Thêm bias cải thiện đáng kể kết quả vì tách biệt được **sở thích tương đối** (qua tích vô hướng) với **mức nền** (qua bias).

---

# 3. Tối ưu hóa

## 3.1. Stochastic Gradient Descent (SGD)

Với mỗi điểm quan sát $(u, i, r_{ui})$, tính lỗi:

$$e_{ui} = r_{ui} - \hat{r}_{ui} = r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i$$

Gradient theo từng embedding:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{p}_u} = -2e_{ui}\mathbf{q}_i + 2\lambda\mathbf{p}_u, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{q}_i} = -2e_{ui}\mathbf{p}_u + 2\lambda\mathbf{q}_i$$

Cập nhật SGD (bỏ hệ số $2$ vào learning rate $\eta$):

$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta\,(e_{ui}\mathbf{q}_i - \lambda\mathbf{p}_u)$$
$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta\,(e_{ui}\mathbf{p}_u - \lambda\mathbf{q}_i)$$

SGD rất hiệu quả: mỗi bước cập nhật chỉ $O(k)$ phép tính, xử lý được hàng triệu điểm quan sát.

## 3.2. Alternating Least Squares (ALS)

Thay vì gradient descent, **cố định $\mathbf{Q}$ và tối ưu $\mathbf{P}$, rồi cố định $\mathbf{P}$ và tối ưu $\mathbf{Q}$** — xen kẽ cho đến hội tụ. Khi cố định một bên, bài toán trở thành hồi quy tuyến tính có nghiệm đóng:

$$\mathbf{p}_u = \big(\mathbf{Q}_{\mathcal{I}_u}^\top \mathbf{Q}_{\mathcal{I}_u} + \lambda \mathbf{I}\big)^{-1} \mathbf{Q}_{\mathcal{I}_u}^\top \mathbf{r}_u$$

Trong đó $\mathbf{Q}_{\mathcal{I}_u}$ là các hàng của $\mathbf{Q}$ ứng với sản phẩm người dùng $u$ đã đánh giá. ALS song song hóa tốt hơn SGD (mỗi người dùng/sản phẩm độc lập) và ổn định hơn — phù hợp với hệ phân tán như Spark.

```python
import numpy as np

def matrix_factorization_sgd(R, k=20, lr=0.01, reg=0.1, n_epochs=50):
    """
    R: ma trận đánh giá (m x n), 0 = chưa đánh giá
    k: số nhân tố ẩn
    """
    m, n = R.shape
    # Khởi tạo ngẫu nhiên nhỏ
    P = np.random.normal(0, 0.1, (m, k))
    Q = np.random.normal(0, 0.1, (n, k))
    
    # Lấy các ô đã quan sát
    observed = [(u, i, R[u, i]) for u in range(m) for i in range(n) if R[u, i] > 0]
    
    for epoch in range(n_epochs):
        np.random.shuffle(observed)
        total_loss = 0
        for u, i, r in observed:
            e = r - P[u] @ Q[i]
            P[u] += lr * (e * Q[i] - reg * P[u])
            Q[i] += lr * (e * P[u] - reg * Q[i])
            total_loss += e**2
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: RMSE = {np.sqrt(total_loss/len(observed)):.4f}")
    return P, Q

# Dự đoán
# R_hat = P @ Q.T
# Gợi ý cho user u: np.argsort(R_hat[u])[::-1]
```

---

# 4. Implicit Feedback và mô hình iALS

Hầu hết dữ liệu thực tế là **implicit** (xem [lọc cộng tác](collaborative-filtering)): không có điểm $1$–$5$, chỉ có nhị phân "tương tác hay không". Định nghĩa:

$$p_{ui} = \begin{cases} 1 & \text{nếu } r_{ui} > 0 \\ 0 & \text{ngược lại} \end{cases}, \qquad c_{ui} = 1 + \alpha \cdot r_{ui}$$

$p_{ui}$ là **preference** (thích hay không), $c_{ui}$ là **confidence** (mức độ tin tưởng). Ô có $r_{ui} > 0$ (đã tương tác nhiều lần) có confidence cao hơn; ô $r_{ui} = 0$ (chưa tương tác) vẫn được đưa vào nhưng với confidence thấp ($c_{ui} = 1$).

Bài toán tối ưu:

$$\mathcal{L} = \sum_{u,i} c_{ui}\big(p_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\big)^2 + \lambda\big(\lVert \mathbf{P} \rVert_F^2 + \lVert \mathbf{Q} \rVert_F^2\big)$$

Tổng trên **toàn bộ** cặp $(u, i)$ — không chỉ ô đã quan sát — nhưng ô chưa tương tác có $c_{ui} = 1$ nhỏ, còn ô đã tương tác có $c_{ui}$ lớn. ALS giải bài toán này hiệu quả (Hu et al., 2008).

---

# 5. Regularization và Bias trong thực tế

Mô hình đầy đủ có bias, thường dùng trong các cuộc thi (Netflix Prize):

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

Hàm mất mát:

$$\mathcal{L} = \sum_{(u,i) \in \Omega} \big(r_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\big)^2 + \lambda\big(\lVert \mathbf{P} \rVert_F^2 + \lVert \mathbf{Q} \rVert_F^2 + \sum_u b_u^2 + \sum_i b_i^2\big)$$

Cập nhật SGD thêm cho bias:

$$b_u \leftarrow b_u + \eta\,(e_{ui} - \lambda b_u), \qquad b_i \leftarrow b_i + \eta\,(e_{ui} - \lambda b_i)$$

---

# 6. Các biến thể nâng cao

## 6.1. SVD++ — tích hợp implicit feedback

**SVD++** (Koren, 2008) mở rộng bằng cách tích hợp thêm thông tin về **tập sản phẩm người dùng đã tương tác** $\mathcal{I}_u$:

$$\hat{r}_{ui} = \mu + b_u + b_i + \left(\mathbf{p}_u + |\mathcal{I}_u|^{-1/2}\sum_{j \in \mathcal{I}_u} \mathbf{y}_j\right)^\top \mathbf{q}_i$$

$\mathbf{y}_j \in \mathbb{R}^k$ là "implicit item factor" của sản phẩm $j$. Vector trong ngoặc là biểu diễn người dùng được làm giàu bởi lịch sử tương tác của họ. SVD++ đạt kết quả tốt hơn nhưng chậm hơn vì phải tổng trên $\mathcal{I}_u$.

## 6.2. Probabilistic Matrix Factorization (PMF)

Diễn giải xác suất của MF: đặt prior Gaussian lên $\mathbf{p}_u$ và $\mathbf{q}_i$, likelihood Gaussian lên $r_{ui}$. Tối đa hóa MAP tương đương bình phương tối thiểu có điều chuẩn $L_2$ — liên kết MF với framework [MLE và MAP](mle-map).

## 6.3. Neural Collaborative Filtering (NCF)

Thay tích vô hướng $\mathbf{p}_u^\top \mathbf{q}_i$ bằng **mạng nơ-ron**:

$$\hat{r}_{ui} = f_\theta([\mathbf{p}_u \,\|\, \mathbf{q}_i])$$

Trong đó $[\cdot \| \cdot]$ là ghép nối (concatenation), $f_\theta$ là MLP nhiều lớp. Mạng nơ-ron học được tương tác **phi tuyến** giữa người dùng và sản phẩm mà tích vô hướng không nắm được. Tuy nhiên, nghiên cứu gần đây (Rendle et al., 2020) cho thấy MF được điều chỉnh tốt thường cạnh tranh được với NCF khi tập dữ liệu đủ thưa.

---

# 7. So sánh các phương pháp gợi ý

| | Content-Based | Memory-Based CF | Matrix Factorization |
|---|---|---|---|
| Dữ liệu cần | Đặc trưng sản phẩm + lịch sử | Ma trận đánh giá | Ma trận đánh giá |
| Cold-start sản phẩm | ✓ | ✗ | ✗ |
| Cold-start người dùng | ✗ | ✗ | ✗ |
| Xử lý dữ liệu thưa | Tốt | Kém | **Tốt** |
| Khả năng mở rộng | Tốt | Kém ($O(m^2)$ hoặc $O(n^2)$) | **Rất tốt** |
| Khám phá (serendipity) | Thấp | Trung bình | **Cao** |
| Diễn giải | Cao | Cao | Thấp (nhân tố ẩn) |
| Huấn luyện | Riêng từng người dùng | Không cần huấn luyện | **Batch gradient descent** |

---

# 8. Đánh giá và thực tế triển khai

## 8.1. Offline evaluation

Chia dữ liệu theo thời gian (train trên lịch sử cũ, test trên tương tác mới) hoặc ngẫu nhiên. Độ đo:

* **RMSE/MAE**: trên điểm đánh giá tường minh.
* **Precision@k, Recall@k, NDCG@k**: đối với top-k gợi ý — quan trọng hơn RMSE trong thực tế.
* **Hit Rate**: tỉ lệ trường hợp sản phẩm đúng xuất hiện trong top-k gợi ý.

## 8.2. Thách thức triển khai thực tế

* **Popularity bias**: mô hình học được thiên về sản phẩm phổ biến (nhiều điểm đánh giá) → cần re-ranking hoặc regularization theo popularity.
* **Feedback loop**: gợi ý ảnh hưởng đến hành vi người dùng → dữ liệu tương lai bị bias bởi gợi ý hiện tại → mô hình tự học trên bias của chính nó.
* **Tốc độ cập nhật**: embedding cần được cập nhật khi có dữ liệu mới; online learning hoặc re-training định kỳ.
* **Phục vụ (serving)**: tích vô hướng giữa embedding người dùng và toàn bộ catalog có thể tốn $O(n \cdot k)$ mỗi request — cần **Approximate Nearest Neighbor (ANN)** như FAISS hoặc ScaNN để phục vụ thời gian thực.

---

# 9. Từ MF đến học biểu diễn hiện đại

Matrix Factorization là nền tảng dẫn đến nhiều kỹ thuật học biểu diễn hiện đại:

* **Word2Vec** (Mikolov et al., 2013) về bản chất là phân rã ma trận đồng xuất hiện từ (PMI matrix) vào không gian thấp chiều.
* **Item2Vec**: áp dụng ý tưởng Word2Vec cho sản phẩm — chuỗi mua hàng tương tự chuỗi từ trong câu.
* **Graph-based CF** (PinSage, LightGCN): lan truyền thông tin qua đồ thị người dùng–sản phẩm để làm giàu embedding, giải quyết cold-start tốt hơn.
* **Two-tower models**: mạng nơ-ron với hai nhánh riêng biệt cho người dùng và sản phẩm, kết hợp embedding đặc trưng phong phú (ảnh, văn bản, hành vi) — nền tảng của YouTube DNN, TikTok recommendation.

---

# 10. Tổng kết

Matrix Factorization học **biểu diễn ẩn** $\mathbf{p}_u$ và $\mathbf{q}_i$ sao cho tích vô hướng xấp xỉ điểm đánh giá:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

Học bằng **SGD** (nhanh, linh hoạt) hoặc **ALS** (song song hóa tốt, dùng cho implicit data). So với memory-based CF, MF xử lý dữ liệu thưa tốt hơn và mở rộng được đến hàng triệu người dùng. Biến thể **iALS** xử lý implicit feedback; **SVD++** tích hợp thêm lịch sử ngầm định; **NCF** thay tích vô hướng bằng mạng nơ-ron.

Ba bài về hệ gợi ý — content-based, collaborative filtering, matrix factorization — hình thành tam giác bổ trợ nhau: **content-based** giải cold-start sản phẩm, **CF** khai thác hành vi tập thể, **MF** học biểu diễn ẩn để mở rộng và tổng quát hóa tốt hơn. Hệ thực tế (Netflix, Spotify, TikTok) kết hợp cả ba dưới dạng **hybrid recommendation**.

> Bài tiếp theo — **Mạng nơ-ron nhân tạo** — chuyển sang học sâu: từ perceptron đơn lớp đến mạng nhiều lớp và hàm kích hoạt phi tuyến, nền tảng của các mô hình gợi ý thế hệ mới.
