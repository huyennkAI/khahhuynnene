# Gợi ý dựa trên nội dung (Content-Based Filtering)

> Hệ gợi ý dựa trên nội dung (content-based filtering) gợi ý cho người dùng những sản phẩm **tương tự** những gì họ đã thích trước đó, dựa trên **đặc trưng của sản phẩm** chứ không phải hành vi của người dùng khác. Nếu bạn thích một bộ phim hành động với diễn viên X, hệ thống tìm các phim khác có các đặc trưng tương tự.
>
> $$\hat{r}_{ui} = \mathbf{w}_u^\top \boldsymbol{\phi}(i) + b_u$$
>
> Mỗi người dùng $u$ có một **hồ sơ sở thích** $\mathbf{w}_u$ học từ lịch sử đánh giá của họ; $\boldsymbol{\phi}(i)$ là vector đặc trưng của sản phẩm $i$. Gợi ý = tìm sản phẩm tối đa hóa điểm dự đoán này.

---

# 1. Kiến trúc tổng quát

Một hệ gợi ý content-based gồm ba thành phần:

**1. Biểu diễn sản phẩm (Item Representation).** Mỗi sản phẩm $i$ được mô tả bởi một vector đặc trưng $\boldsymbol{\phi}(i) \in \mathbb{R}^d$. Đặc trưng có thể là:

* **Thủ công (hand-crafted)**: thể loại phim (one-hot), năm sản xuất, diễn viên, đạo diễn; giá sản phẩm, thương hiệu, danh mục.
* **Từ văn bản**: TF-IDF của mô tả sản phẩm, tiêu đề bài báo (xem [feature engineering](feature-engineering)).
* **Học được (learned)**: embedding từ mạng nơ-ron xử lý ảnh, văn bản, âm thanh.

**2. Hồ sơ người dùng (User Profile).** Mỗi người dùng $u$ có một mô hình dự đoán điểm đánh giá — đơn giản nhất là hồi quy tuyến tính với trọng số $\mathbf{w}_u$ và hệ số chệch $b_u$:

$$\hat{r}_{ui} = \mathbf{w}_u^\top \boldsymbol{\phi}(i) + b_u$$

**3. Gợi ý.** Với người dùng $u$, tính $\hat{r}_{ui}$ cho tất cả sản phẩm chưa đánh giá và trả về $k$ sản phẩm có điểm cao nhất.

---

# 2. Học hồ sơ người dùng

## 2.1. Đặt bài toán

Gọi $\mathcal{I}_u$ là tập sản phẩm người dùng $u$ đã đánh giá với điểm $r_{ui} \in \mathbb{R}$ (ví dụ $1$–$5$ sao). Ta muốn học $\mathbf{w}_u, b_u$ sao cho dự đoán $\hat{r}_{ui}$ gần với điểm thực. Đây là bài toán [hồi quy tuyến tính](linear-regression) **riêng cho từng người dùng**.

## 2.2. Tối ưu bình phương tối thiểu có điều chuẩn

Tối thiểu hàm mất mát MSE + điều chuẩn L2 ([Ridge](overfitting)):

$$\min_{\mathbf{w}_u, b_u} \sum_{i \in \mathcal{I}_u} \big(r_{ui} - \mathbf{w}_u^\top \boldsymbol{\phi}(i) - b_u\big)^2 + \lambda \lVert \mathbf{w}_u \rVert^2$$

Điều chuẩn $\lambda \lVert \mathbf{w}_u \rVert^2$ quan trọng vì mỗi người dùng thường chỉ đánh giá một vài sản phẩm — ít dữ liệu rất dễ [overfit](overfitting).

Viết dưới dạng ma trận: gọi $\mathbf{\Phi}_u \in \mathbb{R}^{|\mathcal{I}_u| \times d}$ là ma trận đặc trưng của các sản phẩm đã đánh giá, $\mathbf{r}_u$ là vector điểm tương ứng. Nghiệm đóng (sau khi gộp bias vào $\mathbf{w}_u$):

$$\mathbf{w}_u^* = (\mathbf{\Phi}_u^\top \mathbf{\Phi}_u + \lambda \mathbf{I})^{-1} \mathbf{\Phi}_u^\top \mathbf{r}_u$$

## 2.3. Hồ sơ dựa trên độ tương đồng

Thay vì học hồi quy, cách đơn giản hơn là xây hồ sơ người dùng bằng **trung bình có trọng số** đặc trưng của các sản phẩm đã tương tác:

$$\mathbf{p}_u = \frac{\sum_{i \in \mathcal{I}_u} r_{ui}\, \boldsymbol{\phi}(i)}{\sum_{i \in \mathcal{I}_u} r_{ui}}$$

Điểm dự đoán sau đó là **độ tương đồng cosine** giữa hồ sơ và sản phẩm:

$$\hat{r}_{ui} = \cos(\mathbf{p}_u, \boldsymbol{\phi}(i)) = \frac{\mathbf{p}_u^\top \boldsymbol{\phi}(i)}{\lVert \mathbf{p}_u \rVert \lVert \boldsymbol{\phi}(i) \rVert}$$

Cách này không cần tối ưu hóa nhưng kém linh hoạt hơn.

---

# 3. Ví dụ: gợi ý phim bằng TF-IDF

Giả sử mỗi phim $i$ được mô tả bằng vector TF-IDF từ tiêu đề và mô tả phim (vocab size $V$):

$$\boldsymbol{\phi}(i) \in \mathbb{R}^V$$

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Bước 1: Xây đặc trưng phim từ văn bản mô tả
descriptions = ["action hero saves the world",
                "romantic comedy love story",
                "action thriller spy mission",
                ...]
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
Phi = tfidf.fit_transform(descriptions)  # (n_items, V) - thưa

# Bước 2: Xây hồ sơ người dùng từ phim đã xem + điểm
# Giả sử user đã xem phim 0 (5★) và phim 2 (4★)
ratings = {0: 5, 2: 4}
profile = sum(r * Phi[i] for i, r in ratings.items()) / sum(ratings.values())

# Bước 3: Tính điểm tương đồng với tất cả phim
scores = cosine_similarity(profile, Phi).flatten()
top_k = scores.argsort()[::-1][:10]  # 10 phim gợi ý
```

---

# 4. Đặc trưng nội dung phong phú: hình ảnh, âm thanh

Với sản phẩm là ảnh (thời trang, bất động sản), văn bản (sách), âm nhạc:

* **Ảnh**: trích đặc trưng từ mạng nơ-ron tích chập (CNN) — lớp trước softmax cho vector $\mathbb{R}^{2048}$ (ResNet) hoặc embedding CLIP.
* **Âm nhạc**: đặc trưng phổ (MFCC, mel spectrogram), nhịp điệu, tonality; hoặc embedding từ mô hình học sâu.
* **Văn bản**: embedding câu từ BERT, Sentence-BERT cho mô tả sản phẩm.

Với các đặc trưng học sâu này, mô hình người dùng thường là một **mạng nơ-ron** thay vì hồi quy tuyến tính — nhưng cấu trúc tổng thể (đặc trưng sản phẩm → mô hình sở thích → gợi ý) vẫn giữ nguyên.

---

# 5. Ưu điểm

**Giải quyết vấn đề cold-start với sản phẩm mới.** Sản phẩm mới chưa có ai đánh giá vẫn được gợi ý ngay nếu có đặc trưng. Đây là lợi thế lớn so với [lọc cộng tác](collaborative-filtering) — vốn cần lịch sử tương tác.

**Không cần thông tin người dùng khác.** Hồ sơ mỗi người dùng học độc lập từ lịch sử của chính họ; không lo vấn đề quyền riêng tư khi chia sẻ dữ liệu giữa người dùng.

**Diễn giải được.** Ta biết tại sao gợi ý một sản phẩm: "vì bạn thích phim hành động của đạo diễn X, đây là phim hành động khác của cùng đạo diễn". Điều này rất có giá trị cho UX và trust.

**Ổn định với người dùng thiểu số (niche users).** Người có sở thích ít phổ biến vẫn nhận được gợi ý tốt nếu sở thích đó phản ánh rõ trong đặc trưng sản phẩm.

---

# 6. Nhược điểm

**Giới hạn trong vùng đặc trưng quan sát được.** Mô hình chỉ gợi ý sản phẩm **tương tự** những gì người dùng đã xem — không có khả năng **khám phá (serendipity)**. Người thích phim hành động mãi chỉ nhận được phim hành động, không bao giờ khám phá ra mình thích phim tài liệu.

**Cold-start với người dùng mới.** Người dùng mới chưa có lịch sử tương tác thì không học được hồ sơ $\mathbf{w}_u$. Phải dùng giải pháp khởi động: hỏi một số câu hỏi ban đầu, dùng hồ sơ mặc định, hoặc kết hợp với lọc cộng tác.

**Chất lượng phụ thuộc hoàn toàn vào đặc trưng.** Nếu đặc trưng sản phẩm không phản ánh được thứ người dùng thực sự quan tâm (ví dụ không bắt được "cảm xúc" hay "phong cách"), mô hình sẽ kém. "Garbage in, garbage out" — nguyên lý từ [feature engineering](feature-engineering) đặc biệt đúng ở đây.

**Mỗi người dùng một mô hình riêng.** $n$ người dùng cần $n$ hồi quy riêng biệt; khi số lượng người dùng lớn, chi phí bộ nhớ và tính toán tăng tuyến tính.

---

# 7. So sánh với Lọc cộng tác

| | Content-Based | Collaborative Filtering |
|---|---|---|
| Dựa trên | Đặc trưng sản phẩm | Hành vi người dùng tương tự |
| Cold-start sản phẩm mới | ✓ Giải quyết được | ✗ Không có dữ liệu |
| Cold-start người dùng mới | ✗ Cần lịch sử | ✗ Cần lịch sử |
| Khả năng khám phá | Thấp (trong vùng đặc trưng) | Cao (qua người dùng tương tự) |
| Cần kỹ thuật đặc trưng | Cao | Thấp |
| Diễn giải | Dễ | Khó hơn |

Hai cách tiếp cận bổ trợ nhau; hệ thực tế thường kết hợp cả hai (**hybrid recommendation**).

---

# 8. Tổng kết

Content-based filtering xây một **mô hình sở thích riêng cho từng người dùng** bằng cách học quan hệ giữa đặc trưng sản phẩm và điểm đánh giá của người đó:

$$\hat{r}_{ui} = \mathbf{w}_u^\top \boldsymbol{\phi}(i) + b_u$$

Học $\mathbf{w}_u$ bằng hồi quy Ridge trên lịch sử của từng người, hoặc đơn giản hơn bằng trung bình đặc trưng có trọng số. Ưu điểm chính là **giải quyết cold-start sản phẩm mới** và **tính diễn giải**; nhược điểm chính là **giới hạn khám phá** và phụ thuộc chất lượng đặc trưng.

> Bài tiếp theo — **Lọc cộng tác** — bỏ qua nội dung sản phẩm, thay vào đó tận dụng sự tương đồng trong hành vi của người dùng: "người thích những thứ bạn thích cũng thích gì?"
