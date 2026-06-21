# Phân cụm K-means

> K-means là thuật toán phân cụm (clustering) **không giám sát (unsupervised)** đơn giản nhất và phổ biến nhất: chia $N$ điểm dữ liệu thành $K$ cụm sao cho mỗi điểm thuộc về cụm có **tâm gần nó nhất**, và tâm của mỗi cụm là **trung bình** của các điểm trong cụm đó.
>
> $$\min_{\{C_k\}, \{\mu_k\}} \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2$$
>
> Đây là bài toán **gán nhãn + tìm tâm** lặp đi lặp lại đến khi hội tụ — gọi là **thuật toán Lloyd**. Đơn giản đến mức cài đặt chỉ vài dòng, nhưng ẩn trong đó là những vấn đề sâu sắc về hội tụ, lựa chọn $K$ và lời nguyền số chiều.

---

# 1. Bài toán phân cụm và trực giác

Phân cụm là bài toán **khám phá cấu trúc ẩn** trong dữ liệu khi không có nhãn: nhóm những điểm **tương tự nhau** lại với nhau, và tách xa những điểm **khác nhau**. Đây là nhiệm vụ cốt lõi của [học không giám sát (unsupervised learning)](hoc-may-co-ban).

K-means cụ thể hóa "tương tự" bằng **khoảng cách Euclid đến tâm cụm**: mỗi cụm được đại diện bởi một điểm tâm (centroid) $\boldsymbol{\mu}_k$, và mỗi điểm được gán cho cụm có tâm gần nó nhất. Mục tiêu là tối thiểu hóa tổng **bình phương khoảng cách** từ mỗi điểm đến tâm cụm của nó — gọi là **tổng bình phương nội cụm (within-cluster sum of squares, WCSS)**:

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2$$

---

# 2. Thuật toán Lloyd: gán — cập nhật — lặp

Bài toán tối ưu hóa $J$ đồng thời theo cả gán nhãn $\{C_k\}$ và tâm $\{\boldsymbol{\mu}_k\}$ là **NP-khó** nói chung. Thuật toán Lloyd giải gần đúng bằng cách **xen kẽ** tối ưu theo từng nhóm biến, cố định nhóm kia.

## 2.1. Khởi tạo

Chọn $K$ tâm ban đầu $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$. Cách đơn giản nhất: chọn ngẫu nhiên $K$ điểm từ dữ liệu. Cách tốt hơn: K-means++ (mục 5).

## 2.2. Bước gán (Assignment step)

Cố định các tâm $\boldsymbol{\mu}_k$. Gán mỗi điểm $\mathbf{x}_i$ về cụm có tâm gần nhất:

$$z_i = \arg\min_{k} \lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2$$

**Đây là tối ưu $J$ theo $\{C_k\}$ khi cố định $\{\boldsymbol{\mu}_k\}**. Với mỗi điểm độc lập, gán về tâm gần nhất rõ ràng giảm đóng góp của nó vào $J$ (hoặc giữ nguyên).

## 2.3. Bước cập nhật (Update step)

Cố định phân cụm $\{C_k\}$. Tính lại tâm mỗi cụm là trung bình các điểm trong cụm đó:

$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}$$

**Đây là tối ưu $J$ theo $\{\boldsymbol{\mu}_k\}$ khi cố định $\{C_k\}$**. Với $C_k$ cố định, $J$ là tổng bình phương khoảng cách từ các điểm trong $C_k$ tới $\boldsymbol{\mu}_k$ — cực tiểu theo $\boldsymbol{\mu}_k$ đạt tại trung bình.

**Chứng minh.** Lấy đạo hàm số hạng liên quan tới $\boldsymbol{\mu}_k$ và cho bằng $\mathbf{0}$:

$$\frac{\partial}{\partial \boldsymbol{\mu}_k} \sum_{\mathbf{x} \in C_k} \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2 = -2 \sum_{\mathbf{x} \in C_k} (\mathbf{x} - \boldsymbol{\mu}_k) = \mathbf{0}$$

$$\Rightarrow \boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x} \qquad \blacksquare$$

## 2.4. Lặp đến hội tụ

Xen kẽ bước Gán và bước Cập nhật cho đến khi không có điểm nào thay đổi cụm — tức phân cụm **ổn định**.

```python
import numpy as np

def kmeans(X, K, max_iter=300, tol=1e-4):
    N, D = X.shape
    # Khởi tạo ngẫu nhiên
    idx = np.random.choice(N, K, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        # Bước gán
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # (N, K)
        labels = dists.argmin(axis=1)

        # Bước cập nhật
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids
```

---

# 3. Hội tụ

## 3.1. Tại sao thuật toán luôn hội tụ?

Mỗi bước đều **giảm hoặc giữ nguyên** $J$:

* Bước Gán không tăng $J$ (mỗi điểm về tâm gần nhất thay vì tâm hiện tại).
* Bước Cập nhật không tăng $J$ (trung bình là nghiệm cực tiểu phương sai).

Vì $J \ge 0$ và là một dãy không tăng bị chặn dưới, nó **hội tụ**. Vì số phân cụm hữu hạn ($K^N$ cách phân) và mỗi lần lặp phân cụm phải thay đổi (nếu không đã dừng), thuật toán **kết thúc trong hữu hạn bước**.

## 3.2. Hội tụ về cực tiểu địa phương, không toàn cục

Đây là điểm yếu cốt lõi: bài toán $\min J$ **không lồi (non-convex)** theo cả hai nhóm biến đồng thời. Lloyd chỉ đảm bảo hội tụ về **cực tiểu địa phương (local minimum)** hoặc điểm yên ngựa, phụ thuộc mạnh vào khởi tạo. Nghiệm toàn cục không có bảo đảm.

Hệ quả thực hành: nên chạy K-means **nhiều lần** với khởi tạo khác nhau rồi giữ nghiệm có $J$ nhỏ nhất:

```python
best_J, best_labels = np.inf, None
for _ in range(10):
    labels, centroids = kmeans(X, K)
    J = sum(np.linalg.norm(X[labels == k] - centroids[k], axis=1).sum()**2
            for k in range(K))
    if J < best_J:
        best_J, best_labels = J, labels
```

---

# 4. Diễn giải K-means như bài toán EM

K-means có một lý giải xác suất đẹp: nó là một trường hợp đặc biệt của **thuật toán Expectation-Maximization (EM)** cho mô hình hỗn hợp Gauss (Gaussian Mixture Model, GMM) khi **phương sai của mọi thành phần tiến về $0$** và **gán nhãn cứng** (hard assignment).

Trong GMM đầy đủ, bước E tính **trách nhiệm mềm** $r_{ik} = P(z_i = k \mid \mathbf{x}_i)$ (điểm $i$ thuộc cụm $k$ bao nhiêu phần trăm). Khi phương sai $\sigma^2 \to 0$, trách nhiệm mềm sụp đổ về **gán cứng**: $r_{ik} \to \mathbb{1}[k = \arg\min_j d(\mathbf{x}_i, \boldsymbol{\mu}_j)]$ — đúng bước Gán của K-means. Bước M cập nhật tâm bằng trung bình có trọng số, với gán cứng trở thành trung bình thường — đúng bước Cập nhật.

> **Ý nghĩa.** K-means là GMM "cứng nhắc" và rẻ tính toán. GMM mềm hơn, cho phép một điểm thuộc một phần về nhiều cụm, nhưng tốn kém hơn và cần ước lượng thêm ma trận hiệp phương sai.

---

# 5. K-means++: khởi tạo thông minh

Khởi tạo ngẫu nhiên dễ dẫn tới nghiệm xấu. **K-means++** (Arthur & Vassilvitskii, 2007) khởi tạo các tâm sao cho chúng **cách xa nhau** theo xác suất tỉ lệ với khoảng cách bình phương tới tâm gần nhất đã chọn:

1. Chọn tâm đầu tiên $\boldsymbol{\mu}_1$ ngẫu nhiên đều.
2. Với mỗi tâm tiếp theo, chọn điểm $\mathbf{x}$ với xác suất $\propto d(\mathbf{x})^2$, trong đó $d(\mathbf{x}) = \min_k \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert$.
3. Lặp đến khi chọn đủ $K$ tâm.

**Kết quả lý thuyết.** K-means++ bảo đảm $\mathbb{E}[J] \le O(\log K) \cdot J^*$, trong đó $J^*$ là giá trị tối ưu toàn cục. Trong thực tế nó cho nghiệm tốt hơn nhiều và là mặc định trong hầu hết thư viện (sklearn dùng K-means++ theo mặc định).

---

# 6. Chọn K: phương pháp Elbow và Silhouette

$K$ là **siêu tham số (hyperparameter)** quan trọng nhất và không có câu trả lời hoàn toàn khách quan — phụ thuộc cấu trúc dữ liệu và mục đích.

## 6.1. Phương pháp Elbow (khuỷu tay)

Vẽ $J(K)$ theo $K$. $J$ luôn giảm khi $K$ tăng (nhiều cụm hơn thì mỗi cụm nhỏ hơn, điểm gần tâm hơn). Nhưng mức giảm thường có một **điểm gãy (elbow)**: trước điểm đó $J$ giảm mạnh (mỗi cụm mới đóng góp lớn), sau đó $J$ giảm chậm (cụm thêm vào chỉ cắt nhỏ cụm đã tốt). Điểm gãy gợi ý $K$ hợp lý.

Nhược điểm: điểm gãy thường mơ hồ, không rõ ràng với dữ liệu thực.

## 6.2. Hệ số Silhouette

**Silhouette score** của một điểm $\mathbf{x}_i$ đo mức nó "thuộc về" cụm của mình so với cụm kề bên. Gọi $a_i$ là khoảng cách trung bình từ $\mathbf{x}_i$ tới các điểm **trong cùng cụm**, và $b_i$ là khoảng cách trung bình tới cụm **gần nhất khác**:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$$

$s_i \approx 1$: điểm nằm gọn trong cụm và xa cụm khác — **tốt**. $s_i \approx 0$: điểm gần ranh giới hai cụm. $s_i < 0$: điểm có lẽ bị gán nhầm cụm.

Trung bình Silhouette trên toàn bộ điểm cho một con số tóm tắt chất lượng phân cụm cho giá trị $K$ đó. Chọn $K$ tối đa hóa Silhouette trung bình.

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

scores = []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, init='k-means++', n_init=10).fit_predict(X)
    scores.append(silhouette_score(X, labels))
best_K = range(2, 11)[scores.index(max(scores))]
```

---

# 7. Biến thể và mở rộng

## 7.1. K-medoids (PAM)

Thay tâm là trung bình bằng **medoid** — một điểm dữ liệu **thực sự** trong tập, chọn sao cho tổng khoảng cách tới các điểm trong cụm là nhỏ nhất. Lợi thế: bền với outlier hơn K-means (trung bình bị một điểm cực lớn kéo lệch mạnh, medoid thì không); nhận mọi độ đo khoảng cách, không chỉ Euclid. Nhược điểm: tốn $O(N^2)$ mỗi bước cập nhật.

## 7.2. Mini-batch K-means

Thay vì dùng toàn bộ dữ liệu mỗi bước, dùng một **mini-batch ngẫu nhiên** để cập nhật tâm. Gần như nhanh bằng K-means thường trên dữ liệu nhỏ, nhưng mở rộng được cho dữ liệu lớn (hàng triệu điểm) mà K-means đầy đủ quá chậm.

## 7.3. Fuzzy C-means

Cho phép **gán mềm** như GMM: mỗi điểm có độ thuộc về (membership degree) $u_{ik} \in [0,1]$ cho mỗi cụm, với $\sum_k u_{ik} = 1$. Hàm mục tiêu phạt các gán mềm theo một tham số làm mờ $m > 1$. Với $m \to 1$ thu lại K-means gán cứng.

---

# 8. Hạn chế của K-means

## 8.1. Giả định hình cầu (spherical clusters)

K-means tối thiểu khoảng cách Euclid tới tâm, nên về bản chất nó giả định các cụm có dạng **hình cầu và kích thước tương đương**. Với cụm hình elipse, hình cong hay mật độ không đều, K-means cho kết quả tệ.

Ví dụ kinh điển: dữ liệu hai vòng đồng tâm (two moons, two circles) — K-means với $K=2$ không thể tách được vì ranh giới tự nhiên không phải đường thẳng Voronoi.

## 8.2. Nhạy với outlier

Vì tâm là **trung bình**, một điểm ngoại lai lớn kéo tâm ra xa. K-medoids ổn định hơn.

## 8.3. Phải biết $K$ trước

Không có cách tự động chọn $K$ hoàn toàn tin cậy. Với một số bài toán, số cụm tự nhiên không rõ ràng.

## 8.4. Lời nguyền số chiều

Như [KNN](knn), khoảng cách Euclid mất ý nghĩa ở nhiều chiều — các điểm có xu hướng cách đều nhau. Cần giảm chiều (PCA, t-SNE) trước khi phân cụm dữ liệu nhiều chiều.

---

# 9. Ưu điểm

* **Đơn giản và nhanh** — mỗi vòng lặp tốn $O(NKD)$, hội tụ thường sau vài chục vòng, rất khả thi cho dữ liệu lớn.
* **Dễ hiểu và diễn giải** — tâm cụm có ý nghĩa trực quan; dễ trực quan hóa và giải thích cho người không chuyên.
* **Hoạt động tốt với cụm hình cầu** — khi dữ liệu có cấu trúc cụm rõ ràng và tương đối đẳng hướng, K-means cho kết quả xuất sắc.
* **Baseline phân cụm tiêu chuẩn** — luôn là điểm khởi đầu trước khi thử các phương pháp phức tạp hơn (DBSCAN, GMM, phân cụm phổ).

---

# 10. Tổng kết

K-means là thuật toán phân cụm nền tảng nhất, tối thiểu hóa tổng bình phương khoảng cách nội cụm:

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2$$

bằng thuật toán Lloyd **Gán → Cập nhật → Lặp**. Mỗi bước giảm $J$, nên hội tụ được đảm bảo — nhưng chỉ về **cực tiểu địa phương**. Ba bài học thực hành quan trọng: dùng **K-means++** để khởi tạo tốt, chạy **nhiều lần** để tránh kẹt cục bộ, và dùng **Silhouette hoặc Elbow** để chọn $K$.

K-means giả định cụm **hình cầu và kích thước đồng đều** — khi vi phạm, hãy thử GMM (cụm elipse), DBSCAN (cụm hình dạng tùy ý theo mật độ) hay phân cụm phổ (cụm phi lồi).

> Bài tiếp theo — **PCA** — giảm chiều dữ liệu bằng cách giữ lại các hướng có **phương sai lớn nhất**, thường được dùng làm bước tiền xử lý trước K-means khi dữ liệu nhiều chiều.
