# K-láng giềng gần nhất (K-Nearest Neighbors)

> K-láng giềng gần nhất (K-Nearest Neighbors, KNN) là một trong những thuật toán **đơn giản nhất** của học máy, nhưng cũng là một trong những thuật toán **giàu trực giác nhất**: muốn biết một điểm dữ liệu thuộc lớp nào, hãy nhìn vào những hàng xóm **gần** nó nhất rồi để chúng "bỏ phiếu".
>
> KNN không xây dựng một hàm $f_\theta$ với tham số nào cả. Nó **lười** đến mức chỉ ghi nhớ toàn bộ dữ liệu huấn luyện, và mọi tính toán bị **trì hoãn** đến tận lúc dự đoán. Đây là đại diện tiêu biểu cho họ thuật toán **học dựa trên mẫu (instance-based learning)**.

---

# 1. Trực giác: "Hãy cho tôi biết bạn của bạn là ai"

Giả sử bạn dọn đến một khu phố mới và muốn đoán xem một ngôi nhà chưa rõ giá thì đáng bao nhiêu. Cách tự nhiên nhất: nhìn vào vài căn nhà **gần nhất** có diện tích, vị trí tương tự rồi lấy giá trung bình của chúng. Đó **chính xác** là tinh thần của KNN.

Giả thiết ngầm đứng sau toàn bộ thuật toán là **giả thiết trơn (smoothness assumption)**: những điểm **gần nhau** trong không gian đặc trưng thì có **nhãn giống nhau**. Nếu giả thiết này đúng, ta không cần học bất cứ quy luật toàn cục nào — chỉ cần một quy luật **cục bộ** rất thô sơ là "bắt chước hàng xóm".

Hình dung trong mặt phẳng hai chiều: mỗi điểm dữ liệu là một chấm có màu (lớp). Để phân loại một điểm mới, ta khoanh một vòng tròn quanh nó cho đến khi tóm đúng $K$ điểm gần nhất, rồi xem màu nào **chiếm đa số** trong vòng tròn đó.

---

# 2. Học lười (lazy learning): không huấn luyện, chỉ ghi nhớ

Phần lớn thuật toán học máy chia làm hai pha: **huấn luyện** (tốn kém, tìm $\theta^{*}$) rồi **dự đoán** (rẻ, chỉ tính $f_{\theta^{*}}(x)$). KNN đảo ngược điều này.

* **Pha huấn luyện** của KNN gần như **rỗng**: chỉ lưu lại tập dữ liệu $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ vào bộ nhớ. Không có gradient descent, không có hàm mất mát cần tối thiểu hóa.
* **Pha dự đoán** mới là nơi mọi công việc diễn ra: với mỗi điểm truy vấn (query) $x$, ta tính khoảng cách từ $x$ tới **toàn bộ** $N$ điểm đã lưu, chọn ra $K$ điểm gần nhất, rồi tổng hợp nhãn của chúng.

Vì tính toán bị **trì hoãn** đến lúc dự đoán, KNN được gọi là **học lười (lazy learning)**, đối lập với **học chăm (eager learning)** như hồi quy tuyến tính hay mạng nơ-ron — những mô hình "học chăm chỉ" trước rồi vứt bỏ dữ liệu huấn luyện sau khi đã có tham số.

> **Hệ quả triết lý.** KNN **không** tóm lược dữ liệu thành tham số. Toàn bộ tập huấn luyện **chính là** mô hình. Điều này khiến nó cực kỳ linh hoạt (ranh giới quyết định có thể uốn lượn tùy ý theo dữ liệu) nhưng cũng cực kỳ tốn kém khi dữ liệu lớn (xem mục 8).

---

# 3. Quy tắc dự đoán

Gọi $N_K(x)$ là tập **$K$ láng giềng gần nhất** của điểm truy vấn $x$ trong tập huấn luyện.

## 3.1. Phân loại — bỏ phiếu đa số (majority vote)

Với bài toán phân loại, nhãn dự đoán là lớp xuất hiện **nhiều nhất** trong $K$ láng giềng:

$$\hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} \mathbb{1}\big[ y_i = c \big]$$

trong đó $\mathbb{1}[\cdot]$ là hàm chỉ thị (bằng $1$ nếu điều kiện đúng, $0$ nếu sai). Đơn giản là **đếm phiếu** cho từng lớp rồi chọn lớp thắng.

Ngoài ra, KNN còn cho **ước lượng xác suất** một cách tự nhiên — tỉ lệ láng giềng thuộc lớp $c$:

$$\hat{p}(c \mid x) = \frac{1}{K} \sum_{i \in N_K(x)} \mathbb{1}\big[ y_i = c \big]$$

Để tránh **hòa phiếu** trong bài toán hai lớp, người ta thường chọn $K$ **lẻ**.

## 3.2. Hồi quy — lấy trung bình

Với bài toán hồi quy, ta thay phép bỏ phiếu bằng phép lấy **trung bình** giá trị của các láng giềng:

$$\hat{y} = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$

Cùng một ý tưởng "bắt chước hàng xóm", chỉ khác cách tổng hợp: đa số cho nhãn rời rạc, trung bình cho giá trị liên tục.

---

# 4. Các độ đo khoảng cách

"Gần" nghĩa là gì? Câu trả lời phụ thuộc vào **độ đo khoảng cách (distance metric)** ta chọn — và lựa chọn này có thể thay đổi hoàn toàn kết quả.

## 4.1. Khoảng cách Euclid (Euclidean)

Khoảng cách "đường chim bay" quen thuộc, là lựa chọn mặc định:

$$d(x, x') = \lVert x - x' \rVert_2 = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}$$

## 4.2. Khoảng cách Manhattan ($L_1$)

Đo theo kiểu "đi taxi trong thành phố ô bàn cờ" — chỉ được đi dọc các trục:

$$d(x, x') = \lVert x - x' \rVert_1 = \sum_{j=1}^{d} \lvert x_j - x'_j \rvert$$

Manhattan **ít nhạy** với điểm dị thường (outlier) hơn Euclid vì không bình phương sai lệch.

## 4.3. Khoảng cách Minkowski — dạng tổng quát

Cả Euclid và Manhattan đều là trường hợp riêng của **Minkowski** bậc $p$:

$$d(x, x') = \left( \sum_{j=1}^{d} \lvert x_j - x'_j \rvert^{p} \right)^{1/p}$$

Với $p = 1$ ta được Manhattan, $p = 2$ ta được Euclid, $p \to \infty$ ta được khoảng cách Chebyshev (lấy sai lệch lớn nhất trên một trục).

## 4.4. Khoảng cách cosine

Khi **hướng** của vector quan trọng hơn **độ lớn** (ví dụ vector đặc trưng văn bản), ta dùng độ tương đồng cosine rồi đổi sang khoảng cách:

$$d_{\cos}(x, x') = 1 - \frac{x^\top x'}{\lVert x \rVert_2 \, \lVert x' \rVert_2}$$

| Độ đo | Công thức cốt lõi | Khi nào dùng |
| --- | --- | --- |
| Euclid ($L_2$) | $\sqrt{\sum (x_j - x'_j)^2}$ | Mặc định, đặc trưng số đồng nhất |
| Manhattan ($L_1$) | $\sum \lvert x_j - x'_j \rvert$ | Nhiều chiều, có outlier |
| Minkowski | $(\sum \lvert x_j - x'_j \rvert^p)^{1/p}$ | Tổng quát, điều chỉnh qua $p$ |
| Cosine | $1 - \dfrac{x^\top x'}{\lVert x \rVert \lVert x' \rVert}$ | Văn bản, dữ liệu thưa, quan tâm hướng |

---

# 5. KNN có trọng số (weighted KNN)

KNN cơ bản đối xử với **mọi** láng giềng như nhau: một hàng xóm sát bên cũng chỉ có **một phiếu** y hệt hàng xóm xa tít ngoài rìa. Điều này hơi vô lý — hàng xóm càng gần thì càng đáng tin.

**KNN có trọng số (weighted KNN)** sửa điều đó bằng cách gán cho mỗi láng giềng một trọng số $w_i$ **giảm theo khoảng cách**, thường là nghịch đảo:

$$w_i = \frac{1}{d(x, x_i) + \varepsilon}$$

với $\varepsilon$ nhỏ để tránh chia cho $0$. Khi đó quy tắc bỏ phiếu trở thành **bỏ phiếu có trọng số**:

$$\hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} w_i \, \mathbb{1}\big[ y_i = c \big]$$

và với hồi quy là **trung bình có trọng số** $\hat{y} = \dfrac{\sum_i w_i y_i}{\sum_i w_i}$.

> Một lợi ích phụ: weighted KNN **ít nhạy** với việc chọn $K$ hơn, vì những láng giềng xa (kém liên quan) tự động bị "giảm tiếng nói" thay vì có phiếu ngang bằng.

---

# 6. Vì sao chuẩn hóa đặc trưng là bắt buộc

Đây là **cái bẫy chết người** nhất của KNN. Vì thuật toán hoàn toàn dựa trên khoảng cách, **thang đo (scale)** của các đặc trưng quyết định trực tiếp ai là "láng giềng".

Hình dung dữ liệu có hai đặc trưng: tuổi (khoảng $20$–$60$) và thu nhập (khoảng $10^7$–$10^8$ đồng). Trong khoảng cách Euclid, số hạng thu nhập **áp đảo** hoàn toàn vì đơn vị của nó lớn gấp hàng triệu lần. Đặc trưng tuổi gần như **vô hình** — KNN biến thành thuật toán chỉ xét thu nhập.

Giải pháp là **chuẩn hóa đặc trưng (feature scaling)** để đưa mọi chiều về cùng một thang, ví dụ chuẩn hóa z-score:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

hoặc co về đoạn $[0, 1]$ bằng min-max scaling. Đây là một phần cốt lõi của [kỹ thuật đặc trưng (feature-engineering)](feature-engineering) — và với KNN, **bỏ qua bước này gần như chắc chắn cho mô hình tồi**, bất kể $K$ hay độ đo bạn chọn.

---

# 7. Chọn $K$: đánh đổi giữa thiên lệch và phương sai

$K$ là **siêu tham số (hyperparameter)** quan trọng nhất, và việc chọn nó là một minh họa kinh điển cho đánh đổi **thiên lệch–phương sai (bias–variance trade-off)**.

* **$K$ nhỏ (ví dụ $K = 1$)** → mô hình **rất linh hoạt**: ranh giới quyết định bám sát từng điểm, kể cả các điểm nhiễu. Phương sai cao, **nhạy nhiễu**, dễ rơi vào [quá khớp (overfitting)](overfitting). Với $K = 1$, mọi điểm huấn luyện đều được dự đoán **đúng tuyệt đối** (chính nó là láng giềng gần nhất) — một dấu hiệu rõ ràng của overfit.
* **$K$ lớn** → mô hình **trơn hơn**: mỗi dự đoán là sự trung bình hóa của nhiều điểm, nên ổn định với nhiễu. Nhưng nếu $K$ quá lớn, ranh giới bị **làm mờ** quá mức, bỏ qua cấu trúc cục bộ thật sự — dẫn tới **dưới khớp (underfitting)**. Cực đoan: $K = N$ thì mọi truy vấn đều cho **cùng một** dự đoán (lớp đa số toàn cục).

Cách chọn $K$ trong thực tế là **kiểm định chéo (cross-validation)**: thử nhiều giá trị $K$ và chọn giá trị cho hiệu năng tốt nhất trên tập kiểm định. Một quy tắc khởi điểm thô là $K \approx \sqrt{N}$.

## 7.1. Ranh giới quyết định (decision boundary)

Ranh giới quyết định của KNN **không** là đường thẳng hay đường cong tham số mà là một tập hợp các mảnh gấp khúc do dữ liệu định ra (liên quan đến sơ đồ Voronoi với $K = 1$). $K$ càng nhỏ ranh giới càng **lởm chởm**; $K$ càng lớn ranh giới càng **mượt**. Đây chính là biểu hiện hình học của đánh đổi thiên lệch–phương sai ở trên.

---

# 8. Độ phức tạp tính toán và tăng tốc

Vì học lười, mọi chi phí dồn vào pha dự đoán. Để trả lời **một** truy vấn bằng cách quét vét cạn (brute force), ta phải tính khoảng cách tới cả $N$ điểm, mỗi khoảng cách tốn $O(d)$ phép tính:

$$O(N d)$$

Với tập dữ liệu lớn (hàng triệu điểm) và nhiều truy vấn, chi phí này nhanh chóng trở nên **không chấp nhận được**. Hơn nữa KNN còn tốn **bộ nhớ** vì phải giữ toàn bộ dữ liệu.

## 8.1. Tăng tốc bằng cấu trúc cây không gian

Thay vì quét tất cả, ta tổ chức dữ liệu thành cấu trúc cây để **loại bỏ sớm** những vùng chắc chắn không chứa láng giềng:

* **KD-tree** — chia không gian đệ quy theo từng trục tọa độ. Truy vấn trung bình còn $O(\log N)$ ở số chiều **thấp**.
* **Ball-tree** — gom điểm vào các "quả cầu" lồng nhau; hoạt động tốt hơn KD-tree khi số chiều **vừa phải**.

Khi cần nhanh hơn nữa và chấp nhận kết quả gần đúng, người ta dùng **láng giềng gần nhất xấp xỉ (Approximate Nearest Neighbors, ANN)** như HNSW hay LSH.

## 8.2. Lời nguyền số chiều (curse of dimensionality)

Có một giới hạn sâu xa hơn cả tốc độ. Khi số chiều $d$ **rất lớn**, mọi cặp điểm có xu hướng trở nên **gần như cách đều nhau**: tỉ số giữa khoảng cách xa nhất và gần nhất tiến về $1$.

$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

Khi đó khái niệm "láng giềng gần nhất" mất hết ý nghĩa — chẳng có điểm nào thực sự gần hơn điểm nào. Đây là **lời nguyền số chiều (curse of dimensionality)**, và nó khiến KNN hoạt động kém ở không gian nhiều chiều. Đồng thời, các cấu trúc cây (KD-tree, Ball-tree) cũng **suy biến** về vét cạn $O(Nd)$. Cách khắc phục là **giảm chiều** (PCA, lựa chọn đặc trưng) — lại thuộc về [kỹ thuật đặc trưng (feature-engineering)](feature-engineering).

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train)  # chuẩn hóa: bắt buộc!
knn = KNeighborsClassifier(n_neighbors=5, weights="distance",
                           metric="minkowski", p=2, algorithm="kd_tree")
knn.fit(X_train, y_train)          # "huấn luyện" = chỉ ghi nhớ dữ liệu
y_pred = knn.predict(X_test)       # mọi tính toán diễn ra ở đây
```

---

# 9. Đánh giá mô hình KNN

Vì KNN không có hàm mất mát để theo dõi khi huấn luyện, việc **đánh giá** trên tập kiểm tra là cách duy nhất biết nó tốt hay xấu. Với phân loại, ta dùng các thước đo quen thuộc — độ chính xác (accuracy), precision, recall, F1 và ma trận nhầm lẫn — trình bày kỹ trong bài [đánh giá phân loại (danh-gia-phan-loai)](danh-gia-phan-loai). Với hồi quy, ta dùng MSE/MAE.

Lưu ý quan trọng: vì KNN ghi nhớ toàn bộ dữ liệu, đánh giá trên **chính tập huấn luyện** sẽ cho kết quả **lạc quan giả tạo** (đặc biệt $K = 1$ luôn đạt $100\%$). Luôn đánh giá trên dữ liệu **chưa từng thấy**.

---

# 10. Ưu điểm

* **Đơn giản và trực quan** — không có lý thuyết phức tạp, dễ hiểu và dễ cài đặt; là baseline tuyệt vời cho bài toán mới.
* **Không cần huấn luyện** — thêm dữ liệu mới chỉ là việc nối thêm vào bộ nhớ, không phải huấn luyện lại từ đầu.
* **Phi tham số (non-parametric)** — không giả định dạng hàm của dữ liệu, nên ranh giới quyết định có thể **uốn lượn tùy ý**, bắt được quan hệ phi tuyến phức tạp.
* **Tự nhiên đa lớp** — bỏ phiếu đa số áp dụng trực tiếp cho nhiều lớp mà không cần thủ thuật như một số mô hình tuyến tính.

---

# 11. Nhược điểm

* **Dự đoán chậm và tốn bộ nhớ** — chi phí $O(Nd)$ mỗi truy vấn và phải lưu toàn bộ dữ liệu; khó dùng cho dữ liệu lớn nếu không có KD-tree/Ball-tree/ANN.
* **Cực kỳ nhạy với thang đo** — bắt buộc chuẩn hóa đặc trưng, nếu không khoảng cách bị một vài đặc trưng áp đảo (mục 6).
* **Suy yếu vì lời nguyền số chiều** — ở nhiều chiều, "gần nhất" mất ý nghĩa và mọi tăng tốc bằng cây đều thất bại.
* **Nhạy với nhiễu và mất cân bằng lớp** — điểm nhiễu trở thành láng giềng sai; lớp đa số dễ thắng phiếu một cách thiên lệch (weighted KNN giảm bớt nhưng không xóa hết).
* **Phụ thuộc lựa chọn $K$ và độ đo** — cần kiểm định chéo cẩn thận để tránh overfit ($K$ nhỏ) hay underfit ($K$ lớn).

---

# 12. Tổng kết

KNN là minh chứng đẹp rằng một ý tưởng **giản dị** — "hãy bắt chước hàng xóm gần nhất" — có thể trở thành một thuật toán hữu dụng mà không cần bất kỳ pha huấn luyện nào. Toàn bộ thuật toán quy về ba quyết định: chọn **độ đo khoảng cách**, chọn số láng giềng **$K$**, và quyết định cách **tổng hợp** phiếu (đều nhau hay có trọng số).

$$\hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} w_i \, \mathbb{1}\big[ y_i = c \big]$$

Sức mạnh của KNN là tính **phi tham số linh hoạt** và sự đơn giản; cái giá của nó là chi phí dự đoán cao, sự lệ thuộc vào chuẩn hóa đặc trưng, và sự sụp đổ trước **lời nguyền số chiều**. Hãy nhớ hai nguyên tắc vàng khi dùng KNN: **luôn chuẩn hóa đặc trưng** trước, và **luôn chọn $K$ bằng kiểm định chéo**.

> KNN cho ta một bài học quan trọng kéo dài suốt học máy: đôi khi "mô hình" tốt nhất không phải là một phương trình thông minh, mà chỉ đơn giản là **dữ liệu tốt cộng với một độ đo khoảng cách hợp lý**.
