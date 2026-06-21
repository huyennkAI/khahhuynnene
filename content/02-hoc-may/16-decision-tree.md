# Cây quyết định (Decision Tree)

> Cây quyết định (decision tree) mô phỏng đúng cách con người ra quyết định: đặt **một chuỗi câu hỏi** dạng "nếu... thì...". Mỗi câu hỏi chia dữ liệu thành các nhóm nhỏ hơn, cho tới khi mỗi nhóm đủ **thuần nhất (pure)** để đưa ra một dự đoán.
>
> $$\text{Toàn bộ dữ liệu} \;\xrightarrow{\;\text{thuộc tính } x_j ?\;}\; \{\text{vùng}_1, \text{vùng}_2, \dots\} \;\xrightarrow{\;\cdots\;}\; \text{các lá thuần nhất}$$
>
> Điểm hấp dẫn nhất của cây: nó là một trong số ít mô hình **giải thích được (interpretable)** — ta đọc thẳng được logic dự đoán mà không cần một công thức trừu tượng nào.

---

# 1. Cấu trúc của một cây

Một cây quyết định gồm ba loại thành phần:

* **Nút gốc (root node)** — chứa toàn bộ tập huấn luyện, nơi câu hỏi đầu tiên được đặt ra.
* **Nút trong (internal node)** — mỗi nút trong ứng với **một câu hỏi trên một thuộc tính (feature)**, ví dụ "Tuổi $> 30$?" hay "Màu = đỏ?". Câu trả lời quyết định ta đi xuống nhánh con nào.
* **Nút lá (leaf node)** — không hỏi gì thêm, chỉ đưa ra **dự đoán**: một nhãn lớp (phân loại) hoặc một giá trị số (hồi quy).

Dự đoán cho một điểm dữ liệu $x$ là một hành trình: bắt đầu từ gốc, tại mỗi nút trong trả lời câu hỏi rồi rẽ nhánh, cứ thế cho tới khi chạm một lá.

Một cây với độ sâu (depth) $d$ thực ra mã hóa một hàm dự đoán bằng cách **chia không gian đặc trưng thành các vùng hình hộp (axis-aligned regions)**. Mỗi câu hỏi "$x_j > t$?" là một nhát cắt vuông góc với trục $x_j$, và mỗi lá tương ứng với một vùng. Bên trong một vùng, mô hình trả về một hằng số duy nhất.

---

# 2. Trực giác: chia để trị

Mục tiêu của việc xây cây là làm cho mỗi vùng (mỗi lá) càng **thuần nhất** càng tốt — nghĩa là các điểm trong cùng một vùng nên có **cùng nhãn**. Nếu một lá chỉ chứa các điểm thuộc lớp "Có", ta tự tin dự đoán "Có" cho mọi điểm rơi vào đó.

Câu hỏi cốt lõi: **chọn thuộc tính nào và ngưỡng nào để hỏi tại mỗi nút?** Một câu hỏi tốt là câu hỏi sau khi chia, các nhóm con trở nên **thuần hơn** so với nhóm cha. Ta cần một thước đo định lượng cho "độ hỗn loạn" — và đó chính là **entropy**.

> Tư tưởng tham lam (greedy): tại mỗi nút, chọn câu hỏi làm **giảm độ hỗn loạn nhiều nhất ngay lập tức**, rồi đệ quy xuống các nhánh con. Đây không bảo đảm cây tối ưu toàn cục (bài toán đó là NP-khó), nhưng cho kết quả tốt với chi phí thấp.

---

# 3. Entropy: đo độ hỗn loạn

Mượn từ lý thuyết thông tin (information theory), **entropy** đo mức độ "bất định" của một tập. Với tập $S$ có các lớp $c$ chiếm tỉ lệ $p_c$:

$$H(S) = -\sum_{c} p_c \log_2 p_c$$

Trực giác qua hai cực trị:

* **Thuần nhất tuyệt đối** — toàn bộ một lớp, $p_c = 1$. Khi đó $H(S) = -1 \cdot \log_2 1 = 0$: không còn bất định nào, không cần thông tin để mô tả.
* **Hỗn loạn tối đa** — hai lớp chia đôi, $p_c = \tfrac12$. Khi đó $H(S) = -\tfrac12\log_2\tfrac12 - \tfrac12\log_2\tfrac12 = 1$ bit: cần đúng một bit để nói điểm thuộc lớp nào.

Đơn vị là **bit** (vì dùng $\log_2$): entropy chính là số bit trung bình tối thiểu để mã hóa nhãn của một điểm lấy ngẫu nhiên từ $S$. Tập càng lẫn lộn nhiều lớp với tỉ lệ cân bằng thì entropy càng cao.

---

# 4. Information gain: đo độ lợi của một câu hỏi

Giả sử ta chia $S$ theo thuộc tính $A$ thành các tập con $\{S_v\}$ ứng với mỗi giá trị $v$ của $A$. Entropy **sau khi chia** là trung bình có trọng số (theo kích thước) entropy của các nhánh con:

$$H(S \mid A) = \sum_{v} \frac{|S_v|}{|S|}\, H(S_v)$$

**Độ lợi thông tin (information gain)** là phần entropy giảm đi nhờ câu hỏi:

$$IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|}\, H(S_v)$$

Số hạng đầu là độ hỗn loạn của nút cha; số hạng sau là độ hỗn loạn còn lại của các con. Hiệu của chúng đo **lượng thông tin câu hỏi $A$ mang lại** — chia càng làm các con thuần thì $IG$ càng lớn. Đây chính là hàm điểm để xếp hạng các câu hỏi ứng viên.

> Một cảnh báo về thiên lệch: $IG$ ưu ái các thuộc tính có **nhiều giá trị** (ví dụ "mã khách hàng"), vì chia nhỏ vụn luôn cho entropy thấp giả tạo. Để khắc phục, thuật toán C4.5 dùng **tỉ lệ lợi (gain ratio)**, chuẩn hóa $IG$ bằng entropy của chính phép chia.

---

# 5. Thuật toán ID3

ID3 (Iterative Dichotomiser 3, Quinlan) xây cây phân loại theo lối **tham lam, đệ quy**:

1. Tính entropy $H(S)$ của nút hiện tại.
2. Với **mỗi** thuộc tính ứng viên $A$, tính $IG(S, A)$.
3. Chọn thuộc tính $A^{*} = \arg\max_A IG(S, A)$ làm câu hỏi cho nút.
4. Tạo một nhánh con cho mỗi giá trị của $A^{*}$, rồi **đệ quy** trên từng tập con $S_v$.
5. **Dừng** khi: tập con đã thuần (entropy $= 0$), hết thuộc tính để hỏi, hoặc tập quá nhỏ. Tạo lá gán nhãn theo lớp đa số.

```python
def id3(S, features):
    if pure(S) or not features:
        return Leaf(majority_class(S))
    A = max(features, key=lambda a: info_gain(S, a))
    node = Node(question=A)
    for v, S_v in split_by(S, A):
        node.children[v] = id3(S_v, features - {A})
    return node
```

ID3 nguyên bản chỉ xử lý thuộc tính **rời rạc** và không cắt tỉa — đó là lý do nó dễ overfit (mục 8). Các phiên bản kế thừa (C4.5, CART) bổ sung xử lý liên tục, giá trị thiếu và cắt tỉa.

---

# 6. Chỉ số Gini và CART

CART (Classification and Regression Trees, Breiman và cộng sự) thay entropy bằng **chỉ số Gini (Gini impurity)** — xác suất phân loại sai nếu ta gán nhãn ngẫu nhiên theo phân phối của tập:

$$G(S) = 1 - \sum_{c} p_c^{2}$$

Gini cũng đạt $0$ khi thuần và cực đại khi cân bằng, rất giống entropy nhưng **rẻ hơn** vì không cần tính logarit. Trên thực tế hai tiêu chí cho cây gần như nhau; sự khác biệt thường nhỏ.

| Tiêu chí | Công thức độ vẩn đục | Đặc điểm |
| --- | --- | --- |
| Entropy (ID3/C4.5) | $-\sum_c p_c \log_2 p_c$ | Nền tảng lý thuyết thông tin, có $\log$ |
| Gini (CART) | $1 - \sum_c p_c^2$ | Tính nhanh, không cần $\log$ |
| Phương sai/MSE | $\frac{1}{|S|}\sum_i (y_i - \bar{y})^2$ | Dùng cho cây hồi quy |

Khác ID3, CART luôn tạo cây **nhị phân** (mỗi nút chia đúng hai nhánh), kể cả với thuộc tính nhiều giá trị.

---

# 7. Thuộc tính liên tục và cây hồi quy

## 7.1. Xử lý thuộc tính liên tục

Với thuộc tính số (ví dụ "thu nhập"), không thể tạo một nhánh cho mỗi giá trị. Thay vào đó, cây tìm một **ngưỡng chia (threshold)** $t$ và đặt câu hỏi nhị phân "$x_j \le t$?". Cách tìm $t$ tốt nhất:

* Sắp xếp các giá trị quan sát của $x_j$.
* Xét các ngưỡng là **trung điểm** giữa hai giá trị liên tiếp.
* Tính $IG$ (hoặc giảm Gini) cho mỗi ngưỡng, chọn ngưỡng cho điểm cao nhất.

Một thuộc tính liên tục có thể được hỏi **nhiều lần** trên các nhánh khác nhau với các ngưỡng khác nhau — đó là cách cây xấp xỉ ranh giới cong bằng nhiều nhát cắt vuông góc trục.

## 7.2. Cây hồi quy (regression tree)

Khi nhãn là số thực, "độ hỗn loạn" không còn là entropy mà là **phương sai (variance)** trong nút. Dự đoán tại một lá là **trung bình** các nhãn rơi vào lá đó, $\bar{y}_S = \frac{1}{|S|}\sum_{i \in S} y_i$. Tiêu chí chia là **giảm sai số bình phương trung bình (MSE)**:

$$\text{Cost}(S, A) = \sum_{v} \frac{|S_v|}{|S|} \cdot \frac{1}{|S_v|}\sum_{i \in S_v}\big(y_i - \bar{y}_{S_v}\big)^2$$

Ta chọn câu hỏi làm **giảm tổng phương sai** nhiều nhất — hoàn toàn song song với việc giảm entropy ở phân loại. Cây hồi quy do đó tạo ra một hàm **bậc thang (piecewise-constant)**: hằng số trong mỗi vùng, nhảy bậc tại các ranh giới.

---

# 8. Overfitting và cắt tỉa (pruning)

Một cây đủ sâu có thể **chia tới khi mỗi lá chỉ còn một điểm** — lúc đó entropy huấn luyện bằng $0$ và độ chính xác trên tập huấn luyện đạt $100\%$. Nhưng đây chính là biểu hiện kinh điển của [overfitting](/02-hoc-may/overfitting): cây học thuộc cả nhiễu, các lá nhỏ vụn chỉ phản ánh ngẫu nhiên của dữ liệu huấn luyện chứ không phải quy luật chung, nên dự đoán trên dữ liệu mới rất kém. Cây càng sâu thì phương sai (variance) càng cao.

Hai chiến lược kiểm soát độ phức tạp:

* **Cắt tỉa trước (pre-pruning / early stopping).** Dừng chia sớm bằng các điều kiện: giới hạn độ sâu tối đa (`max_depth`), số mẫu tối thiểu để chia một nút (`min_samples_split`), hoặc yêu cầu $IG$ phải vượt một ngưỡng. Rẻ nhưng có thể dừng quá sớm (bỏ lỡ một phép chia tốt nằm sau một phép chia tầm thường).
* **Cắt tỉa sau (post-pruning).** Cứ để cây mọc đầy rồi **cắt ngược từ lá lên**: gộp các nhánh con thành một lá nếu việc đó không làm giảm hiệu năng trên tập kiểm định (validation set). CART dùng **cắt tỉa theo độ phức tạp chi phí (cost-complexity pruning)**, cực tiểu hóa $\text{Lỗi} + \alpha \cdot (\text{số lá})$, với $\alpha$ phạt cây lớn.

> Nhìn rộng hơn, độ sâu của cây là một nút điều chỉnh đánh đổi thiên lệch–phương sai (bias–variance trade-off): cây nông thì thiên lệch cao (underfit), cây sâu thì phương sai cao (overfit). Cắt tỉa kéo nó về vùng cân bằng. Việc đánh giá nên dùng các chỉ số phân loại đầy đủ thay vì chỉ độ chính xác (xem [đánh giá phân loại](/02-hoc-may/danh-gia-phan-loai)).

---

# 9. Tiến tới ensemble

Nhược điểm cố hữu của một cây đơn là **bất ổn định (instability)**: thay đổi nhỏ trong dữ liệu có thể đổi hẳn cấu trúc cây ở các nút cao, kéo theo phương sai lớn. Các phương pháp **tập hợp (ensemble)** trị đúng bệnh này:

* **Random Forest** — huấn luyện nhiều cây trên các tập con bootstrap và tập con thuộc tính ngẫu nhiên, rồi lấy **trung bình/bỏ phiếu**. Trung bình hóa nhiều cây ít tương quan **giảm phương sai** mạnh mà gần như không tăng thiên lệch.
* **Gradient Boosting** — xây cây **tuần tự**, mỗi cây mới sửa phần dư (residual) của tổ hợp trước đó; chủ yếu **giảm thiên lệch**, cho độ chính xác rất cao (XGBoost, LightGBM).

Cái giá phải trả: ensemble đánh mất tính giải thích trực tiếp của một cây đơn lẻ.

---

# 10. Ưu điểm

* **Dễ diễn giải (interpretable)** — đọc thẳng được chuỗi câu hỏi dẫn tới mỗi dự đoán; có thể vẽ ra và trình bày cho người không chuyên.
* **Không cần chuẩn hóa (no scaling)** — vì chỉ so sánh ngưỡng trên từng thuộc tính riêng lẻ, cây miễn nhiễm với thang đo và đơn vị, không cần chuẩn hóa hay co giãn dữ liệu.
* **Xử lý hỗn hợp dữ liệu** — dùng được cả thuộc tính rời rạc lẫn liên tục, ít nhạy với ngoại lai (outlier).
* **Tự động chọn đặc trưng** — thuộc tính ít thông tin đơn giản không bao giờ được chọn để chia.

---

# 11. Nhược điểm

* **Dễ overfit** — từ mục 8, cây sâu học thuộc nhiễu; bắt buộc phải cắt tỉa hoặc giới hạn độ sâu.
* **Bất ổn định (high variance)** — nhạy với nhiễu dữ liệu; một mẫu khác đi có thể cho cây hoàn toàn khác.
* **Ranh giới vuông góc trục** — chỉ cắt song song các trục, nên xấp xỉ vụng về các ranh giới chéo hoặc cong (cần nhiều bậc thang).
* **Tham lam, không tối ưu toàn cục** — chọn câu hỏi tốt nhất tại mỗi bước không bảo đảm cây tốt nhất; có thể bỏ lỡ cấu trúc đẹp ẩn sau một phép chia tầm thường.

---

# 12. Tổng kết

Cây quyết định biến bài toán dự đoán thành một chuỗi câu hỏi đơn giản, chia không gian đặc trưng thành các vùng thuần nhất. Toàn bộ thuật toán xoay quanh một thước đo độ hỗn loạn và việc cực đại hóa độ giảm của nó tại mỗi nút:

$$IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|}\, H(S_v), \qquad H(S) = -\sum_{c} p_c \log_2 p_c$$

ID3 chọn thuộc tính có $IG$ lớn nhất rồi đệ quy; CART thay entropy bằng Gini (phân loại) hoặc giảm phương sai (hồi quy) và sinh cây nhị phân. Sức mạnh lớn nhất của cây là **tính giải thích được**; điểm yếu lớn nhất là **dễ overfit và bất ổn định** — chữa bằng cắt tỉa, và triệt để hơn là bằng ensemble.

> Bài tiếp theo — **Random Forest & Gradient Boosting** — gộp nhiều cây thành một bộ dự đoán mạnh: rừng ngẫu nhiên trung bình hóa để **giảm phương sai**, còn boosting xây cây tuần tự để **giảm thiên lệch**, đổi tính giải thích của cây đơn lấy độ chính xác vượt trội.
