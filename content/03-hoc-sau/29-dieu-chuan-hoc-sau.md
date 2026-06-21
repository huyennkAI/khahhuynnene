# Tổng quát hóa & Điều chuẩn trong học sâu

> Một mạng nơ-ron sâu có thể có hàng triệu, thậm chí hàng tỉ tham số — đủ sức **ghi nhớ thuộc lòng** toàn bộ tập huấn luyện. Vậy mà khi triển khai, nó vẫn dự đoán tốt trên dữ liệu chưa từng thấy. **Điều chuẩn (regularization)** là tập hợp các kỹ thuật giúp ta nghiêng cán cân về phía tổng quát hóa thay vì học vẹt.

---

# 1. Nghịch lý tổng quát hóa của mạng sâu

Trong học máy cổ điển, có một niềm tin gần như định luật: mô hình càng nhiều tham số thì càng dễ **quá khớp (overfitting)** — học cả nhiễu trong dữ liệu huấn luyện, rồi thất bại trên dữ liệu mới. Quan điểm đánh đổi độ chệch–phương sai (bias–variance tradeoff) nói rằng độ phức tạp tăng làm phương sai tăng vọt, kéo lỗi kiểm tra đi lên.

Mạng nơ-ron sâu phá vỡ trực giác này một cách ngoạn mục:

> Một mạng có số tham số **nhiều hơn cả số mẫu huấn luyện** vẫn thường tổng quát hóa rất tốt. Thí nghiệm kinh điển (Zhang et al., 2017) cho thấy mạng có thể khớp hoàn hảo cả những **nhãn ngẫu nhiên** — bằng chứng nó dư sức ghi nhớ — nhưng khi học trên nhãn thật lại tổng quát tốt.

Vì sao? Một mạng đủ lớn có vô số nghiệm cùng đạt lỗi huấn luyện bằng $0$. Câu hỏi then chốt không còn là "mô hình có khớp được dữ liệu không" mà là "trong vô số nghiệm khớp được, thuật toán chọn nghiệm nào". Có hai nguồn lực ngầm đẩy về phía nghiệm tổng quát tốt:

- **Điều chuẩn ngầm (implicit regularization)** của bản thân thuật toán tối ưu. Hạ gradient ngẫu nhiên (SGD), khi xuất phát gần $0$, có xu hướng tìm tới các nghiệm "đơn giản" — chuẩn nhỏ, vùng cực tiểu phẳng (flat minima) — vốn ổn định hơn trước nhiễu.
- **Điều chuẩn tường minh (explicit regularization)** mà ta chủ động thêm vào: weight decay, dropout, early stopping, data augmentation. Đây là chủ đề chính của bài.

Một cách hình thức hóa lỗi tổng quát: gọi $L_{\mathcal{D}}(\theta) = \mathbb{E}_{(\mathbf{x}, y)\sim \mathcal{D}}[\ell(\theta; \mathbf{x}, y)]$ là rủi ro thật (population risk) trên phân phối dữ liệu $\mathcal{D}$, và $L_S(\theta) = \frac{1}{n}\sum_{i=1}^{n}\ell(\theta; \mathbf{x}_i, y_i)$ là rủi ro kinh nghiệm (empirical risk) trên $n$ mẫu. Cái ta thực sự quan tâm là **khoảng cách tổng quát hóa**:

$$\text{gap} = L_{\mathcal{D}}(\theta) - L_S(\theta)$$

Quá khớp chính là khi $L_S$ rất nhỏ nhưng $\text{gap}$ lớn. Mọi kỹ thuật điều chuẩn đều nhắm tới việc thu hẹp khoảng cách này, thường bằng cách **hạn chế độ phức tạp hiệu dụng** của mô hình hoặc **làm mượt** hàm số mà nó biểu diễn. Để hiểu sâu hơn về bản chất quá khớp, xem [Overfitting](#/overfitting).

---

# 2. Suy giảm trọng số (weight decay) = điều chuẩn L2

## 2.1. Trực giác

Một hàm trơn, tổng quát tốt thường không cần trọng số lớn. Trọng số lớn khiến đầu ra nhạy cảm dữ dội với thay đổi nhỏ ở đầu vào — đúng dấu hiệu của việc học cả nhiễu. Vậy ta **phạt** các trọng số có độ lớn cao, ép mô hình về phía các hàm "thoải" hơn.

## 2.2. Công thức

Thêm vào hàm mất mát một số hạng phạt tỉ lệ với bình phương chuẩn $L2$ của trọng số:

$$L_{\text{reg}}(\theta) = L_S(\theta) + \frac{\lambda}{2}\,\lVert \mathbf{w} \rVert_2^2 = L_S(\theta) + \frac{\lambda}{2}\sum_j w_j^2$$

trong đó $\lambda > 0$ là hệ số điều chuẩn điều khiển độ mạnh của phạt. Lưu ý ta thường **không phạt độ chệch (bias)**, chỉ phạt trọng số.

## 2.3. Vì sao tên là "suy giảm trọng số"

Tính gradient của số hạng phạt: $\nabla_{\mathbf{w}}\left(\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2\right) = \lambda \mathbf{w}$. Thay vào bước cập nhật hạ gradient:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta\big(\nabla_{\mathbf{w}} L_S + \lambda \mathbf{w}\big) = (1 - \eta\lambda)\,\mathbf{w} - \eta\,\nabla_{\mathbf{w}} L_S$$

Mỗi bước, trước khi trừ đi gradient của lỗi, trọng số bị **nhân co lại** một hệ số $(1 - \eta\lambda) < 1$. Đó chính là lý do gọi là "suy giảm trọng số": trọng số liên tục bị kéo về $0$ trừ khi gradient dữ liệu đẩy chúng ra. Cân bằng đạt được khi lực kéo về $0$ vừa khớp với lực của dữ liệu.

Góc nhìn Bayes: phạt $L2$ tương đương đặt **tiên nghiệm Gauss** $\mathbf{w} \sim \mathcal{N}(0, \tfrac{1}{\lambda} I)$ lên trọng số, và cực đại hóa hậu nghiệm (MAP). Hệ số $\lambda$ lớn = tin chắc trọng số phải nhỏ.

> Lưu ý kỹ thuật: với các bộ tối ưu thích nghi như Adam, "suy giảm trọng số" và "thêm phạt $L2$ vào loss" **không còn tương đương** vì gradient bị chia cho ước lượng phương sai. AdamW tách riêng số hạng suy giảm $-\eta\lambda\mathbf{w}$ ra khỏi gradient thích nghi, cho kết quả tốt hơn rõ rệt.

---

# 3. Dropout

Dropout (Srivastava et al., 2014) là một trong những phát kiến điều chuẩn ảnh hưởng nhất của học sâu, và đẹp ở chỗ vừa đơn giản vừa có nhiều cách diễn giải.

## 3.1. Định nghĩa

Trong mỗi lượt truyền xuôi khi **huấn luyện**, mỗi nơ-ron (đơn vị ẩn) bị **tắt ngẫu nhiên** — đặt đầu ra về $0$ — với xác suất $p$, độc lập nhau. Cụ thể, với mỗi đơn vị $h_i$ ta lấy một mặt nạ Bernoulli:

$$m_i \sim \text{Bernoulli}(1 - p), \qquad h_i' = m_i \cdot h_i$$

Mỗi mini-batch dùng một mặt nạ khác nhau, nên thực chất mỗi bước ta huấn luyện một **kiến trúc con (subnetwork)** khác. Để hiểu các đơn vị ẩn này, xem [Mạng nơ-ron](#/mang-no-ron).

## 3.2. Suy dẫn: giữ nguyên kỳ vọng bằng inverted dropout

Việc tắt ngẫu nhiên làm thay đổi **độ lớn kỳ vọng** của đầu ra, gây lệch giữa lúc huấn luyện và lúc kiểm tra. Xét kỳ vọng của đầu ra sau dropout (chưa hiệu chỉnh), lấy theo mặt nạ:

$$\mathbb{E}[h_i'] = \mathbb{E}[m_i]\cdot h_i = (1 - p)\,h_i$$

Tức tín hiệu trung bình **co lại còn $(1-p)$ lần**. Nếu lúc kiểm tra ta dùng đầy đủ mọi nơ-ron (không tắt gì), tổng tín hiệu vào lớp sau sẽ lớn hơn lúc huấn luyện một hệ số $\tfrac{1}{1-p}$ — phân phối kích hoạt bị lệch, mô hình hoạt động sai.

Có hai cách chữa, cùng mục tiêu **khớp kỳ vọng giữa train và test**:

1. **Hiệu chỉnh lúc test**: lúc kiểm tra nhân mọi đầu ra với $(1-p)$ để bù lại. Bất tiện vì phải đổi hành vi lúc suy luận.
2. **Inverted dropout** (chuẩn hiện nay): hiệu chỉnh ngay lúc huấn luyện bằng cách **chia cho $(1-p)$**:

$$h_i' = \frac{m_i}{1 - p}\, h_i$$

Khi đó kỳ vọng được bảo toàn ngay trong lúc huấn luyện:

$$\mathbb{E}[h_i'] = \frac{\mathbb{E}[m_i]}{1-p}\,h_i = \frac{1-p}{1-p}\,h_i = h_i$$

Nhờ vậy, **lúc kiểm tra ta không cần làm gì cả** — chỉ tắt dropout đi và dùng mạng đầy đủ như bình thường. Đây là lý do mọi framework hiện đại đều cài inverted dropout.

## 3.3. Trực giác: vì sao dropout chống quá khớp

Có hai cách hiểu, bổ trợ nhau:

- **Ngăn đồng thích nghi (co-adaptation).** Nếu một nơ-ron có thể trông cậy vào một nơ-ron bạn cụ thể luôn hiện diện, chúng sẽ "thông đồng" tạo ra các tổ hợp đặc trưng mong manh — chỉ đúng trên dữ liệu huấn luyện. Khi mỗi bạn đồng hành có thể biến mất bất kỳ lúc nào, mỗi nơ-ron buộc phải học các đặc trưng **tự thân hữu ích, dư thừa và mạnh mẽ** hơn.
- **Tập hợp ngầm (implicit ensemble).** Mỗi mặt nạ định nghĩa một mạng con; huấn luyện với dropout giống như huấn luyện đồng thời $2^{(\text{số đơn vị})}$ mạng con chia sẻ trọng số. Lúc kiểm tra, mạng đầy đủ (đã hiệu chỉnh) xấp xỉ **trung bình hình học** các dự đoán của toàn bộ tập hợp đó. Mà tập hợp (ensemble) thì nổi tiếng là giảm phương sai, tăng tổng quát hóa.

## 3.4. Train khác test

Điểm dễ sai nhất khi dùng dropout (và cả [Batch Normalization](#/batch-norm)) là quên rằng lớp có **hai chế độ**:

| Chế độ | Hành vi dropout |
| --- | --- |
| Huấn luyện (`model.train()`) | Lấy mặt nạ ngẫu nhiên, chia cho $(1-p)$ |
| Suy luận (`model.eval()`) | Không tắt gì, không nhân gì — dùng mạng đầy đủ |

```python
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(p=0.5),          # tắt 50% đơn vị khi train
    nn.Linear(256, 10),
)

net.train()   # bật dropout: lấy mặt nạ + chia (1-p)
# ... vòng lặp huấn luyện ...
net.eval()    # tắt dropout: mạng đầy đủ, không hiệu chỉnh
```

Quên gọi `eval()` lúc đánh giá là một lỗi kinh điển khiến kết quả kiểm tra nhiễu loạn vô cớ. Giá trị $p$ điển hình là $0.5$ cho lớp ẩn dày, $0.1$–$0.2$ cho lớp gần đầu vào.

---

# 4. Dừng sớm (early stopping)

Trong quá trình huấn luyện, lỗi trên tập huấn luyện thường giảm đều, nhưng lỗi trên tập kiểm định (validation) đi theo hình chữ U: giảm rồi **tăng trở lại** khi mô hình bắt đầu quá khớp.

Dừng sớm khai thác chính điều đó: theo dõi lỗi kiểm định sau mỗi epoch, và **dừng huấn luyện khi nó ngừng cải thiện** (thường cho phép một số epoch kiên nhẫn — patience — trước khi dừng hẳn), rồi giữ lại bộ trọng số tốt nhất.

$$\theta^{*} = \arg\min_{\theta_t}\; L_{\text{val}}(\theta_t), \qquad t \in \{1, 2, \dots, T_{\max}\}$$

Vì sao đây là điều chuẩn? Khi xuất phát từ trọng số nhỏ, số bước huấn luyện hữu hạn giới hạn việc trọng số có thể đi xa bao nhiêu khỏi $0$. Có thể chứng minh rằng với hồi quy tuyến tính, dừng sớm tương đương gần đúng với một mức phạt $L2$ — số epoch ít đóng vai trò như $\lambda$ lớn. Đây là kỹ thuật gần như miễn phí (chỉ cần một tập kiểm định) và luôn nên dùng.

---

# 5. Tăng cường dữ liệu (data augmentation)

Cách trực tiếp nhất để giảm khoảng cách tổng quát hóa là **có thêm dữ liệu** — vì $\text{gap}$ co lại khi $n$ tăng. Khi không thể thu thập thêm, ta tạo dữ liệu mới từ dữ liệu cũ bằng các phép biến đổi **giữ nguyên nhãn**.

Với ảnh: lật ngang, cắt ngẫu nhiên, xoay nhẹ, đổi độ sáng/tương phản, thêm nhiễu. Một con mèo bị lật vẫn là con mèo, nên nhãn không đổi. Với văn bản: thay từ đồng nghĩa, dịch vòng (back-translation).

Trực giác điều chuẩn: ta đang **dạy mô hình các bất biến (invariance)** mà ta biết phải đúng — ví dụ "phân loại không nên phụ thuộc vào hướng lật ảnh". Điều này tương đương làm mượt hàm số dọc theo các hướng biến đổi đó, ép mô hình bỏ qua những chi tiết không liên quan tới nhãn. Các kỹ thuật hiện đại như Mixup (nội suy tuyến tính cặp ảnh và nhãn) hay CutMix mở rộng ý tưởng này xa hơn.

---

# 6. Chuẩn hóa theo lô cũng điều chuẩn

[Batch Normalization](#/batch-norm) ra đời để giải quyết vấn đề [ổn định số](#/on-dinh-so) và tăng tốc huấn luyện, nhưng nó còn mang một **hiệu ứng điều chuẩn phụ**.

Lý do: thống kê chuẩn hóa (trung bình $\mu_B$ và phương sai $\sigma_B^2$) được tính trên **từng mini-batch ngẫu nhiên**, nên với cùng một mẫu, giá trị chuẩn hóa thay đổi đôi chút tùy vào những mẫu khác lọt vào lô. Đây chính là một dạng **nhiễu ngẫu nhiên** tiêm vào kích hoạt:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Nhiễu này, giống tinh thần của dropout, ngăn mô hình bám quá chặt vào từng mẫu cụ thể, qua đó giảm quá khớp. Hệ quả thực tế: khi đã dùng batch norm mạnh, nhu cầu dùng dropout thường giảm — đó là lý do nhiều kiến trúc hiện đại (ResNet) dùng batch norm mà gần như không dùng dropout ở các lớp tích chập.

---

# 7. So sánh các kỹ thuật

| Kỹ thuật | Cơ chế | Tác động chính | Khác nhau train/test? | Chi phí |
| --- | --- | --- | --- | --- |
| Weight decay (L2) | Phạt $\lVert\mathbf{w}\rVert^2$, co trọng số về $0$ | Hàm trơn, trọng số nhỏ | Không (chỉ ảnh hưởng train) | Gần như $0$ |
| Dropout | Tắt nơ-ron ngẫu nhiên, chia $(1-p)$ | Chống co-adaptation, ensemble ngầm | **Có** (test tắt dropout) | Thấp |
| Early stopping | Dừng khi lỗi val tăng | Giới hạn độ phức tạp hiệu dụng | Không | Gần như $0$ (cần tập val) |
| Data augmentation | Sinh mẫu biến đổi giữ nhãn | Dạy bất biến, tăng $n$ hiệu dụng | Thường chỉ áp dụng khi train | Trung bình |
| Batch norm | Chuẩn hóa theo lô | Ổn định + nhiễu điều chuẩn | **Có** (test dùng thống kê chạy) | Thấp |

Trong thực hành, các kỹ thuật này **bổ sung nhau** và thường được dùng đồng thời: weight decay nhẹ + augmentation + early stopping là bộ ba mặc định cho hầu hết mọi mô hình thị giác, thêm dropout ở các lớp dày toàn kết nối.

---

# 8. Tổng kết

Nghịch lý lớn của học sâu là mô hình **dư thừa năng lực** (overparameterized) lại tổng quát hóa tốt. Lời giải nằm ở chỗ: không phải mọi nghiệm khớp dữ liệu đều như nhau, và cả thuật toán tối ưu (điều chuẩn ngầm) lẫn các kỹ thuật ta chủ động thêm vào (điều chuẩn tường minh) đều đẩy về phía những nghiệm **đơn giản, trơn, ổn định**.

Mọi kỹ thuật trong bài đều quy về một mục tiêu chung — thu hẹp khoảng cách tổng quát hóa $L_{\mathcal{D}} - L_S$ — nhưng bằng những đòn bẩy khác nhau:

$$\underbrace{\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2}_{\text{phạt độ lớn}}, \quad \underbrace{m_i/(1-p)}_{\text{nhiễu cấu trúc}}, \quad \underbrace{\arg\min_t L_{\text{val}}}_{\text{giới hạn thời gian}}, \quad \underbrace{T(\mathbf{x})}_{\text{tăng bất biến}}$$

Hiểu rõ bản chất từng kỹ thuật — đặc biệt là sự khác biệt giữa chế độ huấn luyện và suy luận của dropout cùng batch norm — giúp ta tránh những lỗi tinh vi nhất và phối hợp chúng đúng cách. Điều chuẩn không phải là thứ thêm thắt tùy chọn; nó là một phần không thể tách rời của việc dạy một mạng sâu **học cái cần học, bỏ qua cái nhiễu**.
