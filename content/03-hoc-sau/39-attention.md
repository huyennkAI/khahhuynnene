# Cơ chế chú ý (Attention Mechanism)

> Cơ chế chú ý (attention mechanism) cho phép một mô hình **tự chọn nơi cần nhìn**: thay vì ép toàn bộ thông tin vào một vector cố định, nó tính ra một trung bình có trọng số trên nhiều biểu diễn, với trọng số phụ thuộc vào ngữ cảnh hiện tại.
>
> $$\operatorname{Attention}(\mathbf{q}, \{(\mathbf{k}_i, \mathbf{v}_i)\}) = \sum_{i} \alpha(\mathbf{q}, \mathbf{k}_i)\, \mathbf{v}_i$$
>
> Đây là viên gạch nền của Transformer và gần như mọi mô hình ngôn ngữ lớn hiện đại.

---

# 1. Động cơ: nút thắt của seq2seq

Hãy nhớ lại kiến trúc [seq2seq](#/seq2seq) cho dịch máy: một bộ mã hóa (encoder) RNN đọc câu nguồn rồi nén **toàn bộ** ý nghĩa của nó vào một vector trạng thái cuối cùng $\mathbf{c}$ (context vector). Bộ giải mã (decoder) sau đó sinh từng từ đích chỉ dựa vào đúng một vector $\mathbf{c}$ này.

Đây chính là nút thắt cổ chai (bottleneck). Một câu dài 40 từ và một câu ngắn 4 từ đều bị ép vào cùng một vector cố định cùng kích thước. Khi câu càng dài, $\mathbf{c}$ càng phải "gánh" nhiều thông tin, và những từ ở đầu câu dễ bị lãng quên.

> Trực giác sửa lỗi rất tự nhiên: khi một người dịch sinh ra từ đích thứ $t$, họ không nhìn cả câu nguồn như một khối mờ. Họ **liếc lại** đúng vài từ nguồn liên quan nhất tới từ đang dịch.

Nói cách khác, thay vì một context vector duy nhất dùng chung cho mọi bước, ta muốn **mỗi bước giải mã có một context vector riêng** $\mathbf{c}_t$, được tổng hợp lại từ tất cả các trạng thái nguồn nhưng **nhấn mạnh** vào những trạng thái liên quan. Cơ chế chú ý chính là cách hình thức hóa từ "liên quan" và "nhấn mạnh" đó.

---

# 2. Khung truy vấn – khóa – giá trị

Để tổng quát hóa, attention được phát biểu qua ba vai trò.

- **Truy vấn (query)** $\mathbf{q}$: thứ ta đang cần, "câu hỏi" của bước hiện tại. Trong dịch máy, đây là trạng thái decoder tại bước $t$.
- **Khóa (key)** $\mathbf{k}_i$: nhãn dán để so khớp với truy vấn, dùng để tính độ liên quan. Mỗi vị trí nguồn có một khóa.
- **Giá trị (value)** $\mathbf{v}_i$: nội dung thực sự được lấy ra nếu khóa khớp. Thường (nhưng không bắt buộc) đi kèm với khóa.

Một cách ẩn dụ: attention giống tra cứu trong một cuốn từ điển mềm. Truy vấn là từ ta tra; mỗi mục từ có một khóa (từ khóa) và một giá trị (định nghĩa). Tra cứu cứng (hard lookup) trả về đúng một giá trị có khóa trùng khớp. Tra cứu mềm (soft lookup) của attention trả về **trung bình của mọi giá trị**, trọng số theo mức khớp giữa truy vấn và từng khóa.

Cho một truy vấn $\mathbf{q}$ và $m$ cặp khóa–giá trị $\{(\mathbf{k}_1, \mathbf{v}_1), \dots, (\mathbf{k}_m, \mathbf{v}_m)\}$, **gộp chú ý (attention pooling)** là:

$$\operatorname{Attention}(\mathbf{q}, \{(\mathbf{k}_i, \mathbf{v}_i)\}) = \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i)\, \mathbf{v}_i$$

trong đó $\alpha(\mathbf{q}, \mathbf{k}_i)$ là **trọng số chú ý (attention weight)**, một số vô hướng đo mức liên quan giữa truy vấn và khóa thứ $i$.

---

# 3. Trọng số chú ý là một phân phối softmax

Ta muốn trọng số $\alpha(\mathbf{q}, \mathbf{k}_i)$ hành xử như một phân phối xác suất: không âm và tổng bằng $1$. Khi đó attention pooling là một **tổ hợp lồi (convex combination)** của các giá trị, nên kết quả luôn nằm trong bao lồi của các value — ổn định về số học.

Cách đạt được điều này: tính một **điểm số (score)** thô $a(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ cho mỗi khóa bằng một hàm chấm điểm (scoring function), rồi chuẩn hóa qua softmax:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \operatorname{softmax}\big(a(\mathbf{q}, \mathbf{k}_i)\big) = \frac{\exp\!\big(a(\mathbf{q}, \mathbf{k}_i)\big)}{\sum_{j=1}^{m} \exp\!\big(a(\mathbf{q}, \mathbf{k}_j)\big)}$$

Score càng cao → khóa càng được coi là liên quan → trọng số càng lớn → giá trị tương ứng đóng góp càng nhiều vào kết quả. Toàn bộ thiết kế của các biến thể attention rút lại về một câu hỏi duy nhất: **chọn hàm chấm điểm $a(\mathbf{q}, \mathbf{k})$ như thế nào?**

> Lưu ý đẹp: vì softmax khả vi và tổng có trọng số khả vi, toàn bộ phép gộp chú ý là **khả vi end-to-end**. Mô hình tự học cách chú ý qua lan truyền ngược, không cần ta dạy nó "nên nhìn đâu".

---

# 4. Một ví dụ cổ điển: hồi quy Nadaraya–Watson

Trước khi học sâu, ý tưởng "trung bình có trọng số theo độ tương đồng" đã tồn tại trong thống kê dưới tên **hồi quy Nadaraya–Watson (1964)**. Đây là attention pooling phiên bản không tham số.

Cho dữ liệu $\{(x_i, y_i)\}$, để dự đoán tại điểm truy vấn $x$, ta lấy trung bình các $y_i$ với trọng số theo mức gần gũi giữa $x$ và $x_i$:

$$\hat{f}(x) = \sum_{i=1}^{n} \frac{K(x - x_i)}{\sum_{j=1}^{n} K(x - x_j)}\, y_i$$

với $K$ là một nhân (kernel). Đối chiếu khung query–key–value: truy vấn là $x$, khóa là $x_i$, giá trị là $y_i$, còn nhân $K$ đóng vai trò scoring. Nếu chọn nhân Gauss $K(u) = \exp(-u^2/2)$, công thức trên chính là softmax trên các điểm $-\tfrac{1}{2}(x - x_i)^2$ — tức một attention pooling với hàm chấm điểm dựa trên khoảng cách. Attention hiện đại chỉ khác ở chỗ: thay nhân cố định bằng **một hàm chấm điểm có tham số học được** trên các vector nhiều chiều.

---

# 5. Các hàm chấm điểm (scoring functions)

Trong thực hành, ta gộp cho **nhiều truy vấn cùng lúc**. Xếp $n$ truy vấn thành ma trận $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, $m$ khóa thành $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, và $m$ giá trị thành $\mathbf{V} \in \mathbb{R}^{m \times d_v}$.

## 5.1. Chú ý cộng (additive / Bahdanau attention)

Khi truy vấn và khóa có **số chiều khác nhau**, ta chiếu cả hai về cùng một không gian ẩn rồi cộng:

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh\!\big(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}\big) \in \mathbb{R}$$

trong đó $\mathbf{W}_q, \mathbf{W}_k$ và $\mathbf{w}_v$ là tham số học được. Bản chất đây là một mạng nơ-ron một lớp ẩn với hàm kích hoạt $\tanh$, dùng làm bộ chấm điểm. Ưu điểm: linh hoạt, không đòi $d_q = d_k$. Nhược điểm: tốn thêm tham số và chậm hơn vì có phép $\tanh$.

## 5.2. Chú ý tích vô hướng (dot-product attention)

Khi truy vấn và khóa **cùng số chiều** $d_k$, có một bộ chấm điểm rẻ hơn nhiều: chỉ cần tích vô hướng

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$$

Tích vô hướng đo mức "đồng hướng" giữa hai vector, là một thước đo tương đồng tự nhiên, lại tận dụng được phép nhân ma trận tối ưu cao trên GPU. Không có tham số riêng cho bộ chấm điểm — gọn và nhanh.

| Hàm chấm điểm | Công thức $a(\mathbf{q},\mathbf{k})$ | Tham số riêng | Yêu cầu |
| --- | --- | --- | --- |
| Cộng (additive) | $\mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | $\mathbf{W}_q, \mathbf{W}_k, \mathbf{w}_v$ | $d_q, d_k$ tùy ý |
| Tích vô hướng | $\mathbf{q}^\top \mathbf{k}$ | không | $d_q = d_k$ |
| Tích vô hướng có tỉ lệ | $\mathbf{q}^\top \mathbf{k}/\sqrt{d_k}$ | không | $d_q = d_k$ |

---

# 6. Chú ý tích vô hướng có tỉ lệ (scaled dot-product)

Đây là dạng attention được Transformer dùng và là dạng quan trọng nhất:

$$\boxed{\;\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}\;}$$

Trong đó $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times m}$ chứa toàn bộ điểm số (phần tử $(i,j)$ là $\mathbf{q}_i^\top \mathbf{k}_j$), softmax áp theo từng hàng cho ra ma trận trọng số, rồi nhân với $\mathbf{V}$ để gộp giá trị. Điểm khác biệt duy nhất so với dot-product thường là thừa số chia $\sqrt{d_k}$. Tại sao lại cần nó?

## 6.1. Suy dẫn: vì sao chia cho $\sqrt{d_k}$

Trực giác: khi số chiều $d_k$ lớn, tích vô hướng có xu hướng phình ra rất to (cả dương lẫn âm). Mà softmax với đầu vào biên độ lớn thì **bão hòa** — gần như dồn toàn bộ khối lượng vào một phần tử, biến phân phối mềm thành phân phối gần one-hot. Khi đó gradient của softmax gần như triệt tiêu ở mọi nơi, mô hình **khó học**. Ta hãy lượng hóa điều này.

**Mệnh đề.** Giả sử các thành phần của $\mathbf{q}$ và $\mathbf{k}$ là các biến ngẫu nhiên độc lập, trung bình $0$, phương sai $1$. Khi đó tích vô hướng $\mathbf{q}^\top\mathbf{k}$ có trung bình $0$ và **phương sai $d_k$**.

**Chứng minh.** Viết $\mathbf{q}^\top\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$. Vì $q_i, k_i$ độc lập và đều trung bình $0$:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0 \quad\Longrightarrow\quad \mathbb{E}[\mathbf{q}^\top\mathbf{k}] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$

Với phương sai, do các số hạng $q_i k_i$ độc lập nhau nên phương sai cộng dồn:

$$\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i)$$

Mỗi số hạng, vì $q_i \perp k_i$ và cả hai trung bình $0$:

$$\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - \big(\mathbb{E}[q_i k_i]\big)^2 = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1$$

Do đó $\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k} 1 = d_k$. $\blacksquare$

Vậy độ lệch chuẩn của điểm số là $\sqrt{d_k}$, **tăng theo căn của số chiều**. Với $d_k = 64$, điểm số dao động cỡ $\pm 8$; với $d_k = 512$, dao động cỡ $\pm 22$ — đủ để đẩy softmax vào vùng bão hòa.

Cách sửa: **chuẩn hóa lại phương sai về $1$**. Chia điểm số cho độ lệch chuẩn $\sqrt{d_k}$:

$$\operatorname{Var}\!\left(\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{d_k}}\right) = \frac{1}{d_k}\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = \frac{d_k}{d_k} = 1$$

Sau khi chia, điểm số có phương sai $1$ **bất kể $d_k$ lớn bao nhiêu**. Softmax do đó luôn nằm trong vùng có gradient lành mạnh, huấn luyện ổn định. Đây chính xác là lý do của thừa số $\tfrac{1}{\sqrt{d_k}}$ — không phải một hằng số tùy tiện, mà là phép chuẩn hóa phương sai có suy dẫn rõ ràng.

> Tóm lại: additive attention tự "miễn dịch" với $d_k$ lớn nhờ có lớp chiếu học được, nên không cần tỉ lệ. Dot-product attention thì cần thừa số $1/\sqrt{d_k}$ để bù lại đúng phần phương sai do số chiều sinh ra. Đổi lại nó nhanh hơn và ít tham số hơn — đó là lý do Transformer chọn nó.

---

# 7. Bahdanau attention trong dịch máy

Bahdanau và cộng sự (2014) là những người đầu tiên gắn attention vào [seq2seq](#/seq2seq), trực tiếp đập tan nút thắt ở mục 1. Ý tưởng: ở mỗi bước giải mã $t$, dùng **trạng thái decoder $\mathbf{s}_{t-1}$ làm truy vấn**, còn **các trạng thái ẩn của encoder $\mathbf{h}_1, \dots, \mathbf{h}_T$ làm đồng thời khóa và giá trị**.

Context vector giờ được tính lại ở **mỗi bước**, không còn cố định:

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha(\mathbf{s}_{t-1}, \mathbf{h}_i)\, \mathbf{h}_i, \qquad \alpha(\mathbf{s}_{t-1}, \mathbf{h}_i) = \operatorname{softmax}\big(a(\mathbf{s}_{t-1}, \mathbf{h}_i)\big)$$

với $a$ là bộ chấm điểm cộng ở mục 5.1 (bản gốc Bahdanau dùng additive). Decoder sinh từ đích thứ $t$ dựa trên context $\mathbf{c}_t$ riêng cho bước đó.

Tác động thực tế và quan sát được:

- **Câu dài không còn xuống chất lượng đột ngột**: thông tin xa không bị nén vào một vector duy nhất nữa.
- **Trọng số $\alpha$ có thể trực quan hóa**: vẽ ma trận $\alpha_{t,i}$ ra, ta thấy mô hình tự học một kiểu **căn chỉnh mềm (soft alignment)** giữa từ nguồn và từ đích — ví dụ khi dịch "the cat" sang tiếng Pháp, lúc sinh "chat" thì trọng số dồn vào "cat".

Đây là minh chứng đầu tiên rằng attention không chỉ là một mẹo kỹ thuật, mà còn học ra cấu trúc có ý nghĩa ngôn ngữ.

---

# 8. Từ attention tới self-attention và multi-head

Trong Bahdanau attention, truy vấn đến từ decoder còn khóa/giá trị đến từ encoder — đây là **cross-attention** giữa hai chuỗi. Một bước tổng quát hóa táo bạo: nếu ta cho **truy vấn, khóa và giá trị đều đến từ cùng một chuỗi** thì sao?

Đó là **tự chú ý (self-attention)**: mỗi vị trí trong câu tạo ra truy vấn của riêng nó và nhìn vào mọi vị trí khác (kể cả chính nó) trong cùng câu để thu thập ngữ cảnh. Khác với RNN xử lý tuần tự, self-attention nối **mọi cặp vị trí trực tiếp trong một bước**, nên đường lan truyền thông tin giữa hai từ xa nhau chỉ dài $O(1)$ thay vì $O(\text{khoảng cách})$ — và toàn bộ tính song song được.

Một bộ chấm điểm đơn lẻ chỉ học được một kiểu quan hệ. Để mô hình nhìn dữ liệu qua nhiều "lăng kính" cùng lúc — quan hệ cú pháp, quan hệ ngữ nghĩa, quan hệ vị trí... — ta chạy nhiều attention song song rồi ghép lại, gọi là [chú ý đa đầu (multi-head attention)](#/multi-head-attention). Khi xếp chồng nhiều lớp self-attention đa đầu lại, ta thu được [Transformer](#/transformer), kiến trúc đứng sau gần như mọi mô hình ngôn ngữ lớn ngày nay.

---

# 9. Tổng kết

Cơ chế chú ý ra đời để gỡ một nút thắt cụ thể của [seq2seq](#/seq2seq): không thể nén cả một câu vào một context vector cố định. Lời giải là một ý tưởng đơn giản nhưng có sức mạnh vượt xa bài toán gốc — **tra cứu mềm có trọng số**:

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

Ba điểm cần nhớ:

- **Khung query–key–value** thống nhất mọi biến thể: đầu ra luôn là trung bình các value, trọng số là softmax của điểm tương đồng giữa query và key. Hồi quy Nadaraya–Watson là phiên bản cổ điển, không tham số của chính ý tưởng này.
- **Hàm chấm điểm** quyết định cách đo tương đồng: additive (Bahdanau) linh hoạt với số chiều bất kỳ; dot-product nhanh và gọn nhưng cần thừa số $1/\sqrt{d_k}$ để chống bão hòa softmax — một phép chuẩn hóa phương sai có suy dẫn chặt chẽ, không phải hằng số tùy ý.
- **Bahdanau attention** chứng minh giá trị thực tế bằng cách học ra căn chỉnh mềm trong dịch máy.

> Bài tiếp theo — [chú ý đa đầu (multi-head attention)](#/multi-head-attention) — cho mô hình nhìn qua nhiều lăng kính song song; rồi [Transformer](#/transformer) xếp chồng self-attention đa đầu thành kiến trúc thống trị xử lý ngôn ngữ tự nhiên hiện đại.
