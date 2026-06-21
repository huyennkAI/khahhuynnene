# Tự chú ý & Đa đầu (Self-Attention, Positional Encoding, Multi-Head Attention)

> Cơ chế chú ý (attention) cho phép một mô hình "nhìn" vào mọi vị trí trong chuỗi và tự quyết định nên tập trung vào đâu. Khi mỗi vị trí trong **cùng một chuỗi** đồng thời đóng vai trò câu hỏi (query), khóa (key) và giá trị (value), ta có **tự chú ý (self-attention)** — viên gạch nền của kiến trúc Transformer.
>
> $$\text{token}_i \;\longrightarrow\; (\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i) \;\longrightarrow\; \text{trộn thông tin từ toàn chuỗi}$$
>
> Đây là cơ chế đứng sau BERT, GPT và gần như toàn bộ mô hình ngôn ngữ hiện đại.

---

# 1. Từ chú ý tới tự chú ý

Trong cơ chế [chú ý (attention)](#/attention) tổng quát, ta có ba thành phần: một tập **truy vấn (query)**, một tập **khóa (key)** và một tập **giá trị (value)**. Mỗi truy vấn so khớp với mọi khóa để lấy ra một bộ trọng số, rồi dùng trọng số đó để lấy trung bình có trọng số trên các giá trị.

Trong các bài toán dịch máy ban đầu, query đến từ chuỗi đích còn key/value đến từ chuỗi nguồn — đó là **chú ý chéo (cross-attention)**. Tự chú ý là một trường hợp đặc biệt nhưng mạnh mẽ:

> **Tự chú ý:** cả query, key và value đều được sinh ra từ **chính chuỗi đầu vào**. Mỗi vị trí tự hỏi "trong toàn bộ chuỗi này, tôi nên chú ý đến những vị trí nào?" rồi tự thu thập thông tin tương ứng.

Cho một chuỗi $n$ vector đầu vào $\mathbf{x}_1, \dots, \mathbf{x}_n$, mỗi $\mathbf{x}_i \in \mathbb{R}^{d}$. Ta chiếu mỗi vector thành ba vai trò bằng ba ma trận học được $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$:

$$\mathbf{q}_i = \mathbf{W}^Q \mathbf{x}_i, \qquad \mathbf{k}_i = \mathbf{W}^K \mathbf{x}_i, \qquad \mathbf{v}_i = \mathbf{W}^V \mathbf{x}_i$$

Đầu ra tại vị trí $i$ là tổng có trọng số trên **mọi** value trong chuỗi:

$$\mathbf{y}_i = \sum_{j=1}^{n} \alpha_{ij}\, \mathbf{v}_j, \qquad \alpha_{ij} = \operatorname{softmax}_j\!\left( \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} \right)$$

Trọng số $\alpha_{ij}$ đo mức độ "liên quan" giữa vị trí $i$ và $j$. Vì $i$ chạy qua mọi giá trị $j$ (kể cả chính nó), mọi cặp vị trí đều có thể trao đổi thông tin trong **một lớp duy nhất**.

## 1.1. Dạng ma trận và scaled dot-product

Gộp các vector thành ma trận $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_k}$ (mỗi hàng là một vị trí), toàn bộ phép tự chú ý viết gọn thành:

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}$$

Tích $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$ là **ma trận điểm tương đồng (score matrix)**: phần tử $(i, j)$ chính là $\mathbf{q}_i^\top \mathbf{k}_j$.

**Vì sao chia cho $\sqrt{d_k}$?** Giả sử các thành phần của $\mathbf{q}_i$ và $\mathbf{k}_j$ độc lập, trung bình $0$ và phương sai $1$. Khi đó tích vô hướng

$$\mathbf{q}_i^\top \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il}\, k_{jl}$$

là tổng của $d_k$ số hạng độc lập, mỗi số hạng có phương sai $1$, nên phương sai của tổng bằng $d_k$ và độ lệch chuẩn bằng $\sqrt{d_k}$. Khi $d_k$ lớn, các điểm số có biên độ rất lớn, đẩy $\operatorname{softmax}$ vào vùng bão hòa — gradient gần như triệt tiêu. Chia cho $\sqrt{d_k}$ đưa phương sai về lại $1$, giữ softmax ở vùng có gradient lành mạnh.

---

# 2. So sánh CNN, RNN và self-attention

Để hiểu vì sao tự chú ý lại thay thế được [mạng hồi tiếp (RNN)](#/rnn) và [tích chập (CNN)](#/tich-chap) trong xử lý chuỗi, ta so sánh ba kiến trúc trên ba trục: **độ phức tạp tính toán** mỗi lớp, **số phép tuần tự (sequential operations)** không song song hóa được, và **độ dài đường phụ thuộc lớn nhất (maximum path length)** — quãng đường ngắn nhất mà thông tin phải đi qua giữa hai vị trí xa nhau.

Xét một chuỗi dài $n$ với số chiều biểu diễn $d$, và CNN có cửa sổ tích chập rộng $k$.

| Tiêu chí | CNN | RNN | Self-Attention |
| --- | --- | --- | --- |
| Độ phức tạp mỗi lớp | $O(k\,n\,d^2)$ | $O(n\,d^2)$ | $O(n^2 d)$ |
| Số phép tuần tự | $O(1)$ | $O(n)$ | $O(1)$ |
| Đường phụ thuộc lớn nhất | $O(\log_k n)$ | $O(n)$ | $O(1)$ |
| Song song hóa | tốt | kém | tốt |

**RNN.** Trạng thái ẩn $\mathbf{h}_t$ phụ thuộc $\mathbf{h}_{t-1}$, nên phải tính **tuần tự** qua $n$ bước — không song song hóa được. Để thông tin từ token đầu chạm token cuối, tín hiệu phải đi qua $n$ bước liên tiếp; đường phụ thuộc dài $O(n)$ khiến gradient dễ tiêu biến trên chuỗi dài.

**CNN.** Mỗi lớp tích chập song song hoàn toàn (số phép tuần tự $O(1)$), nhưng một lớp chỉ "nhìn" được cửa sổ rộng $k$. Để nối hai vị trí cách nhau $n$, cần xếp chồng nhiều lớp, đường phụ thuộc co lại còn $O(\log_k n)$ — tốt hơn RNN nhưng vẫn tăng theo $n$.

**Self-attention.** Mọi cặp vị trí nối trực tiếp qua đúng **một** phép chú ý, nên đường phụ thuộc lớn nhất chỉ là $O(1)$ — đây là ưu thế cốt lõi cho phụ thuộc tầm xa (long-range dependency). Toàn bộ ma trận điểm tính song song nên số phép tuần tự cũng $O(1)$. Cái giá phải trả: ma trận $\mathbf{Q}\mathbf{K}^\top$ có $n^2$ phần tử, nên chi phí $O(n^2 d)$ tăng bậc hai theo độ dài chuỗi — nút thắt với văn bản rất dài.

> Tóm lại: self-attention đổi **bộ nhớ/chi phí bậc hai theo $n$** lấy **đường phụ thuộc ngắn nhất có thể và song song hóa tối đa**. Với độ dài chuỗi vừa phải và phần cứng song song (GPU/TPU), đây là một giao dịch cực kỳ có lợi.

---

# 3. Vì sao cần mã hóa vị trí (positional encoding)

Hãy nhìn lại công thức $\mathbf{y}_i = \sum_j \alpha_{ij}\mathbf{v}_j$ với $\alpha_{ij}$ phụ thuộc vào $\mathbf{q}_i^\top\mathbf{k}_j$. Có một tính chất quan trọng ẩn trong đó:

> **Self-attention là bất biến với hoán vị (permutation-equivariant).** Nếu ta xáo trộn thứ tự các token đầu vào, đầu ra cũng chỉ bị xáo trộn theo đúng cách đó — bản thân phép tính **không hề biết** vị trí nào đứng trước, vị trí nào đứng sau.

Điều này khác hẳn RNN (vốn xử lý tuần tự nên mang sẵn thứ tự) và CNN (cửa sổ trượt mang sẵn thông tin lân cận cục bộ). Với ngôn ngữ, thứ tự là tối quan trọng: "chó cắn người" khác hẳn "người cắn chó". Vậy ta phải **bơm thông tin vị trí** vào đầu vào trước khi đưa vào lớp chú ý.

## 3.1. Mã hóa vị trí sin/cos

Giải pháp của Transformer là cộng thẳng một **vector mã hóa vị trí (positional encoding)** $\mathbf{p}_{pos} \in \mathbb{R}^d$ vào embedding của token tại vị trí $pos$:

$$\tilde{\mathbf{x}}_{pos} = \mathbf{x}_{pos} + \mathbf{p}_{pos}$$

Mã hóa được định nghĩa bằng các hàm sin và cos với tần số khác nhau trên từng chiều. Với vị trí $pos$ và chỉ số chiều $i \in \{0, 1, \dots, d/2 - 1\}$:

$$PE_{(pos,\, 2i)} = \sin\!\left( \frac{pos}{10000^{2i/d}} \right), \qquad PE_{(pos,\, 2i+1)} = \cos\!\left( \frac{pos}{10000^{2i/d}} \right)$$

Mỗi cặp chiều $(2i, 2i+1)$ ứng với một tần số $\omega_i = 10000^{-2i/d}$. Chiều đầu (i nhỏ) dao động nhanh; chiều sau (i lớn) dao động rất chậm. Đây chính là cách biểu diễn vị trí bằng nhiều "kim đồng hồ" quay ở tốc độ khác nhau — giống cách hệ nhị phân dùng các bit dao động ở tần số gấp đôi nhau để mã hóa số nguyên, nhưng ở dạng liên tục.

## 3.2. Vì sao chọn sin/cos? Tính chất dịch tuyến tính

Lựa chọn sin/cos không tùy tiện. Nó cho một tính chất rất đẹp:

> **Mệnh đề.** Mã hóa của vị trí $pos + \delta$ là một **phép biến đổi tuyến tính** (chỉ phụ thuộc $\delta$) của mã hóa tại $pos$. Nhờ vậy mô hình dễ học **vị trí tương đối** giữa hai token.

**Chứng minh.** Xét cặp chiều ứng với tần số $\omega_i$. Đặt $\mathbf{u}_{pos} = \big(\sin(\omega_i\, pos),\ \cos(\omega_i\, pos)\big)$. Dùng công thức cộng cung:

$$
\begin{aligned}
\sin\big(\omega_i (pos + \delta)\big) &= \sin(\omega_i\, pos)\cos(\omega_i\, \delta) + \cos(\omega_i\, pos)\sin(\omega_i\, \delta) \\
\cos\big(\omega_i (pos + \delta)\big) &= \cos(\omega_i\, pos)\cos(\omega_i\, \delta) - \sin(\omega_i\, pos)\sin(\omega_i\, \delta)
\end{aligned}
$$

Viết lại dưới dạng ma trận, ta thấy $\mathbf{u}_{pos+\delta}$ chính là $\mathbf{u}_{pos}$ quay đi một góc $\omega_i\delta$:

$$
\begin{pmatrix} \sin(\omega_i (pos+\delta)) \\ \cos(\omega_i (pos+\delta)) \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos(\omega_i \delta) & \sin(\omega_i \delta) \\ -\sin(\omega_i \delta) & \cos(\omega_i \delta) \end{pmatrix}}_{\text{ma trận quay } \mathbf{R}(\omega_i \delta)}
\begin{pmatrix} \sin(\omega_i\, pos) \\ \cos(\omega_i\, pos) \end{pmatrix}
$$

Ma trận quay $\mathbf{R}(\omega_i\delta)$ **không phụ thuộc $pos$**, chỉ phụ thuộc độ lệch $\delta$. Do đó với mỗi $\delta$ cố định, $\mathbf{p}_{pos+\delta} = \mathbf{M}_\delta\, \mathbf{p}_{pos}$ với $\mathbf{M}_\delta$ là ma trận khối-đường-chéo gồm các khối quay. $\blacksquare$

Hai hệ quả quan trọng:

* **Học vị trí tương đối dễ dàng.** Một lớp tuyến tính chỉ cần học $\mathbf{M}_\delta$ là biểu diễn được quan hệ "cách nhau $\delta$ bước" cho mọi vị trí.
* **Ngoại suy độ dài (length extrapolation).** Vì sin/cos xác định cho mọi $pos$, mô hình có thể xử lý chuỗi dài hơn cả chuỗi từng thấy khi huấn luyện — điều mà bảng embedding vị trí học được không làm được.

---

# 4. Chú ý đa đầu (Multi-Head Attention)

Một lớp tự chú ý đơn cho mỗi vị trí một cách "trộn" thông tin duy nhất. Nhưng quan hệ trong ngôn ngữ rất đa dạng: có quan hệ cú pháp (chủ ngữ–động từ), có quan hệ ngữ nghĩa (đại từ–danh từ mà nó thay thế), có quan hệ lân cận. Một bộ trọng số $\alpha_{ij}$ khó lòng nắm bắt tất cả cùng lúc.

> **Ý tưởng đa đầu (multi-head):** thay vì một phép chú ý trên toàn bộ không gian $d$ chiều, hãy chạy $h$ phép chú ý **song song** trên $h$ không gian con khác nhau, mỗi "đầu" tự do học một kiểu quan hệ riêng, rồi nối kết quả lại.

## 4.1. Công thức

Với mỗi đầu $j = 1, \dots, h$, ta có ba ma trận chiếu riêng $\mathbf{W}^Q_j, \mathbf{W}^K_j \in \mathbb{R}^{d \times d_k}$ và $\mathbf{W}^V_j \in \mathbb{R}^{d \times d_v}$, đưa Q/K/V về một không gian con chiều thấp hơn (thường $d_k = d_v = d/h$):

$$\mathbf{H}_j = \operatorname{Attention}\!\left( \mathbf{Q}\mathbf{W}^Q_j,\ \mathbf{K}\mathbf{W}^K_j,\ \mathbf{V}\mathbf{W}^V_j \right) \in \mathbb{R}^{n \times d_v}$$

Nối (Concat) đầu ra của tất cả các đầu rồi chiếu lại về không gian gốc bằng ma trận $\mathbf{W}^O \in \mathbb{R}^{h d_v \times d}$:

$$\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\mathbf{H}_1, \dots, \mathbf{H}_h)\, \mathbf{W}^O$$

## 4.2. Vì sao nhiều đầu? Vì sao chia chiều?

**Nhiều kiểu quan hệ.** Mỗi đầu có ma trận chiếu riêng nên định nghĩa "độ tương đồng" theo cách riêng. Trong thực tế, các đầu khác nhau học những vai trò khác nhau: một đầu chuyên nhìn từ liền kề, một đầu chuyên liên kết đại từ với danh từ ở xa, một đầu chuyên bám động từ chính. Nối nhiều góc nhìn cho biểu diễn phong phú hơn hẳn một đầu duy nhất.

**Gần như miễn phí về chi phí.** Vì mỗi đầu hoạt động trên không gian con chiều $d/h$, tổng chi phí của $h$ đầu xấp xỉ chi phí của **một** đầu đầy đủ chiều $d$:

$$\underbrace{h}_{\text{số đầu}} \times \underbrace{O\!\left(n^2 \cdot \tfrac{d}{h}\right)}_{\text{mỗi đầu}} = O(n^2 d)$$

Ta được nhiều không gian biểu diễn mà gần như không tăng tính toán so với một đầu — đây là lý do đa đầu là lựa chọn mặc định.

---

# 5. Chú ý có mặt nạ (Masked Attention)

Trong các mô hình **sinh tự hồi quy (autoregressive)** như GPT, bộ giải mã (decoder) phải sinh token thứ $t$ chỉ dựa vào các token $1, \dots, t$ đã sinh trước — **không được nhìn tương lai**. Nhưng self-attention mặc định cho mỗi vị trí nhìn toàn chuỗi, kể cả các vị trí phía sau. Đây là rò rỉ thông tin (leakage): khi huấn luyện song song trên cả câu, mô hình sẽ "gian lận" bằng cách nhìn đáp án.

Giải pháp là **mặt nạ nhân quả (causal mask)**: trước khi lấy softmax, đặt các điểm số ứng với vị trí tương lai ($j > i$) thành $-\infty$:

$$\alpha_{ij} = \operatorname{softmax}_j\!\left( \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} + m_{ij} \right), \qquad m_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

Vì $\exp(-\infty) = 0$, mọi trọng số chú ý vào tương lai bị triệt tiêu sau softmax, đảm bảo $\mathbf{y}_i$ chỉ phụ thuộc các vị trí $\le i$. Mẹo dùng $-\infty$ (thực tế là một số âm rất lớn) thay vì gán $0$ trực tiếp giúp softmax vẫn chuẩn hóa đúng trên đúng tập vị trí hợp lệ.

Phân biệt hai chế độ chú ý trong một Transformer encoder–decoder:

* **Chú ý hai chiều (bidirectional)** trong encoder và trong BERT: không mặt nạ, mỗi vị trí nhìn toàn chuỗi — hợp cho bài toán hiểu.
* **Chú ý nhân quả (causal)** trong decoder và trong GPT: có mặt nạ tam giác, hợp cho bài toán sinh.

---

# 6. Dẫn tới Transformer

Ba thành phần ở trên — tự chú ý, mã hóa vị trí và chú ý đa đầu — kết hợp với mạng truyền thẳng theo vị trí (position-wise feed-forward), kết nối tắt (residual) và chuẩn hóa lớp (layer normalization), tạo thành một khối Transformer. Xếp chồng nhiều khối như vậy cho ta kiến trúc [Transformer](#/transformer) đầy đủ, nền tảng của mọi mô hình ngôn ngữ lớn hiện nay.

Điều đáng nói là toàn bộ khối này **không có một phép hồi tiếp hay tích chập nào** — đúng tinh thần tiêu đề bài báo gốc "Attention Is All You Need". Mọi phụ thuộc, dù gần hay xa, đều được học qua các trọng số chú ý.

---

# 7. Tổng kết

Tự chú ý là một ý tưởng đơn giản đến bất ngờ: cho mỗi vị trí tự sinh ra query/key/value từ chính chuỗi, rồi để nó tự quyết định trộn thông tin từ đâu qua một phép softmax trên tích vô hướng có chia tỉ lệ:

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}$$

So với [RNN](#/rnn) và [CNN](#/tich-chap), self-attention cho **đường phụ thuộc $O(1)$** và **song song hóa hoàn toàn**, đổi lại chi phí bậc hai $O(n^2 d)$ theo độ dài chuỗi. Vì phép tính bất biến với hoán vị, ta phải cộng **mã hóa vị trí sin/cos** — vốn cho phép biểu diễn vị trí tương đối bằng một phép quay tuyến tính và ngoại suy được độ dài. **Chú ý đa đầu** chạy nhiều phép chú ý song song trên các không gian con để học nhiều kiểu quan hệ mà gần như không tăng chi phí, còn **mặt nạ nhân quả** biến self-attention thành công cụ sinh tự hồi quy hợp lệ.

> Ghép tất cả lại — cộng thêm feed-forward, residual và layer norm — ta có khối Transformer. Bài tiếp theo về [Transformer](#/transformer) sẽ lắp ráp các mảnh này thành kiến trúc encoder–decoder hoàn chỉnh và giải thích vì sao nó trở thành xương sống của toàn bộ AI hiện đại.
