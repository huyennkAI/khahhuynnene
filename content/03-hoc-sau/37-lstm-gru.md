# LSTM & GRU

> Mạng hồi quy thường (vanilla RNN) về lý thuyết có thể nhớ vô hạn, nhưng trong thực tế lại **quên rất nhanh**: gradient lan ngược qua nhiều bước thời gian bị triệt tiêu (vanishing) nên mô hình không học được các phụ thuộc xa.
>
> $$\text{"Mây đen kéo đến, gió nổi lên, và rồi trời bắt đầu } \underline{\;?\;}\text{"}$$
>
> Để điền được từ "mưa", mô hình phải nhớ manh mối từ đầu câu. LSTM và GRU sinh ra để giải quyết đúng bài toán này — bằng cách thêm các **cổng (gate)** điều khiển dòng thông tin.

---

# 1. Vấn đề: RNN thường quên những gì ở xa

Mạng hồi quy thường (xem [RNN](#/rnn)) cập nhật trạng thái ẩn theo công thức:

$$\mathbf{h}_t = \tanh\!\left(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h\right)$$

Mỗi bước, trạng thái cũ $\mathbf{h}_{t-1}$ bị nhân với ma trận $\mathbf{W}_{hh}$ rồi ép qua hàm phi tuyến $\tanh$. Khi huấn luyện bằng lan truyền ngược theo thời gian (backpropagation through time), gradient của mất mát tại bước $t$ truyền về bước $k$ xa hơn phải đi qua một **tích các ma trận Jacobian**:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=k+1}^{t} \operatorname{diag}\!\big(\tanh'(\cdot)\big)\,\mathbf{W}_{hh}^{\top}$$

Đây là cội nguồn của mọi rắc rối. Tích này gồm $t - k$ thừa số nhân liên tiếp:

- Nếu các thừa số có độ lớn **nhỏ hơn 1** (giá trị riêng của $\mathbf{W}_{hh}$ nhỏ, hoặc $\tanh'$ bão hòa về $0$), tích co lại theo cấp số nhân $\to 0$: **gradient triệt tiêu (vanishing gradient)**. Tín hiệu học từ quá khứ xa biến mất, mô hình chỉ còn nhớ vài bước gần nhất.
- Nếu các thừa số **lớn hơn 1**, tích phình ra theo cấp số nhân $\to \infty$: **gradient bùng nổ (exploding gradient)**, làm việc huấn luyện mất ổn định.

Bùng nổ còn chữa được bằng cắt gradient (gradient clipping), nhưng triệt tiêu thì khó hơn nhiều vì nó âm thầm xóa thông tin (xem thêm [Ổn định số](#/on-dinh-so)). Vấn đề mấu chốt:

> $\tanh$ luôn ép tín hiệu về khoảng $(-1, 1)$ và $\tanh'$ luôn $\le 1$, nên mỗi bước qua RNN thường tín hiệu lại bị **bóp nhỏ một lần nữa**. Không có con đường nào để thông tin đi xuyên qua nhiều bước mà không bị suy giảm.

LSTM và GRU sửa đúng chỗ này: tạo ra một **con đường gần như tuyến tính** cho thông tin chảy qua thời gian.

---

# 2. Ý tưởng cốt lõi: cổng và đường truyền cộng

Trước khi vào công thức, hãy nắm hai trực giác trung tâm.

**Trực giác 1 — Cổng (gating).** Thay vì mỗi bước đều ghi đè toàn bộ trạng thái, ta để mạng **tự học khi nào nên nhớ, khi nào nên quên, khi nào nên đọc ra**. Một cổng là một vector hệ số trong khoảng $(0, 1)$, sinh ra bởi hàm sigmoid:

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

Nhân từng phần tử (element-wise, ký hiệu $\odot$) một vector với cổng giống như vặn các "van" độc lập cho từng chiều: cổng gần $1$ cho thông tin đi qua, cổng gần $0$ chặn lại.

**Trực giác 2 — Đường truyền cộng (additive path).** RNN thường cập nhật theo kiểu *nhân* ($\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h}_{t-1} + \dots)$), nên gradient thành tích các ma trận và dễ triệt tiêu. LSTM thay bằng cập nhật kiểu *cộng* trên một ô nhớ riêng:

$$\mathbf{c}_t = (\text{giữ lại phần cũ}) + (\text{thêm phần mới})$$

Cộng thay vì nhân chính là chìa khóa — ta sẽ thấy ở mục 4 vì sao nó làm gradient chảy ổn định.

---

# 3. LSTM (Long Short-Term Memory)

LSTM duy trì **hai** dòng trạng thái song song qua thời gian:

- **Ô nhớ (cell state)** $\mathbf{c}_t$ — bộ nhớ dài hạn, là "băng tải" chạy thẳng qua các bước với rất ít can thiệp.
- **Trạng thái ẩn (hidden state)** $\mathbf{h}_t$ — đầu ra ngắn hạn tại mỗi bước, đồng thời là thứ được dùng cho dự đoán và truyền sang bước sau.

Tại mỗi bước, LSTM tính ba cổng và một ứng viên từ đầu vào $\mathbf{x}_t$ và trạng thái ẩn trước $\mathbf{h}_{t-1}$.

## 3.1. Ba cổng và ô nhớ ứng viên

**Cổng quên (forget gate)** $\mathbf{f}_t$ — quyết định giữ lại bao nhiêu phần ô nhớ cũ:

$$\mathbf{f}_t = \sigma\!\left(\mathbf{W}_{xf}\mathbf{x}_t + \mathbf{W}_{hf}\mathbf{h}_{t-1} + \mathbf{b}_f\right)$$

**Cổng vào (input gate)** $\mathbf{i}_t$ — quyết định ghi bao nhiêu thông tin mới vào ô nhớ:

$$\mathbf{i}_t = \sigma\!\left(\mathbf{W}_{xi}\mathbf{x}_t + \mathbf{W}_{hi}\mathbf{h}_{t-1} + \mathbf{b}_i\right)$$

**Cổng ra (output gate)** $\mathbf{o}_t$ — quyết định lộ bao nhiêu phần ô nhớ ra ngoài thành trạng thái ẩn:

$$\mathbf{o}_t = \sigma\!\left(\mathbf{W}_{xo}\mathbf{x}_t + \mathbf{W}_{ho}\mathbf{h}_{t-1} + \mathbf{b}_o\right)$$

**Ô nhớ ứng viên (candidate cell)** $\tilde{\mathbf{c}}_t$ — nội dung mới *đề xuất* để ghi vào, dùng $\tanh$ nên nằm trong $(-1, 1)$:

$$\tilde{\mathbf{c}}_t = \tanh\!\left(\mathbf{W}_{xc}\mathbf{x}_t + \mathbf{W}_{hc}\mathbf{h}_{t-1} + \mathbf{b}_c\right)$$

Lưu ý ba cổng dùng $\sigma$ (vai trò "van", giá trị $0$–$1$), còn ứng viên dùng $\tanh$ (vai trò "nội dung", giá trị $-1$–$1$).

## 3.2. Cập nhật ô nhớ và trạng thái ẩn

Đây là trái tim của LSTM. Ô nhớ mới là **phần cũ được cổng quên giữ lại, cộng với phần mới được cổng vào cho vào**:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

Rồi trạng thái ẩn là ô nhớ được "nén" qua $\tanh$ và lọc bởi cổng ra:

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Hãy đọc lại công thức $\mathbf{c}_t$ theo nghĩa định tính:

- $\mathbf{f}_t \to 1,\ \mathbf{i}_t \to 0$: ô nhớ **giữ nguyên** $\mathbf{c}_{t-1}$ — thông tin được bảo toàn qua bước này (nhớ dài hạn).
- $\mathbf{f}_t \to 0,\ \mathbf{i}_t \to 1$: ô nhớ **xóa cũ, ghi mới** hoàn toàn — như reset bộ nhớ.
- Giá trị trung gian: pha trộn mềm giữa nhớ và quên, học được từ dữ liệu.

> Khác biệt căn bản so với RNN thường: ở đây $\mathbf{c}_{t-1}$ đi vào $\mathbf{c}_t$ qua một phép **cộng** và một phép **nhân với cổng** $\mathbf{f}_t$ — chứ không bị ép qua ma trận trọng số rồi $\tanh$. Chính con đường này nuôi gradient.

---

# 4. Vì sao đường ô nhớ chống được triệt tiêu gradient

Đây là phần lý giải then chốt. Xét gradient của ô nhớ chảy ngược dọc theo "băng tải" $\mathbf{c}$. Lấy đạo hàm công thức cập nhật ô nhớ:

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \operatorname{diag}(\mathbf{f}_t) + (\text{các số hạng qua } \mathbf{f}_t, \mathbf{i}_t, \tilde{\mathbf{c}}_t)$$

Số hạng trội là $\operatorname{diag}(\mathbf{f}_t)$. Khi truyền gradient từ bước $t$ về bước $k$ dọc theo riêng đường ô nhớ, ta được tích:

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_k} \approx \prod_{i=k+1}^{t} \operatorname{diag}(\mathbf{f}_i)$$

So sánh với RNN thường (mục 1): ở đó tích chứa $\mathbf{W}_{hh}^{\top}$ và $\tanh'$ — những thừa số mà mạng **không kiểm soát trực tiếp** và thường $< 1$. Còn ở LSTM, thừa số là cổng quên $\mathbf{f}_i$ mà **mạng tự học**:

- Khi mạng cần nhớ lâu, nó học để $\mathbf{f}_i \approx 1$. Tích các số gần $1$ **không co về $0$** — gradient được bảo toàn qua hàng trăm bước. Đây gọi là **constant error carousel**: lỗi quay vòng gần như không suy giảm.
- Quan trọng hơn, cập nhật là **cộng** ($\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$). Đạo hàm của một tổng tách thành tổng các đạo hàm, nên gradient có một "lối tắt" cộng dồn (giống ý tưởng kết nối tắt residual) thay vì bị nhân nén liên tục.

> Tóm lại: RNN thường buộc gradient đi qua một tích các ma trận cố định và phi tuyến bão hòa $\Rightarrow$ triệt tiêu. LSTM mở một đường cộng, với hệ số suy giảm là cổng quên do mạng tự điều khiển $\Rightarrow$ giữ được gradient khi cần. Đó là toàn bộ phép màu.

Một mẹo thực hành đi kèm: khởi tạo **bias của cổng quên** $\mathbf{b}_f$ ở giá trị dương (ví dụ $1$ hoặc $2$) để buổi đầu $\mathbf{f}_t \approx 1$, giúp mạng mặc định *nhớ* và học phụ thuộc xa dễ hơn.

---

# 5. GRU (Gated Recurrent Unit)

GRU (Cho et al., 2014) là phiên bản **gọn hơn**: bỏ ô nhớ riêng, chỉ giữ một trạng thái ẩn $\mathbf{h}_t$, và dùng **hai** cổng thay vì ba.

## 5.1. Hai cổng của GRU

**Cổng cập nhật (update gate)** $\mathbf{z}_t$ — đóng vai trò gộp của cổng quên và cổng vào trong LSTM: quyết định giữ trạng thái cũ hay nhận trạng thái mới:

$$\mathbf{z}_t = \sigma\!\left(\mathbf{W}_{xz}\mathbf{x}_t + \mathbf{W}_{hz}\mathbf{h}_{t-1} + \mathbf{b}_z\right)$$

**Cổng đặt lại (reset gate)** $\mathbf{r}_t$ — quyết định bỏ qua bao nhiêu phần trạng thái cũ khi tính ứng viên:

$$\mathbf{r}_t = \sigma\!\left(\mathbf{W}_{xr}\mathbf{x}_t + \mathbf{W}_{hr}\mathbf{h}_{t-1} + \mathbf{b}_r\right)$$

## 5.2. Ứng viên và cập nhật trạng thái

Trạng thái ẩn ứng viên, trong đó cổng reset lọc trạng thái cũ trước khi đưa vào $\tanh$:

$$\tilde{\mathbf{h}}_t = \tanh\!\left(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{r}_t \odot (\mathbf{W}_{hh}\mathbf{h}_{t-1}) + \mathbf{b}_h\right)$$

Trạng thái mới là **nội suy lồi (convex interpolation)** giữa cũ và mới, điều khiển bởi $\mathbf{z}_t$:

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

Vẻ đẹp của công thức này nằm ở chỗ một cổng duy nhất điều khiển cả "nhớ" lẫn "ghi":

- $\mathbf{z}_t \to 0$: $\mathbf{h}_t \approx \mathbf{h}_{t-1}$ — **sao chép trạng thái cũ**, bỏ qua đầu vào. Đây cũng là đường truyền cộng tạo lối tắt cho gradient, đúng tinh thần LSTM.
- $\mathbf{z}_t \to 1$: $\mathbf{h}_t \approx \tilde{\mathbf{h}}_t$ — **cập nhật hoàn toàn** theo nội dung mới.
- $\mathbf{r}_t \to 0$: ứng viên gần như quên hết quá khứ, chỉ dựa vào $\mathbf{x}_t$ — hữu ích khi bắt đầu một "đoạn" mới trong chuỗi.

Cũng như LSTM, hệ số nhân với $\mathbf{h}_{t-1}$ là $(1 - \mathbf{z}_t)$ do mạng tự học; khi nó gần $1$, gradient chảy qua nhiều bước mà không triệt tiêu.

---

# 6. So sánh LSTM vs GRU

Cả hai giải cùng một bài toán (phụ thuộc xa) bằng cùng một ý tưởng (cổng + đường truyền cộng), khác nhau ở **mức độ phức tạp**.

| Tiêu chí | LSTM | GRU |
| --- | --- | --- |
| Số cổng | 3 ($\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t$) | 2 ($\mathbf{r}_t, \mathbf{z}_t$) |
| Trạng thái truyền đi | 2 ($\mathbf{c}_t$ và $\mathbf{h}_t$) | 1 ($\mathbf{h}_t$) |
| Ô nhớ riêng | có (cell state $\mathbf{c}_t$) | không |
| Số tham số | nhiều hơn (~$4$ khối trọng số) | ít hơn (~$3$ khối) |
| Tốc độ huấn luyện | chậm hơn | nhanh hơn |
| Khả năng biểu diễn | linh hoạt hơn (tách nhớ/đọc) | gọn, đủ dùng nhiều trường hợp |
| Khi nào dùng | chuỗi rất dài, dữ liệu lớn, cần tối đa độ chính xác | dữ liệu vừa/nhỏ, cần huấn luyện nhanh, ít tham số |

Vài kết luận thực hành:

- **Hiệu năng** thường ngang nhau trên đa số tác vụ; không có người thắng tuyệt đối. Hãy thử cả hai nếu có thể.
- **GRU** rẻ hơn về bộ nhớ và tính toán (ít cổng, một trạng thái), thường hội tụ nhanh hơn trên dữ liệu ít.
- **LSTM** tách bạch bộ nhớ dài hạn ($\mathbf{c}_t$) và đầu ra ngắn hạn ($\mathbf{h}_t$), nên đôi khi nhỉnh hơn trên các phụ thuộc cực dài hoặc dữ liệu phong phú.

---

# 7. Tổng kết

Vanilla RNN thất bại ở phụ thuộc xa không phải vì thiếu năng lực biểu diễn, mà vì **gradient của nó là một tích bị nén liên tục** qua thời gian (mục 1). LSTM và GRU sửa đúng cơ chế đó:

$$\underbrace{\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t}_{\text{LSTM}} \qquad\qquad \underbrace{\mathbf{h}_t = (1 - \mathbf{z}_t)\odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t}_{\text{GRU}}$$

Cả hai đều dựa trên cùng hai trụ cột: **cổng sigmoid** để học khi nào nhớ/quên/đọc, và **đường truyền cộng** để mở lối tắt cho gradient (constant error carousel). LSTM dùng ba cổng cùng một ô nhớ riêng; GRU rút gọn còn hai cổng và một trạng thái — đơn giản hơn mà thường không kém hiệu quả.

Những kiến trúc có cổng này từng là xương sống của xử lý chuỗi suốt nhiều năm: dịch máy, nhận dạng tiếng nói, mô hình ngôn ngữ. Bước tiếp theo là ghép chúng lại thành một mạng đọc một chuỗi và sinh ra một chuỗi khác — kiến trúc **mã hóa–giải mã** trong bài [Seq2Seq](#/seq2seq), nền tảng dẫn tới cơ chế attention và Transformer.

> Một câu để nhớ: RNN thường *nhân* trạng thái qua thời gian nên gradient chết; LSTM và GRU *cộng* trạng thái qua thời gian dưới sự điều khiển của các cổng, nên trí nhớ — và gradient — sống sót đường dài.
