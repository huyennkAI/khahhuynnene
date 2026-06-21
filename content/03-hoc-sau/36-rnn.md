# Mạng nơ-ron hồi quy (Recurrent Neural Networks - RNN)

> Mạng nơ-ron hồi quy (recurrent neural network) sinh ra để xử lý **dữ liệu chuỗi (sequential data)** — văn bản, âm thanh, chuỗi thời gian — nơi thứ tự quyết định ý nghĩa.
>
> Ý tưởng cốt lõi nằm gọn trong một vòng lặp: giữ một **trạng thái ẩn (hidden state)** tóm tắt toàn bộ quá khứ, rồi cập nhật nó tại mỗi bước thời gian.
>
> $$\mathbf{h}_{t-1} \;\xrightarrow{\;\mathbf{x}_t\;}\; \mathbf{h}_t \;\xrightarrow{\;\mathbf{x}_{t+1}\;}\; \mathbf{h}_{t+1} \;\rightarrow\; \cdots$$
>
> Đây là nền tảng của LSTM, GRU và toàn bộ kiến trúc xử lý chuỗi trước thời Transformer.

---

# 1. Vì sao mạng truyền thẳng không đủ?

Một mạng nơ-ron truyền thẳng (feedforward network) ánh xạ một đầu vào có **kích thước cố định** sang một đầu ra. Nhưng chuỗi thì có độ dài thay đổi, và quan trọng hơn, các phần tử trong chuỗi **phụ thuộc lẫn nhau theo thứ tự**.

Hãy xét bài toán dự đoán từ tiếp theo trong câu "mây đen kéo đến, trời sắp ___". Để đoán đúng "mưa", mô hình phải nhớ những từ đứng trước. Một mạng truyền thẳng nhìn mỗi từ độc lập sẽ vứt bỏ ngữ cảnh đó.

Trong bài [Mô hình chuỗi](#/mo-hinh-chuoi) ta đã thấy giả định Markov bậc $n$ — chỉ nhìn $n$ phần tử gần nhất — bị giới hạn cứng về tầm nhìn quá khứ. RNN gỡ bỏ giới hạn này bằng một trực giác đơn giản:

> Thay vì cố nhồi cả quá khứ vào đầu vào, hãy nén quá khứ vào một **vec-tơ trạng thái** và mang nó theo qua từng bước.

Trạng thái ẩn $\mathbf{h}_t$ đóng vai trò "bộ nhớ": nó là bản tóm tắt của tất cả những gì mạng đã thấy từ đầu chuỗi tới bước $t$. Mỗi khi gặp một đầu vào mới, mạng trộn nó với bộ nhớ cũ để tạo ra bộ nhớ mới.

---

# 2. Công thức hồi quy

## 2.1. Cập nhật trạng thái ẩn

Giả sử tại bước thời gian $t$ ta có đầu vào $\mathbf{x}_t \in \mathbb{R}^{d}$ và trạng thái ẩn trước đó $\mathbf{h}_{t-1} \in \mathbb{R}^{h}$. Trạng thái ẩn mới được tính bằng:

$$\mathbf{h}_t = \phi\!\left(\mathbf{W}_{hh}\,\mathbf{h}_{t-1} + \mathbf{W}_{xh}\,\mathbf{x}_t + \mathbf{b}_h\right)$$

trong đó:

- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$ chiếu đầu vào mới vào không gian ẩn,
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ truyền trạng thái cũ sang trạng thái mới (đây là **vòng hồi quy**),
- $\mathbf{b}_h \in \mathbb{R}^{h}$ là độ chệch (bias),
- $\phi$ là hàm kích hoạt phi tuyến, thường là $\tanh$.

Chính số hạng $\mathbf{W}_{hh}\,\mathbf{h}_{t-1}$ làm nên "tính hồi quy": trạng thái hiện tại được định nghĩa **đệ quy** theo trạng thái trước. Khai triển ra, $\mathbf{h}_t$ là một hàm hợp của toàn bộ $\mathbf{x}_1, \dots, \mathbf{x}_t$.

## 2.2. Lớp đầu ra

Từ trạng thái ẩn, ta sinh đầu ra (chẳng hạn phân phối trên từ vựng):

$$\mathbf{o}_t = \mathbf{W}_{hq}\,\mathbf{h}_t + \mathbf{b}_q$$

với $\mathbf{W}_{hq} \in \mathbb{R}^{q \times h}$ và $\mathbf{b}_q \in \mathbb{R}^{q}$. Trong bài toán phân loại (như mô hình ngôn ngữ), ta đưa $\mathbf{o}_t$ qua softmax để có xác suất:

$$\hat{\mathbf{y}}_t = \operatorname{softmax}(\mathbf{o}_t)$$

## 2.3. Vì sao đây là một mạng "sâu theo thời gian"

Mở vòng lặp (unrolling) ra theo thời gian, RNN trông như một mạng truyền thẳng rất sâu, mỗi lớp ứng với một bước thời gian:

$$\mathbf{x}_1 \to \boxed{\mathbf{h}_1} \to \boxed{\mathbf{h}_2} \to \cdots \to \boxed{\mathbf{h}_T} \to \mathbf{o}_T$$

Điểm khác biệt mấu chốt so với mạng sâu thông thường: **mọi lớp dùng chung một bộ trọng số** $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{b}_h$. Đây chính là chia sẻ trọng số theo thời gian, chủ đề của mục tiếp theo.

---

# 3. Chia sẻ trọng số theo thời gian

Một đặc trưng định nghĩa của RNN là **chia sẻ tham số (parameter sharing across time)**: cùng một ma trận $\mathbf{W}_{hh}, \mathbf{W}_{xh}$ được áp dụng tại mọi bước $t$.

Điều này mang lại ba lợi ích lớn:

**Xử lý chuỗi dài bất kỳ.** Vì cùng một phép biến đổi lặp lại, mô hình không bị ràng buộc bởi độ dài chuỗi. Một RNN huấn luyện trên câu 10 từ vẫn chạy được trên câu 100 từ.

**Số tham số không nở theo độ dài.** Nếu mỗi bước có ma trận riêng, số tham số sẽ phình lên tỉ lệ với $T$. Chia sẻ trọng số giữ kích thước mô hình cố định.

**Tổng quát hóa theo vị trí.** Một quy luật học được ở vị trí thứ 3 (ví dụ "sau chủ ngữ thường là động từ") tự động áp dụng cho mọi vị trí khác — giống như chia sẻ kernel trong mạng tích chập (CNN) đem lại bất biến tịnh tiến, chia sẻ trọng số theo thời gian đem lại bất biến theo thời điểm.

Cái giá của chia sẻ trọng số sẽ lộ ra ở mục 6: vì cùng một $\mathbf{W}_{hh}$ được nhân lặp đi lặp lại khi lan truyền ngược, gradient dễ bùng nổ hoặc tiêu biến.

---

# 4. Mô hình ngôn ngữ mức ký tự

Một ứng dụng kinh điển làm rõ cơ chế RNN là **mô hình ngôn ngữ mức ký tự (character-level language model)**. Nhiệm vụ: cho một chuỗi ký tự, dự đoán ký tự kế tiếp.

Xét chuỗi "machine". Tại mỗi bước, đầu vào là một ký tự (mã hóa one-hot), còn nhãn là ký tự ngay sau nó:

| Bước $t$ | Đầu vào $\mathbf{x}_t$ | Nhãn $y_t$ |
| --- | --- | --- |
| 1 | `m` | `a` |
| 2 | `a` | `c` |
| 3 | `c` | `h` |
| 4 | `h` | `i` |
| 5 | `i` | `n` |

Tại mỗi bước, mạng cập nhật $\mathbf{h}_t$ rồi xuất $\hat{\mathbf{y}}_t = \operatorname{softmax}(\mathbf{o}_t)$ — một phân phối trên toàn bộ bảng ký tự. Hàm mất mát là **cross-entropy** trung bình trên cả chuỗi (xem [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)):

$$L = -\frac{1}{T}\sum_{t=1}^{T} \log \hat{y}_{t}\big[y_t\big]$$

trong đó $\hat{y}_t[y_t]$ là xác suất mô hình gán cho ký tự đúng. Để đánh giá, người ta thường dùng **perplexity** — lũy thừa $e$ của cross-entropy trung bình — diễn giải như "số lựa chọn phân vân trung bình" tại mỗi bước; perplexity bằng 1 nghĩa là dự đoán hoàn hảo.

```python
import torch, torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.W_xh = nn.Linear(vocab, hidden, bias=False)
        self.W_hh = nn.Linear(hidden, hidden)       # vong hoi quy
        self.W_hq = nn.Linear(hidden, vocab)        # lop dau ra

    def forward(self, xs, h):
        outs = []
        for x in xs:                                # lap qua tung buoc thoi gian
            h = torch.tanh(self.W_xh(x) + self.W_hh(h))
            outs.append(self.W_hq(h))               # o_t
        return torch.stack(outs), h
```

Lưu ý cùng `self.W_hh` được tái sử dụng cho mọi `x` trong vòng lặp — đó chính là chia sẻ trọng số theo thời gian được hiện thực hóa.

---

# 5. Lan truyền ngược theo thời gian (BPTT)

Để huấn luyện, ta cần gradient của hàm mất mát theo các trọng số. Vì RNN là một mạng sâu khi mở vòng lặp, ta áp dụng [lan truyền ngược](#/backprop) trên đồ thị đã mở — gọi là **lan truyền ngược theo thời gian (Backpropagation Through Time - BPTT)**.

## 5.1. Thiết lập

Để đơn giản, bỏ qua bias và phi tuyến tại lớp ẩn, viết gọn tổng mất mát:

$$L = \sum_{t=1}^{T} \ell_t, \qquad \ell_t = \ell(\mathbf{o}_t, y_t)$$

Vì $\mathbf{W}_{hh}$ tham gia vào **mọi** trạng thái ẩn $\mathbf{h}_1, \dots, \mathbf{h}_T$, gradient của nó là tổng đóng góp từ tất cả các bước:

$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \mathbf{W}_{hh}}$$

## 5.2. Lan truyền dọc theo chuỗi trạng thái ẩn

Mấu chốt là tính $\dfrac{\partial \ell_t}{\partial \mathbf{h}_k}$ cho $k \le t$, tức ảnh hưởng của trạng thái $\mathbf{h}_k$ ở quá khứ lên mất mát $\ell_t$ ở hiện tại. Theo quy tắc chuỗi, ta phải đi ngược từ $\mathbf{h}_t$ về $\mathbf{h}_k$ qua từng bước trung gian:

$$\frac{\partial \ell_t}{\partial \mathbf{h}_k} = \frac{\partial \ell_t}{\partial \mathbf{h}_t} \cdot \prod_{j=k+1}^{t} \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$$

Từ công thức hồi quy $\mathbf{h}_j = \phi(\mathbf{W}_{hh}\mathbf{h}_{j-1} + \mathbf{W}_{xh}\mathbf{x}_j + \mathbf{b}_h)$, đạo hàm mỗi mắt xích là:

$$\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \operatorname{diag}\!\big(\phi'(\cdot)\big)\,\mathbf{W}_{hh}^{\top}$$

Thay vào, ta thấy gradient chứa một **tích của $(t-k)$ ma trận**, mỗi ma trận đều có thừa số $\mathbf{W}_{hh}^{\top}$:

$$\frac{\partial \ell_t}{\partial \mathbf{h}_k} = \frac{\partial \ell_t}{\partial \mathbf{h}_t}\prod_{j=k+1}^{t} \operatorname{diag}\!\big(\phi'(\cdot)\big)\,\mathbf{W}_{hh}^{\top}$$

> Đây là điểm then chốt: tín hiệu gradient từ bước $t$ muốn về tới bước $k$ phải đi qua **lũy thừa của $\mathbf{W}_{hh}$**. Chính phép nhân lặp này quyết định số phận của RNN.

---

# 6. Vanishing và exploding gradients

## 6.1. Suy dẫn trực giác

Hãy xấp xỉ tích các ma trận bằng một đại lượng vô hướng để thấy bản chất. Gọi $\lambda$ là trị riêng trội (giá trị riêng lớn nhất theo trị tuyệt đối) của $\mathbf{W}_{hh}$. Khi đó độ lớn của tích $(t-k)$ thừa số xấp xỉ tỉ lệ với:

$$\Big\lVert \prod_{j=k+1}^{t} \mathbf{W}_{hh}^{\top} \Big\rVert \;\sim\; |\lambda|^{\,t-k}$$

Hàm mũ này có hai chế độ tai họa khi khoảng cách $t-k$ lớn:

- **Tiêu biến (vanishing).** Nếu $|\lambda| < 1$ thì $|\lambda|^{t-k} \to 0$. Gradient từ các bước xa **co về 0**, mô hình gần như không học được phụ thuộc dài hạn.
- **Bùng nổ (exploding).** Nếu $|\lambda| > 1$ thì $|\lambda|^{t-k} \to \infty$. Gradient **phình vô hạn**, cập nhật nhảy loạn, mất mát hóa `NaN`.

Hàm kích hoạt còn làm trầm trọng thêm tiêu biến: với $\tanh$ ta luôn có $\phi'(\cdot) \le 1$, mỗi mắt xích nhân thêm một thừa số $\le 1$, càng kéo gradient về 0.

Đây chính là biểu hiện cực đoan của vấn đề [ổn định số](#/on-dinh-so) trong các mạng rất sâu: tích nhiều thừa số khiến đại lượng hoặc nổ tung hoặc triệt tiêu. RNN bị nặng vì độ sâu **bằng độ dài chuỗi**, có thể tới hàng trăm bước, lại còn dùng chung một $\mathbf{W}_{hh}$ nên không có chuyện "may rủi bù trừ" giữa các lớp.

## 6.2. Gradient clipping — chặn bùng nổ

Bùng nổ tuy nguy hiểm nhưng dễ chữa: chỉ cần **cắt chuẩn gradient (gradient clipping)**. Gọi $\mathbf{g}$ là vec-tơ gradient gộp toàn bộ tham số, ta co nó lại nếu chuẩn vượt ngưỡng $\theta$:

$$\mathbf{g} \leftarrow \min\!\left(1, \frac{\theta}{\lVert \mathbf{g}\rVert}\right)\mathbf{g}$$

Khi $\lVert \mathbf{g}\rVert \le \theta$ thì gradient giữ nguyên; khi vượt ngưỡng, gradient được **giữ nguyên hướng** nhưng thu độ dài về đúng $\theta$. Mẹo này giữ cho bước cập nhật không bao giờ nhảy quá xa, ngăn mất mát hóa `NaN`.

## 6.3. Truncated BPTT — chặn chi phí và tiêu biến

Lan truyền ngược qua **toàn bộ** chuỗi dài vừa tốn bộ nhớ (phải lưu mọi $\mathbf{h}_t$) vừa làm gradient tiêu biến qua quãng đường dài. Giải pháp thực dụng là **BPTT cắt ngắn (truncated BPTT)**: chỉ lan truyền ngược trong một cửa sổ $\tau$ bước gần nhất rồi cắt gradient ở đó.

$$\frac{\partial \ell_t}{\partial \mathbf{W}_{hh}} \approx \sum_{k=t-\tau}^{t} \frac{\partial \ell_t}{\partial \mathbf{h}_k}\frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

Trong khi đó, trạng thái ẩn $\mathbf{h}_t$ ở **lượt truyền xuôi** vẫn được mang theo liên tục giữa các lô (chỉ tách khỏi đồ thị tính gradient). Cách này đánh đổi một chút thiên lệch (bỏ phụ thuộc xa hơn $\tau$) lấy chi phí ổn định và gradient ít nhiễu.

---

# 7. Vì sao cần LSTM và GRU?

Gradient clipping và truncated BPTT chỉ chữa được **bùng nổ** và chi phí. Vấn đề **tiêu biến** — nguyên nhân khiến RNN không nắm được phụ thuộc dài hạn (ví dụ chủ ngữ ở đầu câu quyết định động từ ở cuối câu) — vẫn còn nguyên, vì nó nằm ở cấu trúc nhân lặp $\mathbf{W}_{hh}$.

Lời giải triệt để không nằm ở thuật toán huấn luyện mà ở **kiến trúc**. Các biến thể [LSTM và GRU](#/lstm-gru) thêm vào những **cổng (gates)** và một **đường truyền trạng thái gần như cộng tuyến** (constant error carousel), cho phép gradient chảy qua nhiều bước mà không bị nhân lặp triệt tiêu. Nhờ đó chúng giữ được trí nhớ dài hạn — điều mà RNN thuần không làm nổi.

---

# 8. Ưu điểm

* **Xử lý chuỗi độ dài bất kỳ** — cùng một vòng hồi quy chạy cho mọi độ dài, không cần cố định kích thước đầu vào.
* **Tham số gọn nhờ chia sẻ trọng số** — số tham số không nở theo $T$, và quy luật học được tổng quát hóa qua mọi vị trí thời gian.
* **Bộ nhớ tóm tắt quá khứ** — trạng thái ẩn $\mathbf{h}_t$ nén toàn bộ tiền sử thành một vec-tơ kích thước cố định, vượt giới hạn cứng của giả định Markov.
* **Nền tảng tư duy** — toàn bộ ý tưởng "mang trạng thái qua thời gian" là gốc rễ của LSTM, GRU và cả cơ chế attention sau này.

---

# 9. Hạn chế

Chính cấu trúc nhân lặp $\mathbf{W}_{hh}$ vừa là sức mạnh vừa là gót chân Achilles của RNN.

**Gradient tiêu biến/bùng nổ.** Như mục 6 chứng minh, gradient chứa lũy thừa của $\mathbf{W}_{hh}$, nên phụ thuộc dài hạn rất khó học. Đây là hạn chế cốt tử, chỉ giải quyết được phần bùng nổ bằng clipping.

**Tính tuần tự cản trở song song hóa.** $\mathbf{h}_t$ phụ thuộc $\mathbf{h}_{t-1}$, nên không thể tính các bước thời gian đồng thời như Transformer — huấn luyện trên chuỗi dài chậm.

**Trí nhớ hữu hạn và bị pha loãng.** Toàn bộ quá khứ bị ép vào một vec-tơ cố định; thông tin xa dần bị các cập nhật mới ghi đè.

---

# 10. Tổng kết

RNN khởi đầu từ một trực giác mộc mạc: **mang một trạng thái tóm tắt quá khứ đi qua từng bước thời gian**, với cùng một bộ trọng số tái sử dụng khắp chuỗi.

$$\mathbf{h}_t = \phi\!\left(\mathbf{W}_{hh}\,\mathbf{h}_{t-1} + \mathbf{W}_{xh}\,\mathbf{x}_t + \mathbf{b}_h\right), \qquad \mathbf{o}_t = \mathbf{W}_{hq}\,\mathbf{h}_t + \mathbf{b}_q$$

Hai hệ quả lớn của thiết kế này đan vào nhau. Chia sẻ trọng số cho phép xử lý chuỗi dài tùy ý với ít tham số; nhưng cùng cơ chế ấy, khi lan truyền ngược theo thời gian, lại sinh ra **tích các $\mathbf{W}_{hh}$** dẫn tới vanishing/exploding gradients. Gradient clipping và truncated BPTT vá được phần bùng nổ và chi phí, nhưng tiêu biến vẫn là rào cản với phụ thuộc dài hạn.

Đó là lý do RNN thuần ít khi được dùng trực tiếp ngày nay. Nhưng tư tưởng của nó — bộ nhớ hồi quy — sống tiếp trong các kiến trúc mạnh hơn.

> Bài tiếp theo — **LSTM & GRU** — giữ nguyên ý tưởng trạng thái hồi quy nhưng thêm các **cổng** điều tiết luồng thông tin và một đường truyền trạng thái cộng tuyến, giải quyết tận gốc bài toán gradient tiêu biến để học được phụ thuộc dài hạn.
