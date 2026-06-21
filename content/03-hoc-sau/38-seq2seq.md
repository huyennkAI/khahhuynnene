# Mô hình chuỗi-sang-chuỗi (Sequence-to-Sequence) & Encoder–Decoder

> Rất nhiều bài toán quan trọng có dạng **chuỗi vào, chuỗi ra** với độ dài khác nhau: dịch máy (machine translation), tóm tắt, hỏi đáp, sinh chú thích ảnh. Kiến trúc **encoder–decoder** ra đời để xử lý đúng dạng ánh xạ này.
>
> $$\text{chuỗi nguồn } (x_1, \dots, x_T) \;\longrightarrow\; \text{chuỗi đích } (y_1, \dots, y_{T'})$$
>
> Đây là cây cầu nối từ [RNN](#/rnn), [LSTM/GRU](#/lstm-gru) tới cơ chế [Attention](#/attention) và Transformer.

---

# 1. Vì sao cần một kiến trúc riêng?

Một RNN thông thường ánh xạ chuỗi vào thành chuỗi ra **cùng độ dài**: mỗi bước thời gian một đầu ra. Nhưng dịch một câu tiếng Anh 5 từ sang tiếng Việt 8 từ thì độ dài nguồn $T$ và đích $T'$ **khác nhau**, và sự tương ứng giữa các vị trí cũng không thẳng hàng (từ thứ nhất nguồn không nhất thiết sinh ra từ thứ nhất đích).

Ta cần một mô hình:

- Đọc **toàn bộ** chuỗi nguồn trước khi bắt đầu sinh.
- Sinh chuỗi đích có độ dài **tùy ý**, tự quyết định khi nào dừng.
- Mô hình hóa được xác suất có điều kiện của cả câu đích:

$$P(y_1, \dots, y_{T'} \mid x_1, \dots, x_T)$$

Lời giải kinh điển (Sutskever và cộng sự, 2014; Cho và cộng sự, 2014) là tách mô hình làm hai phần: một **encoder** đọc và nén nguồn, một **decoder** sinh đích.

---

# 2. Kiến trúc encoder–decoder tổng quát

Ý tưởng cốt lõi là phân rã ánh xạ chuỗi-sang-chuỗi thành hai giai đoạn qua một trạng thái trung gian gọi là **vector ngữ cảnh** (context vector) $\mathbf{c}$.

**Bộ mã hóa (encoder)** nhận chuỗi nguồn và nén nó thành biểu diễn cố định:

$$\mathbf{c} = f_{\text{enc}}(x_1, \dots, x_T)$$

**Bộ giải mã (decoder)** nhận $\mathbf{c}$ và sinh chuỗi đích từng phần tử:

$$y_t = f_{\text{dec}}(\mathbf{c},\, y_1, \dots, y_{t-1})$$

Điểm đẹp của kiểu kiến trúc này là **trừu tượng hóa**: encoder và decoder có thể là bất kỳ mạng nào (RNN, CNN, hay sau này là Transformer), miễn là chúng giao tiếp qua $\mathbf{c}$. Bài này tập trung vào hiện thân RNN/LSTM kinh điển vì nó làm lộ rõ nút thắt dẫn tới Attention.

---

# 3. Seq2Seq với RNN/LSTM cho dịch máy

## 3.1. Encoder nén chuỗi nguồn

Encoder là một RNN (thường là [LSTM/GRU](#/lstm-gru) để nhớ phụ thuộc xa). Nó đọc lần lượt từng token nguồn, cập nhật trạng thái ẩn:

$$\mathbf{h}_t = g(\mathbf{h}_{t-1},\, \mathbf{x}_t), \qquad t = 1, \dots, T$$

trong đó $\mathbf{x}_t$ là embedding của token nguồn thứ $t$. Sau khi đọc hết câu, **trạng thái ẩn cuối cùng** được lấy làm vector ngữ cảnh:

$$\mathbf{c} = \mathbf{h}_T$$

Toàn bộ ý nghĩa của câu nguồn — dù dài bao nhiêu — bị ép vào một vector $\mathbf{c}$ có **số chiều cố định**. Hãy ghi nhớ điểm này; nó chính là nút thắt ở mục 7.

## 3.2. Decoder sinh chuỗi đích từng bước

Decoder là một RNN khác, khởi tạo trạng thái ẩn từ $\mathbf{c}$. Tại mỗi bước, nó nhận token đích trước đó và sinh phân phối trên từ vựng đích:

$$\mathbf{s}_t = g'(\mathbf{s}_{t-1},\, \mathbf{y}_{t-1},\, \mathbf{c}), \qquad \hat{P}(y_t \mid \cdot) = \text{softmax}(\mathbf{W}\mathbf{s}_t + \mathbf{b})$$

Quá trình bắt đầu bằng token đặc biệt `<bos>` (begin of sequence) và kết thúc khi decoder sinh ra `<eos>` (end of sequence). Nhờ token `<eos>`, mô hình **tự quyết định độ dài** chuỗi đích.

## 3.3. Phân rã xác suất bằng quy tắc nhân

Toàn bộ mô hình định nghĩa xác suất câu đích theo lối **tự hồi quy** (autoregressive), dùng quy tắc nhân xác suất:

$$P(y_1, \dots, y_{T'} \mid \mathbf{c}) = \prod_{t=1}^{T'} P(y_t \mid y_1, \dots, y_{t-1},\, \mathbf{c})$$

Mỗi thừa số là một bước softmax của decoder. Đây là lý do decoder phải sinh **tuần tự**: token sau phụ thuộc mọi token trước.

---

# 4. Huấn luyện: Teacher Forcing

Có một câu hỏi tinh tế khi huấn luyện: ở mỗi bước, decoder cần token đích trước đó $y_{t-1}$ làm đầu vào. Ta nên dùng token **mô hình tự đoán** hay token **thật** từ dữ liệu?

**Teacher forcing** chọn token thật: luôn nạp $y_{t-1}$ là nhãn đúng (ground truth), bất kể mô hình đã dự đoán gì ở bước trước.

- **Lợi ích**: huấn luyện ổn định và hội tụ nhanh, vì sai lầm ở một bước không bị nhân lên (propagate) sang các bước sau. Mọi bước được giám sát bằng đầu vào đúng.
- **Cái giá**: lệch phân phối khi suy luận (exposure bias) — lúc suy luận mô hình phải tự ăn đầu ra của chính nó, vốn có thể chứa lỗi mà nó chưa từng gặp lúc huấn luyện.

Trên thực tế teacher forcing vẫn là lựa chọn mặc định nhờ tốc độ; một số biến thể (scheduled sampling) trộn ngẫu nhiên token thật và token dự đoán để giảm exposure bias.

---

# 5. Hàm mất mát có mặt nạ (masked loss)

Các câu trong một lô (batch) có độ dài khác nhau, nên ta **đệm** (padding) chúng về cùng độ dài bằng token `<pad>`. Nhưng các token đệm này **vô nghĩa** — nếu tính chúng vào mất mát, mô hình sẽ học cách "dự đoán pad", làm hỏng tín hiệu học.

Giải pháp là một **mặt nạ** (mask) $m_t \in \{0, 1\}$, bằng $0$ tại các vị trí đệm. Hàm mất mát cross-entropy được nhân với mặt nạ:

$$L = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T'_{\max}} m_t \sum_{c} y_{t,c} \log \hat{y}_{t,c}$$

Chỉ những bước thời gian **thật** mới đóng góp vào gradient. Đây là chi tiết kỹ thuật nhỏ nhưng thiết yếu để seq2seq huấn luyện đúng trên dữ liệu độ dài thay đổi.

```python
def masked_loss(logits, targets, valid_len):
    # logits: (batch, T', vocab); targets: (batch, T')
    mask = (torch.arange(targets.size(1))[None, :]
            < valid_len[:, None]).float()          # 1 ở token thật, 0 ở pad
    ce = F.cross_entropy(logits.transpose(1, 2), targets,
                         reduction='none')          # (batch, T')
    return (ce * mask).sum() / mask.sum()
```

---

# 6. Suy luận: Greedy Search vs Beam Search

Khi huấn luyện ta có câu đích thật. Khi suy luận thì không — phải **tự sinh** câu đích sao cho xác suất

$$\prod_{t=1}^{T'} P(y_t \mid y_{<t}, \mathbf{c})$$

càng lớn càng tốt. Tìm chuỗi tối ưu tuyệt đối là bất khả thi: số chuỗi đích có thể có là $|\mathcal{V}|^{T'}$ — bùng nổ tổ hợp. Ta cần thuật toán tìm kiếm xấp xỉ.

## 6.1. Tìm kiếm tham lam (greedy search)

Cách đơn giản nhất: mỗi bước chọn token có xác suất **cao nhất ngay tại bước đó**:

$$y_t = \arg\max_{y \in \mathcal{V}} P(y \mid y_{<t}, \mathbf{c})$$

Chi phí thấp ($T'$ bước, mỗi bước một lần softmax), nhưng **không tối ưu**. Lý do: chọn tối ưu cục bộ tại mỗi bước không đảm bảo tích xác suất toàn chuỗi lớn nhất.

> **Ví dụ vì sao greedy hỏng.** Giả sử ở bước 1, token A có xác suất $0.5$, token B có $0.4$. Greedy chọn A. Nhưng nếu chọn B thì bước 2 lại có một token xác suất $0.9$, trong khi sau A token tốt nhất chỉ $0.3$. Đường qua B cho $0.4 \times 0.9 = 0.36$, lớn hơn đường qua A là $0.5 \times 0.3 = 0.15$. Greedy đã bỏ lỡ vì nó **không nhìn xa**.

## 6.2. Tìm kiếm chùm (beam search)

Beam search là điểm cân bằng giữa greedy (rẻ nhưng thiển cận) và tìm kiếm vét cạn (tối ưu nhưng không khả thi). Ý tưởng: giữ lại $k$ **chuỗi ứng viên** tốt nhất ở mỗi bước (gọi $k$ là **bề rộng chùm**, beam width), thay vì chỉ $1$ như greedy.

Tại mỗi bước, từ $k$ ứng viên hiện có, mở rộng mỗi ứng viên bằng mọi token khả dĩ, rồi giữ lại $k$ chuỗi có log-xác suất tích lũy cao nhất:

$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, \mathbf{c})$$

(Dùng tổng log thay vì tích xác suất để tránh tràn số dưới — underflow.)

**Chuẩn hóa độ dài.** Tổng log luôn âm và càng dài càng nhỏ, nên beam search ngây thơ thiên vị câu **ngắn**. Người ta chia điểm cho $T'^{\alpha}$ (thường $\alpha \approx 0.75$) để công bằng giữa các độ dài.

| Thuật toán | Số ứng viên giữ lại | Chi phí | Chất lượng |
| --- | --- | --- | --- |
| Greedy | $1$ | thấp nhất | dễ kẹt tối ưu cục bộ |
| Beam ($k$) | $k$ | $\sim k$ lần greedy | tốt, cân bằng |
| Vét cạn | $|\mathcal{V}|^{T'}$ | bất khả thi | tối ưu tuyệt đối |

Khi $k = 1$ beam search **suy biến** thành greedy; khi $k = |\mathcal{V}|^{T'}$ nó thành vét cạn. Thực tế $k = 4$ đến $10$ đã cho chất lượng tốt với chi phí chấp nhận được.

---

# 7. Đánh giá: BLEU (nhắc ngắn)

Làm sao chấm điểm một bản dịch tự động? **BLEU** (Bilingual Evaluation Understudy) so khớp các cụm $n$-gram giữa câu sinh và câu tham chiếu (reference):

$$\text{BLEU} = \underbrace{\text{BP}}_{\text{phạt câu ngắn}} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

trong đó $p_n$ là độ chính xác $n$-gram (tỉ lệ $n$-gram của câu sinh khớp với tham chiếu), và $\text{BP}$ là **brevity penalty** trừng phạt câu dịch quá ngắn. BLEU cao thì gần với bản dịch người, nhưng nó chỉ đo trùng lặp bề mặt nên không nắm được ngữ nghĩa sâu — vẫn cần đánh giá người bổ sung.

---

# 8. Nút thắt: một vector cố định cho cả câu

Hãy quay lại điểm đã đánh dấu ở mục 3.1. Toàn bộ thiết kế seq2seq cơ bản đặt cược vào một giả định:

> Mọi thông tin của câu nguồn — dài bao nhiêu cũng vậy — có thể nén trọn vào **một vector ngữ cảnh $\mathbf{c} = \mathbf{h}_T$ có số chiều cố định**.

Giả định này vỡ với **câu dài**. Một vector vài trăm chiều đơn giản không đủ dung lượng để chứa mọi sắc thái của một câu 40–50 từ. Hệ quả quan sát được:

- Chất lượng dịch **tụt mạnh khi câu dài ra** — đường cong BLEU giảm rõ rệt theo độ dài nguồn.
- Có **nút cổ chai thông tin** (information bottleneck): mọi token đích phải nhìn câu nguồn qua đúng một vector duy nhất, không thể "ngó lại" từng phần nguồn.
- Thông tin **đầu câu nguồn bị phai** (forgetting) khi đi qua nhiều bước RNN tới $\mathbf{h}_T$, ngay cả khi dùng [LSTM/GRU](#/lstm-gru).

Trực giác con người thì khác hẳn: khi dịch một câu dài, ta không đọc một lần rồi nhắm mắt viết — ta **liên tục liếc lại** đúng phần nguồn liên quan đến từ đang dịch.

> Đây chính là **động cơ cho cơ chế chú ý (attention)**: thay vì ép câu nguồn vào một vector cố định, hãy cho decoder ở **mỗi bước** truy cập **toàn bộ** các trạng thái ẩn nguồn $\mathbf{h}_1, \dots, \mathbf{h}_T$ và **tự chọn** phần nào liên quan nhất. Vector ngữ cảnh trở thành **động** — khác nhau ở mỗi bước sinh.

Đó là nội dung của bài tiếp theo về [Attention](#/attention), trực tiếp gỡ nút thắt vừa nêu.

---

# 9. Hạn chế → động cơ cho Attention

Tóm lại các giới hạn của seq2seq encoder–decoder thuần RNN:

- **Nút cổ chai vector cố định**: nén cả câu vào một $\mathbf{c}$ làm mất thông tin với câu dài (mục 8).
- **Phai thông tin đường xa**: trạng thái đầu chuỗi nguồn khó sống sót tới $\mathbf{h}_T$.
- **Sinh tuần tự**: decoder không song song hóa được theo bước thời gian, suy luận chậm.
- **Exposure bias**: lệch giữa teacher forcing lúc huấn luyện và tự sinh lúc suy luận (mục 4).

Hai hạn chế đầu được [Attention](#/attention) giải quyết bằng vector ngữ cảnh động; hạn chế "sinh tuần tự" sẽ được Transformer (self-attention) xử lý triệt để hơn.

---

# 10. Tổng kết

Kiến trúc **encoder–decoder** phân rã bài toán chuỗi-sang-chuỗi thành: nén nguồn thành ngữ cảnh, rồi sinh đích tự hồi quy:

$$P(y_1, \dots, y_{T'} \mid x_{1:T}) = \prod_{t=1}^{T'} P(y_t \mid y_{<t},\, \mathbf{c})$$

- **Encoder** (RNN/[LSTM-GRU](#/lstm-gru)) đọc nguồn, lấy $\mathbf{c} = \mathbf{h}_T$; **decoder** sinh đích từng bước tới `<eos>`.
- **Huấn luyện** dùng teacher forcing và **mất mát có mặt nạ** để xử lý độ dài thay đổi.
- **Suy luận** dùng **beam search** thay greedy để cân bằng chất lượng và chi phí; chấm điểm bằng **BLEU**.
- **Nút thắt** cốt lõi: một vector ngữ cảnh cố định không tải nổi câu dài.

> Chính nút thắt này mở đường cho bước tiến lớn tiếp theo của xử lý chuỗi: [cơ chế Attention](#/attention) cho phép decoder nhìn lại toàn bộ nguồn ở mỗi bước, và từ đó là Transformer — nền tảng của mọi mô hình ngôn ngữ hiện đại.
