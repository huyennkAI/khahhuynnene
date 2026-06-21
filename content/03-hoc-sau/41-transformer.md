# Kiến trúc Transformer

> Transformer (giới thiệu năm 2017 trong bài báo *"Attention Is All You Need"*) vứt bỏ hoàn toàn hồi quy (recurrence) và tích chập (convolution), chỉ giữ lại **cơ chế chú ý (attention)**. Nhờ đó nó vừa **tính toán song song** trên cả chuỗi, vừa **nắm phụ thuộc xa (long-range dependency)** trong một bước duy nhất.
>
> $$\text{Đầu vào} \;\rightarrow\; \underbrace{[\,\text{Encoder}\,]^{N}}_{\text{hiểu}} \;\rightarrow\; \underbrace{[\,\text{Decoder}\,]^{N}}_{\text{sinh}} \;\rightarrow\; \text{Đầu ra}$$
>
> Đây là viên gạch nền của toàn bộ kỷ nguyên mô hình ngôn ngữ lớn (LLM): GPT, BERT, Claude đều là Transformer.

---

# 1. Động cơ ra đời: vì sao phải bỏ hồi quy?

Trước Transformer, các kiến trúc mạnh nhất cho chuỗi đều dựa trên hồi quy: RNN, LSTM, GRU, thường lắp trong khung [seq2seq](#/seq2seq) encoder–decoder. Chúng xử lý chuỗi **từng bước một**: trạng thái ẩn $\mathbf{h}_t$ phụ thuộc vào $\mathbf{h}_{t-1}$.

Chính tính tuần tự đó sinh ra hai điểm nghẽn cố hữu.

**Không song song hóa được.** Vì $\mathbf{h}_t$ cần $\mathbf{h}_{t-1}$, ta buộc phải tính tuần tự $T$ bước cho một chuỗi dài $T$. GPU vốn giỏi nhân ma trận song song lại phải đứng chờ — huấn luyện chậm, không tận dụng được phần cứng.

**Phụ thuộc xa bị suy yếu.** Thông tin từ token đầu muốn ảnh hưởng tới token cuối phải đi qua $T$ bước trung gian. Mỗi bước nhân thêm trọng số và đi qua hàm phi tuyến, khiến gradient **tiêu biến hoặc bùng nổ (vanishing/exploding gradient)**. Quãng đường giữa hai token cách xa nhau tỉ lệ với $O(T)$.

Trực giác của Transformer đến từ một câu hỏi:

> Nếu mục tiêu là cho mỗi token "nhìn thấy" mọi token khác, vì sao phải truyền tin qua từng bước tuần tự? Hãy cho **mọi cặp token kết nối trực tiếp** trong một phép tính ma trận duy nhất.

Đặt cạnh nhau để thấy rõ điều Transformer thay đổi:

| Tiêu chí | RNN/LSTM | Self-Attention |
| --- | --- | --- |
| Tính toán theo bước thời gian | tuần tự ($T$ bước) | song song (1 bước ma trận) |
| Quãng đường giữa hai token bất kỳ | $O(T)$ | $O(1)$ |
| Phép tính mỗi tầng | $O(T \cdot d^2)$ | $O(T^2 \cdot d)$ |
| Phụ thuộc xa | yếu (gradient suy biến) | mạnh (kết nối trực tiếp) |

Transformer đổi chi phí bộ nhớ $O(T^2)$ lấy hai thứ vô giá: **song song hóa** và **đường truyền tin ngắn**. Trên phần cứng hiện đại, đây là một cuộc đổi chác cực kỳ có lãi.

---

# 2. Tổng quan kiến trúc encoder–decoder

Transformer giữ khung encoder–decoder của [seq2seq](#/seq2seq), nhưng thay từng tầng hồi quy bằng các **khối chú ý (attention block)** xếp chồng.

- **Bộ mã hóa (encoder)** gồm $N$ khối giống nhau xếp chồng (bản gốc $N=6$). Nó đọc toàn bộ chuỗi nguồn và biến mỗi token thành một biểu diễn giàu ngữ cảnh.
- **Bộ giải mã (decoder)** cũng gồm $N$ khối xếp chồng. Nó sinh chuỗi đích **từng token một**, mỗi bước nhìn vào những token đã sinh cộng với toàn bộ đầu ra của encoder.

Điểm mấu chốt: bên trong mỗi khối, mọi vị trí được xử lý **song song** bằng các phép nhân ma trận; tính tuần tự chỉ còn xuất hiện ở pha suy luận của decoder (mục 8).

Ba loại attention xuất hiện trong kiến trúc:

1. **Self-attention của encoder** — mỗi token nguồn chú ý tới mọi token nguồn.
2. **Masked self-attention của decoder** — mỗi token đích chỉ chú ý tới các token đích **đứng trước** nó.
3. **Cross-attention** — token đích chú ý tới toàn bộ biểu diễn của encoder.

---

# 3. Nhắc lại lõi attention

Toàn bộ Transformer được dựng trên một công thức duy nhất. Mỗi token được chiếu thành ba vector:

- **Truy vấn (query)** $\mathbf{Q}$ — "tôi đang tìm thông tin gì".
- **Khóa (key)** $\mathbf{K}$ — "tôi chứa thông tin gì để được tìm thấy".
- **Giá trị (value)** $\mathbf{V}$ — "nội dung thực sự tôi mang theo".

Chú ý theo tích vô hướng có tỉ lệ (scaled dot-product attention):

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}$$

Tích $\mathbf{Q}\mathbf{K}^\top$ cho ma trận điểm tương đồng giữa mọi cặp token; $\operatorname{softmax}$ biến mỗi hàng thành phân phối trọng số; nhân với $\mathbf{V}$ là lấy trung bình có trọng số các giá trị.

Vì sao chia cho $\sqrt{d_k}$? Nếu các thành phần của $\mathbf{Q}, \mathbf{K}$ độc lập, trung bình $0$, phương sai $1$, thì tích vô hướng $\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ có phương sai $d_k$. Khi $d_k$ lớn, điểm số phình to, đẩy $\operatorname{softmax}$ vào vùng bão hòa nơi gradient gần như bằng $0$. Chia cho $\sqrt{d_k}$ đưa phương sai về lại $1$, giữ gradient khỏe mạnh. Chi tiết về nhiều đầu chú ý xem bài [multi-head attention](#/multi-head-attention) và [attention](#/attention).

---

# 4. Khối encoder

Mỗi khối encoder gồm hai tầng con, mỗi tầng con được bọc bởi residual và LayerNorm.

## 4.1. Multi-head self-attention

Tầng con thứ nhất là **chú ý đa đầu (multi-head attention)** ở chế độ tự chú ý (self-attention): cả $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ đều đến từ cùng chuỗi đầu vào $\mathbf{X}$.

Thay vì một phép attention duy nhất, ta chạy $h$ "đầu" song song. Mỗi đầu chiếu đầu vào xuống một không gian con riêng rồi chú ý độc lập:

$$\operatorname{head}_i = \operatorname{Attention}(\mathbf{X}\mathbf{W}_i^Q,\; \mathbf{X}\mathbf{W}_i^K,\; \mathbf{X}\mathbf{W}_i^V)$$

$$\operatorname{MultiHead}(\mathbf{X}) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h)\,\mathbf{W}^O$$

Mỗi đầu có thể chuyên một kiểu quan hệ khác nhau — đầu này bắt cú pháp, đầu kia bắt đồng tham chiếu, đầu khác bắt vị trí. Đây là phiên bản chú ý của "nhiều bộ lọc" trong CNN. Xem sâu ở [multi-head attention](#/multi-head-attention).

Vì mọi token cùng được chiếu và nhân ma trận một lượt, tầng này **không hề tuần tự** — đây chính là chỗ Transformer ăn đứt RNN về tốc độ.

## 4.2. Mạng truyền thẳng theo vị trí (positionwise FFN)

Tầng con thứ hai là **mạng truyền thẳng theo vị trí (positionwise feed-forward network)**: một MLP hai lớp áp dụng **độc lập và giống hệt nhau** cho từng vị trí token:

$$\operatorname{FFN}(\mathbf{x}) = \max(0,\; \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2$$

Hai điểm cần nhấn mạnh.

**Dùng chung trọng số mọi vị trí.** Cùng một $\mathbf{W}_1, \mathbf{W}_2$ được áp cho mọi token trong chuỗi — giống một tích chập $1\times 1$. Nhờ vậy số tham số không phụ thuộc độ dài chuỗi, và ta xử lý được chuỗi dài tùy ý.

**Vì sao cần FFN khi đã có attention?** Attention bản chất là **trộn thông tin tuyến tính** giữa các token (trung bình có trọng số các value). FFN bổ sung **biến đổi phi tuyến theo từng token**, thường nở chiều ẩn lên $4d$ rồi nén lại. Attention quyết định *trộn thông tin từ đâu*, FFN quyết định *xử lý thông tin đó thế nào*.

## 4.3. Residual + LayerNorm (Add & Norm)

Mỗi tầng con $\operatorname{Sublayer}(\cdot)$ được bọc bởi **kết nối phần dư (residual connection)** rồi **chuẩn hóa tầng (layer normalization)**:

$$\mathbf{y} = \operatorname{LayerNorm}\!\big(\mathbf{x} + \operatorname{Sublayer}(\mathbf{x})\big)$$

Kết nối phần dư tạo một "đường cao tốc" cho gradient: đạo hàm của $\mathbf{x} + \operatorname{Sublayer}(\mathbf{x})$ luôn có thành phần hằng số $1$, nên gradient không bị tiêu biến dù mạng xếp chồng rất sâu. Đây là điều kiện sống còn để huấn luyện $N$ khối chồng lên nhau.

**Vì sao LayerNorm chứ không phải BatchNorm cho chuỗi?** [Chuẩn hóa theo lô (batch norm)](#/batch-norm) tính trung bình và phương sai **trên chiều lô (batch)**, cho từng đặc trưng:

$$\operatorname{BatchNorm}:\ \text{thống kê trên } \{\text{các mẫu trong lô}\}$$

Điều này gây ba rắc rối với chuỗi ngôn ngữ:

1. **Độ dài chuỗi thay đổi.** Các câu trong một lô dài ngắn khác nhau, phải đệm (padding); thống kê theo lô bị nhiễu bởi các vị trí đệm rỗng.
2. **Phụ thuộc kích thước lô.** Thống kê theo lô kém tin cậy khi lô nhỏ, và lệch giữa huấn luyện với suy luận (suy luận thường xử lý một câu một lúc).
3. **Suy luận tự hồi quy.** Decoder sinh từng token; không có "lô các token tương lai" để lấy thống kê.

[Chuẩn hóa tầng (layer norm)](#/batch-norm) tránh hết bằng cách tính thống kê **trên chiều đặc trưng của từng token, độc lập với lô**:

$$\operatorname{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}, \qquad \mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

Mỗi token tự chuẩn hóa bằng chính các chiều của nó, nên hoạt động đồng nhất ở mọi độ dài chuỗi, mọi kích thước lô, và ở cả huấn luyện lẫn suy luận tuần tự.

---

# 5. Khối decoder

Mỗi khối decoder có **ba** tầng con (encoder chỉ có hai), đều được bọc Add & Norm.

## 5.1. Masked self-attention

Tầng con thứ nhất là self-attention trên chuỗi đích, nhưng **có che (masked)**. Lý do thuộc về bản chất bài toán: decoder sinh văn bản **tự hồi quy (autoregressive)** — token thứ $t$ được sinh chỉ dựa trên các token $1, \dots, t-1$. Nếu để nó nhìn thấy token tương lai khi huấn luyện thì đó là **gian lận** (nó chép luôn đáp án), và mô hình sẽ vô dụng lúc suy luận.

Ta che bằng cách cộng một mặt nạ $-\infty$ vào ma trận điểm trước khi $\operatorname{softmax}$, tại mọi vị trí "tương lai":

$$\big(\mathbf{Q}\mathbf{K}^\top\big)_{ij} \;\leftarrow\; \begin{cases} \big(\mathbf{Q}\mathbf{K}^\top\big)_{ij} & j \le i \\[4pt] -\infty & j > i \end{cases}$$

Sau $\operatorname{softmax}$, $e^{-\infty} = 0$, nên token $i$ có trọng số chú ý $0$ lên mọi token đứng sau. Mẹo này cho phép huấn luyện **toàn bộ chuỗi đích song song** mà vẫn tôn trọng ràng buộc nhân quả — ta không phải sinh tuần tự lúc huấn luyện.

## 5.2. Cross-attention (encoder–decoder attention)

Tầng con thứ hai là **chú ý chéo (cross-attention)** — chính là chỗ decoder "đọc" chuỗi nguồn. Điểm khác biệt nằm ở nguồn của $\mathbf{Q}, \mathbf{K}, \mathbf{V}$:

$$\mathbf{Q} \;\text{từ decoder}, \qquad \mathbf{K},\, \mathbf{V} \;\text{từ đầu ra encoder}$$

Trực giác: mỗi token đang được sinh (query) hỏi xem nên lấy thông tin từ những token nguồn nào (key/value). Đây chính là cơ chế đối sánh nguồn–đích đã thay thế cho attention trong [seq2seq](#/seq2seq) cổ điển — nhưng giờ là đa đầu và song song.

## 5.3. Positionwise FFN

Tầng con thứ ba giống hệt FFN ở mục 4.2: một MLP hai lớp theo từng vị trí, dùng chung trọng số.

---

# 6. Mã hóa vị trí (positional encoding)

Có một lỗ hổng: self-attention là phép **bất biến hoán vị (permutation-invariant)** — đảo thứ tự token đầu vào thì đầu ra cũng chỉ đảo theo, giá trị không đổi. Vậy mô hình **không hề biết thứ tự từ**. Với ngôn ngữ, "chó cắn người" và "người cắn chó" không thể như nhau.

Cách sửa: cộng thẳng một tín hiệu vị trí vào embedding đầu vào. Bản gốc dùng hàm sin/cos với nhiều tần số:

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

trong đó $pos$ là vị trí token, $i$ là chỉ số chiều. Mỗi vị trí nhận một "dấu vân tay" duy nhất tổ hợp từ nhiều bước sóng. Một tính chất đẹp: $PE_{pos+k}$ là một biến đổi tuyến tính của $PE_{pos}$ (do công thức cộng góc của sin/cos), nên mô hình dễ học các quan hệ vị trí **tương đối**. Vì là hàm dạng đóng, nó còn ngoại suy được sang chuỗi dài hơn lúc huấn luyện.

---

# 7. Luồng huấn luyện

Khi huấn luyện (ví dụ dịch máy), ta đã có sẵn cả câu nguồn lẫn câu đích đúng. Nhờ vậy:

1. Encoder xử lý **toàn bộ** câu nguồn song song, cho ra biểu diễn ngữ cảnh.
2. Decoder nhận **toàn bộ** câu đích đúng làm đầu vào (kỹ thuật **teacher forcing**), nhưng masked self-attention (mục 5.1) đảm bảo mỗi vị trí chỉ thấy quá khứ.
3. Toàn bộ token đích được dự đoán **trong một lượt truyền xuôi (forward pass)**, so với nhãn thật bằng [hàm mất mát cross-entropy](#/ham-mat-mat-toi-uu), rồi lan truyền ngược.

Đây là khác biệt then chốt với RNN: huấn luyện không cần lặp $T$ bước tuần tự, mà song song hóa hoàn toàn trên cả chuỗi.

```python
# Sơ đồ một khối encoder (pre-norm để gọn)
def encoder_block(x):
    x = x + multi_head_attention(layer_norm(x))  # tự chú ý + residual
    x = x + ffn(layer_norm(x))                    # FFN + residual
    return x
```

---

# 8. Luồng suy luận

Lúc suy luận, ta **không có** câu đích, nên decoder buộc phải **tự hồi quy (autoregressive)**:

1. Encoder chạy **một lần duy nhất** trên câu nguồn (key/value cho cross-attention không đổi).
2. Decoder sinh token đầu tiên, nối nó vào đầu vào, sinh token thứ hai, và cứ thế lặp đến khi gặp token kết thúc.

Như vậy tính tuần tự — thứ Transformer xóa bỏ ở pha huấn luyện — **quay lại ở pha sinh** của decoder. Trên thực tế, các key/value đã tính ở bước trước được lưu lại (**KV cache**) để khỏi tính lại, giúp suy luận nhanh hơn nhiều.

---

# 9. Bảng tổng hợp các thành phần

| Thành phần | Vai trò | Có ở |
| --- | --- | --- |
| Multi-head self-attention | mỗi token trộn thông tin từ mọi token cùng chuỗi | encoder, decoder |
| Masked self-attention | như trên nhưng che token tương lai (nhân quả) | decoder |
| Cross-attention | query từ decoder hỏi key/value từ encoder | decoder |
| Positionwise FFN | biến đổi phi tuyến theo từng token, dùng chung trọng số | encoder, decoder |
| Residual + LayerNorm | đường cao tốc gradient + ổn định huấn luyện sâu | mọi tầng con |
| Positional encoding | tiêm thông tin thứ tự vào embedding | đầu vào |

---

# 10. Độ phức tạp

Với chuỗi dài $T$ và số chiều mô hình $d$:

| Tầng | Độ phức tạp/tầng | Phép tuần tự | Quãng đường tối đa |
| --- | --- | --- | --- |
| Self-attention | $O(T^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Hồi quy (RNN) | $O(T \cdot d^2)$ | $O(T)$ | $O(T)$ |

Self-attention trả giá bằng chi phí bậc hai $O(T^2)$ theo độ dài chuỗi (mọi cặp token), nhưng đổi lại số phép tuần tự chỉ còn $O(1)$ và quãng đường truyền tin tối đa cũng $O(1)$. Khi $T$ không quá lớn so với $d$ (trường hợp thường gặp), đây là đánh đổi cực có lợi. Chi phí $O(T^2)$ cũng chính là động lực cho hàng loạt biến thể attention thưa và tuyến tính về sau.

---

# 11. Ưu điểm

* **Song song hóa hoàn toàn khi huấn luyện** — không còn vòng lặp tuần tự theo thời gian như RNN, tận dụng tối đa GPU/TPU.
* **Bắt phụ thuộc xa** — mọi cặp token kết nối trực tiếp, quãng đường truyền tin $O(1)$, gradient không suy biến theo độ dài.
* **Mở rộng cực tốt (scaling)** — kiến trúc đồng nhất, dễ tăng số khối $N$, số chiều $d$, số đầu $h$; hiệu năng tăng đều theo dữ liệu và tham số (scaling laws).
* **Linh hoạt** — cùng một khối phục vụ cả hiểu (encoder) lẫn sinh (decoder).

---

# 12. Hạn chế

* **Chi phí bậc hai $O(T^2)$** theo độ dài chuỗi — bộ nhớ và tính toán tăng nhanh với văn bản dài.
* **Không có thiên kiến quy nạp về thứ tự** — phải tiêm thủ công bằng positional encoding.
* **Suy luận vẫn tuần tự** — decoder sinh từng token một, không song song hóa được pha sinh.
* **Đói dữ liệu** — vì ít thiên kiến quy nạp, Transformer cần lượng dữ liệu lớn mới phát huy.

---

# 13. Tổng kết

Transformer là một ý tưởng kiến trúc giản dị mà sâu xa: **thay hồi quy bằng chú ý**. Bằng cách cho mọi token kết nối trực tiếp qua self-attention, nó đồng thời gỡ được hai nút thắt của RNN — tính tuần tự cản trở song song hóa, và quãng đường dài làm suy yếu phụ thuộc xa.

Bộ khung chỉ gồm vài viên gạch lặp lại: multi-head attention để trộn thông tin, positionwise FFN để biến đổi phi tuyến, residual + LayerNorm để huấn luyện sâu ổn định, và positional encoding để biết thứ tự. Encoder xếp $N$ khối để hiểu, decoder xếp $N$ khối với masked self-attention và cross-attention để sinh.

Chính tính đồng nhất và khả năng mở rộng đó đã mở đường cho [tiền huấn luyện quy mô lớn (LLM pretraining)](#/llm-pretraining): chỉ cần giữ nguyên kiến trúc, đổ vào hàng nghìn tỉ token và phóng to số tham số, ta thu được GPT, BERT, Claude. "Attention is all you need" hóa ra đúng theo nghĩa đen.

> Bài tiếp theo — **tiền huấn luyện LLM** — sẽ giữ nguyên khối Transformer ở đây nhưng thay đổi mục tiêu học: dự đoán token kế tiếp trên kho văn bản khổng lồ, để biến kiến trúc này thành một mô hình ngôn ngữ tổng quát.
