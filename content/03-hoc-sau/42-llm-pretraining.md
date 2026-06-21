# Tiền huấn luyện quy mô lớn & Mô hình ngôn ngữ lớn (LLM)

> Ý tưởng trung tâm của các mô hình ngôn ngữ lớn (large language model, LLM) gọn tới mức ngạc nhiên: **dạy một [Transformer](#/transformer) đoán văn bản trên một kho ngữ liệu khổng lồ, không cần nhãn**, rồi tái sử dụng nó cho mọi tác vụ về sau.
>
> $$\underbrace{\text{văn bản thô khổng lồ}}_{\text{không nhãn}} \;\xrightarrow{\text{tiền huấn luyện}}\; \text{mô hình nền (foundation model)} \;\xrightarrow{\text{fine-tune / prompting}}\; \text{tác vụ cụ thể}$$
>
> Đây là nền tảng của GPT, BERT, T5 và gần như toàn bộ NLP hiện đại.

---

# 1. Vì sao lại là tự giám sát?

Học có giám sát (supervised learning) cổ điển cần dữ liệu gán nhãn thủ công — đắt đỏ và khan hiếm. Mỗi tác vụ mới (phân loại cảm xúc, trả lời câu hỏi, dịch máy) lại đòi một bộ nhãn riêng.

Nhưng văn bản thô thì **gần như vô hạn và miễn phí**: web, sách, mã nguồn, bách khoa toàn thư. Câu hỏi đặt ra:

> Liệu có thể tạo ra "bài toán có nhãn" ngay từ chính văn bản không nhãn không?

Câu trả lời là **học tự giám sát (self-supervised learning)**: ta che hoặc giấu một phần văn bản, rồi bắt mô hình dự đoán phần bị giấu từ phần còn lại. Nhãn chính là token bị giấu — sinh ra tự động, không cần con người.

Tín hiệu giám sát này tuy "rẻ" nhưng cực kỳ giàu: để đoán đúng từ tiếp theo trong "Thủ đô của Việt Nam là ___", mô hình buộc phải học cú pháp, ngữ nghĩa, sự kiện thực tế và cả suy luận. Dự đoán văn bản chính xác **đòi hiểu văn bản**.

---

# 2. Tiền thân: embedding từ (word2vec)

Trước kỷ nguyên Transformer, bước đột phá đầu tiên của học tự giám sát trên văn bản là **word2vec** (Mikolov, 2013). Ý tưởng: biểu diễn mỗi từ bằng một vector dày đặc (dense embedding) sao cho các từ xuất hiện trong ngữ cảnh giống nhau có vector gần nhau.

Biến thể **skip-gram** huấn luyện bằng cách cho từ trung tâm $w_c$ dự đoán các từ ngữ cảnh $w_o$ xung quanh trong một cửa sổ:

$$\max_{\theta} \; \prod_{c} \prod_{-m \le j \le m,\, j \ne 0} P(w_{c+j} \mid w_c), \qquad P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{k} \exp(\mathbf{u}_k^\top \mathbf{v}_c)}$$

Đây đã là tinh thần "che rồi đoán": ngữ cảnh đóng vai nhãn tự sinh. Nhưng word2vec gán cho mỗi từ **một vector tĩnh duy nhất** — không phân biệt nghĩa của "ngân hàng" trong "ngân hàng sông" và "ngân hàng Nhà nước".

Bước nhảy của LLM là thay embedding tĩnh bằng **biểu diễn theo ngữ cảnh (contextual representation)**: vector của một token phụ thuộc vào cả câu, nhờ cơ chế [chú ý đa đầu (multi-head attention)](#/multi-head-attention) trong Transformer.

---

# 3. Ba họ kiến trúc Transformer

Cùng dựa trên khối Transformer, nhưng tùy theo **cách dòng chú ý được phép chảy** và **mục tiêu tiền huấn luyện**, người ta chia thành ba họ. Đây là khung phân loại quan trọng nhất của bài.

| Tiêu chí | Encoder-only (BERT) | Decoder-only (GPT) | Encoder–Decoder (T5/BART) |
| --- | --- | --- | --- |
| Hướng chú ý | hai chiều (bidirectional) | một chiều (causal, trái→phải) | encoder hai chiều + decoder một chiều |
| Mục tiêu tiền huấn luyện | masked LM (che token rồi đoán) | autoregressive LM ($\prod_t P(x_t\mid x_{<t})$) | khử nhiễu seq2seq (span corruption) |
| Việc hợp | hiểu / biểu diễn / phân loại | sinh văn bản / hội thoại | biến đổi chuỗi → chuỗi (dịch, tóm tắt) |
| Token thấy được | toàn câu | chỉ token bên trái | nguồn toàn bộ, đích bên trái |
| Đại diện | BERT, RoBERTa | GPT, LLaMA, PaLM | T5, BART |

Ba mục dưới đây đi sâu vào từng họ.

---

## 3.1. Encoder-only — BERT và masked language modeling

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin 2018) chỉ dùng phần encoder. Vì encoder dùng chú ý **hai chiều**, mỗi token được nhìn thấy cả ngữ cảnh bên trái lẫn bên phải cùng lúc — lý tưởng cho việc *hiểu* hơn là *sinh*.

Nhưng nếu cho phép nhìn cả hai phía thì không thể dùng mục tiêu "đoán từ tiếp theo" được (mô hình sẽ thấy luôn đáp án). Giải pháp là **masked language modeling (MLM)**: che ngẫu nhiên khoảng 15% số token bằng ký hiệu đặc biệt `[MASK]`, rồi bắt mô hình khôi phục chúng từ ngữ cảnh hai phía.

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\setminus \mathcal{M}})$$

trong đó $\mathcal{M}$ là tập vị trí bị che, còn $x_{\setminus \mathcal{M}}$ là phần còn lại của câu. Ví dụ:

```text
Đầu vào:  Thủ đô của [MASK] là Hà Nội .
Mục tiêu: [MASK] = "Việt Nam"
```

BERT còn học thêm tác vụ phụ **dự đoán câu kế tiếp (next sentence prediction)** để nắm quan hệ giữa hai câu. Sau tiền huấn luyện, ta gắn thêm một lớp đầu nhỏ và **fine-tune** cho phân loại, gán nhãn token, hay trích xuất câu trả lời.

> Điểm mạnh: biểu diễn theo ngữ cảnh hai chiều, rất hợp cho hiểu và biểu diễn. Điểm yếu: không sinh văn bản tự nhiên được, vì `[MASK]` chỉ xuất hiện khi huấn luyện chứ không có khi suy luận thật.

---

## 3.2. Decoder-only — GPT và mô hình ngôn ngữ tự hồi quy

**GPT** (Generative Pre-trained Transformer) chỉ dùng phần decoder với **chú ý nhân quả (causal/masked self-attention)**: token tại vị trí $t$ chỉ được nhìn các token đứng *trước* nó. Đây chính là một [mô hình tự hồi quy (autoregressive model)](#/autoregressive-models): phân tích xác suất của cả câu bằng quy tắc chuỗi.

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

Mục tiêu tiền huấn luyện đơn giản là cực đại hóa log-likelihood — tức đoán **token kế tiếp**:

$$\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

```text
Đầu vào:  Thủ đô của Việt Nam là
Dự đoán:  Hà   (rồi   Nội ,   .   ...)
```

Khác với MLM, mục tiêu này tự nhiên cho phép **sinh văn bản**: sau khi học xong, ta lấy mẫu từng token một, mỗi token sinh ra lại nối vào đầu vào để sinh token kế. Vì mục tiêu huấn luyện (đoán token kế) trùng khít với cách dùng (sinh token kế), họ decoder-only mở rộng cực tốt và là nền tảng của hầu hết LLM hội thoại hiện nay (GPT-4, LLaMA, Claude).

> Điểm mạnh: sinh trôi chảy, một mục tiêu duy nhất dùng được cho mọi tác vụ qua prompting. Điểm yếu: biểu diễn chỉ một chiều, về lý thuyết kém BERT ở các tác vụ hiểu thuần túy (dù khoảng cách này thu hẹp nhanh khi quy mô tăng).

---

## 3.3. Encoder–Decoder — T5 và BART

Họ thứ ba giữ nguyên kiến trúc seq2seq đầy đủ của Transformer gốc: một encoder hai chiều đọc chuỗi nguồn, một decoder tự hồi quy sinh chuỗi đích, nối với nhau bằng **chú ý chéo (cross-attention)**. Đây là lựa chọn tự nhiên cho các tác vụ **biến đổi chuỗi → chuỗi**: dịch máy, tóm tắt, viết lại.

**T5** (Text-to-Text Transfer Transformer) đẩy ý tưởng tới cực hạn: gói *mọi* tác vụ NLP vào cùng một khuôn "văn bản vào → văn bản ra". Phân loại cảm xúc trở thành sinh ra chữ "positive"; dịch trở thành sinh ra câu đích. Mục tiêu tiền huấn luyện là **khử nhiễu kiểu span**: che cả một đoạn token liên tiếp rồi bắt decoder sinh lại đoạn đó.

```text
Đầu vào (đã che):  Thủ đô của <X> là Hà Nội .
Mục tiêu:          <X> Việt Nam
```

**BART** dùng tinh thần tương tự: làm hỏng văn bản (xáo trộn, xóa, che span) rồi học khôi phục bản gốc — kết hợp được ưu thế hiểu hai chiều của encoder với khả năng sinh của decoder.

> Điểm mạnh: linh hoạt cho tác vụ có cả đầu vào và đầu ra dạng chuỗi. Điểm yếu: nặng hơn (hai ngăn xếp), và với tác vụ thuần sinh thì decoder-only thường gọn và hiệu quả hơn.

---

# 4. Hai cách dùng mô hình đã tiền huấn luyện

Có mô hình nền rồi, làm sao đưa nó về tác vụ đích? Có hai con đường.

## 4.1. Fine-tuning

**Fine-tuning** tiếp tục huấn luyện mô hình trên dữ liệu có nhãn của tác vụ đích, cập nhật (toàn bộ hoặc một phần) trọng số. Vì khởi đầu đã từ một mô hình "biết ngôn ngữ", ta chỉ cần ít dữ liệu và ít bước là đạt chất lượng cao. Đây là cách dùng chuẩn của BERT.

Để tiết kiệm, các kỹ thuật **fine-tuning hiệu quả tham số (parameter-efficient)** như LoRA chỉ huấn luyện một lượng nhỏ trọng số bổ sung, đóng băng phần còn lại — giảm mạnh chi phí khi mô hình có hàng tỉ tham số.

## 4.2. In-context learning (prompting)

Khi mô hình đủ lớn, xuất hiện một khả năng đáng kinh ngạc: **học trong ngữ cảnh (in-context learning)**. Ta **không** đụng tới trọng số; thay vào đó mô tả tác vụ và đưa vài ví dụ ngay trong prompt, rồi để mô hình suy ra câu trả lời:

```text
Dịch sang tiếng Anh:
"con mèo" => "the cat"
"con chó" => "the dog"
"con gà" =>            ← mô hình tự sinh: "the chicken"
```

- **Zero-shot**: chỉ mô tả tác vụ, không ví dụ.
- **Few-shot**: kèm vài ví dụ mẫu.

In-context learning biến một mô hình duy nhất thành công cụ đa năng: cùng bộ trọng số giải vô số tác vụ chỉ bằng cách đổi prompt — không cần huấn luyện lại lần nào.

---

# 5. Luật tỉ lệ và năng lực nổi

Vì sao đổ tài nguyên khổng lồ vào tiền huấn luyện lại đáng? Câu trả lời nằm ở **luật tỉ lệ (scaling laws)** (Kaplan 2020): chất lượng mô hình (đo bằng loss kiểm thử) cải thiện **trơn tru và dự đoán được** theo lũy thừa của ba yếu tố — số tham số $N$, lượng dữ liệu $D$, và lượng tính toán $C$:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}$$

Loss giảm theo hàm lũy thừa nghĩa là: cứ tăng quy mô đủ lớn, mô hình gần như **chắc chắn** tốt lên. Chính sự dự đoán được này biện minh cho việc đầu tư vào các mô hình ngày càng lớn.

Thú vị hơn, một số khả năng **không** xuất hiện dần dần mà **bật lên đột ngột** khi vượt một ngưỡng quy mô — gọi là **năng lực nổi (emergent abilities)**: làm toán nhiều bước, suy luận chuỗi (chain-of-thought), làm theo chỉ dẫn. Ở mô hình nhỏ chúng gần như bằng 0, rồi xuất hiện bất ngờ ở mô hình đủ lớn. Đây là một trong những lý do "lớn hơn" lại tạo ra khác biệt *về chất*, không chỉ về lượng.

---

# 6. Tổng kết

Mọi LLM hiện đại đều quy về một công thức chung:

$$\boxed{\;\text{[Transformer](\#/transformer)} \;+\; \text{dữ liệu khổng lồ} \;+\; \text{mục tiêu tự giám sát} \;=\; \text{mô hình nền}\;}$$

Điểm cốt lõi không phải là một thuật toán mới lạ, mà là **quy mô**: một mục tiêu dự đoán đơn giản (che rồi đoán, hoặc đoán token kế) khi áp lên đủ dữ liệu và đủ tham số sẽ sinh ra năng lực ngôn ngữ tổng quát.

Ba họ kiến trúc phân vai rõ ràng theo hướng chú ý và mục tiêu: **encoder-only (BERT)** dùng chú ý hai chiều với masked LM, mạnh ở *hiểu*; **decoder-only (GPT)** dùng chú ý nhân quả với mục tiêu tự hồi quy $\prod_t P(x_t\mid x_{<t})$, mạnh ở *sinh*; **encoder–decoder (T5/BART)** giữ kiến trúc seq2seq, mạnh ở *biến đổi chuỗi*.

Từ embedding tĩnh của word2vec, qua biểu diễn theo ngữ cảnh của Transformer, đến những mô hình hàng trăm tỉ tham số biết học ngay trong prompt — đây là con đường đã đưa NLP từ "nhiều mô hình chuyên biệt" tới "một mô hình nền cho tất cả".

> Transformer cộng dữ liệu lớn chính là nền của LLM hiện đại. Bài tiếp theo sẽ đi sâu vào cách **sinh và lấy mẫu văn bản** từ một mô hình tự hồi quy đã huấn luyện, cùng các chiến lược giải mã (greedy, beam search, top-$k$, nucleus sampling).
