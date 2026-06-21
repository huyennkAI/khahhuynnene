# Mô hình chuỗi & Mô hình ngôn ngữ (Sequence & Language Models)

> Phần lớn dữ liệu thú vị nhất trên đời đều có **thứ tự**: câu chữ, giá cổ phiếu, sóng âm, chuỗi DNA. Khác với một bức ảnh tĩnh, ở đây **vị trí trong chuỗi mang ý nghĩa** — đảo trật tự là đổi nội dung.
>
> $$P(x_1, x_2, \dots, x_T)$$
>
> Bài này xây nền tảng cho mọi mô hình xử lý chuỗi: cách phân rã xác suất của cả chuỗi, giả định Markov, mô hình n-gram, mô hình ngôn ngữ, và độ đo perplexity — dẫn tới nhu cầu một mô hình có **trạng thái**.

---

# 1. Dữ liệu chuỗi và phụ thuộc thời gian

Một **chuỗi (sequence)** là một dãy có thứ tự $x_1, x_2, \dots, x_T$, trong đó chỉ số $t$ thường mang nghĩa **thời gian** hoặc **vị trí**. Đặc trưng cốt lõi phân biệt dữ liệu chuỗi với dữ liệu thông thường là: các phần tử **không độc lập** với nhau.

Hãy lấy một câu tiếng Việt làm ví dụ:

> "Hôm nay trời rất ..."

Từ tiếp theo rất có thể là "đẹp", "nóng", "lạnh" — nhưng gần như không bao giờ là "bàn" hay "chạy". Bối cảnh phía trước **ràng buộc** phần tử phía sau. Đây chính là **phụ thuộc thời gian (temporal dependency)**.

Trong học máy cổ điển, ta thường giả định các mẫu là **độc lập và cùng phân phối (i.i.d.)**. Giả định này **sụp đổ** với dữ liệu chuỗi. Giá cổ phiếu hôm nay phụ thuộc hôm qua; một nốt nhạc phụ thuộc các nốt vừa vang lên. Vì thế ta cần một bộ công cụ riêng để mô hình hóa sự phụ thuộc đó.

Có hai kiểu bài toán chuỗi điển hình:

| Kiểu bài toán | Mục tiêu | Ví dụ |
| --- | --- | --- |
| Hồi quy chuỗi (sequence regression) | Dự đoán giá trị thực tiếp theo | Dự báo giá, nhiệt độ |
| Mô hình hóa xác suất (density modeling) | Ước lượng $P(x_1,\dots,x_T)$ | Mô hình ngôn ngữ |

Trọng tâm của bài là kiểu thứ hai: gán một **xác suất** cho cả chuỗi.

---

# 2. Mô hình tự hồi quy (autoregressive)

## 2.1. Phân rã xác suất theo quy tắc nhân

Câu hỏi trung tâm: làm sao gán xác suất cho một chuỗi dài tùy ý? Không gian các chuỗi là vô cùng lớn, ta không thể lập bảng tra cứu.

Mẹo nền tảng là dùng **quy tắc nhân xác suất (chain rule of probability)** để bẻ một xác suất chung phức tạp thành tích của các xác suất có điều kiện đơn giản hơn:

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1}) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

trong đó $x_{<t}$ là ký hiệu gọn cho **toàn bộ lịch sử** $x_1, \dots, x_{t-1}$ (quy ước $P(x_1 \mid x_{<1}) = P(x_1)$).

Phép phân rã này **luôn đúng**, không cần giả định gì — nó chỉ là quy tắc nhân áp dụng lặp lại. Vẻ đẹp của nó là biến bài toán "ước lượng phân phối trên cả chuỗi" thành bài toán "dự đoán phần tử kế tiếp dựa trên quá khứ", lặp lại $T$ lần.

Một mô hình học $P(x_t \mid x_{<t})$ rồi dùng chính đầu ra của nó làm đầu vào cho bước sau được gọi là **mô hình tự hồi quy (autoregressive model)** — "tự hồi quy" vì nó hồi quy biến lên chính các giá trị quá khứ của nó. Đây là khung sườn chung của GPT và mọi mô hình ngôn ngữ sinh hiện đại; xem sâu hơn ở [Mô hình tự hồi quy](#/autoregressive-models).

## 2.2. Vấn đề: lịch sử dài vô hạn

Phân rã trên tuy đúng nhưng vướng một khó khăn thực tế: số hạng $P(x_t \mid x_{<t})$ có **độ dài điều kiện tăng dần**. Khi $t$ lớn, mô hình phải nhìn lại một lịch sử rất dài, và số lượng các "lịch sử" khả dĩ bùng nổ theo cấp số nhân.

Có hai hướng giải quyết, chia đôi toàn bộ lịch sử mô hình chuỗi:

1. **Cắt bớt lịch sử** — chỉ nhìn lại một cửa sổ cố định $n-1$ phần tử. Đây là giả định Markov, dẫn tới mô hình **n-gram** (mục 3).
2. **Nén lịch sử vào một trạng thái** — duy trì một vector tóm tắt toàn bộ quá khứ. Đây là ý tưởng của **mô hình có trạng thái** (mục 8), dẫn tới [Mạng nơ-ron hồi quy](#/rnn).

---

# 3. Giả định Markov và mô hình n-gram

## 3.1. Giả định Markov

**Giả định Markov (Markov assumption)** phát biểu rằng tương lai chỉ phụ thuộc một **đoạn hữu hạn** gần nhất của quá khứ, chứ không phải toàn bộ. Với cửa sổ độ dài $n-1$:

$$P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-n+1}, \dots, x_{t-1})$$

Đây là một **xấp xỉ** (dấu $\approx$, không phải dấu $=$): ta cố tình vứt bỏ thông tin xa để đổi lấy tính khả thi. Khi giả định này đúng, chuỗi được gọi là **bậc $n-1$ Markov**.

Thay vào phân rã tự hồi quy, ta được **mô hình n-gram**:

$$P(x_1, \dots, x_T) \approx \prod_{t=1}^{T} P(x_t \mid x_{t-n+1}, \dots, x_{t-1})$$

Các trường hợp thường gặp:

| Tên | $n$ | Điều kiện | Công thức một số hạng |
| --- | --- | --- | --- |
| Unigram | 1 | Không gì cả (độc lập) | $P(x_t)$ |
| Bigram | 2 | 1 từ trước | $P(x_t \mid x_{t-1})$ |
| Trigram | 3 | 2 từ trước | $P(x_t \mid x_{t-1}, x_{t-2})$ |

## 3.2. Ước lượng bằng đếm tần suất

Điểm hấp dẫn của n-gram: ta ước lượng các xác suất chỉ bằng **đếm tần suất (frequency counting)** trên kho ngữ liệu, theo **hợp lý cực đại (maximum likelihood)**. Với bigram:

$$\hat{P}(x_t \mid x_{t-1}) = \frac{\text{count}(x_{t-1}, x_t)}{\text{count}(x_{t-1})}$$

Tức là: trong tất cả các lần từ $x_{t-1}$ xuất hiện, có bao nhiêu phần trăm lần ngay sau nó là $x_t$. Không cần gradient, không cần huấn luyện lặp — chỉ cần duyệt văn bản và đếm.

## 3.3. Hai hạn chế chí tử

Sự đơn giản của n-gram đi kèm hai vấn đề nghiêm trọng.

**Bùng nổ tổ hợp (combinatorial explosion).** Với một bộ từ vựng $|\mathcal{V}|$ từ, số n-gram khả dĩ là $|\mathcal{V}|^n$. Nếu $|\mathcal{V}| = 10^4$ và $n = 3$, ta đã có $10^{12}$ trigram cần đếm và lưu — vượt xa khả năng bộ nhớ. Tăng $n$ để bắt phụ thuộc xa hơn khiến chi phí tăng theo **cấp số nhân**.

**Tính thưa (sparsity).** Dù kho ngữ liệu lớn đến đâu, phần lớn các n-gram hợp lệ **không bao giờ xuất hiện** trong dữ liệu huấn luyện. Khi đó $\text{count}(\cdot) = 0$, kéo theo $\hat{P} = 0$, và cả tích $\prod_t P(\cdot)$ tụt về $0$ — mô hình tuyên bố một câu hoàn toàn hợp lý là "không thể xảy ra".

Người ta vá lỗi này bằng **làm trơn (smoothing)**, ví dụ Laplace cộng thêm một lượng nhỏ vào mọi đếm:

$$\hat{P}(x_t \mid x_{t-1}) = \frac{\text{count}(x_{t-1}, x_t) + \alpha}{\text{count}(x_{t-1}) + \alpha\,|\mathcal{V}|}$$

Nhưng làm trơn chỉ là miếng dán: n-gram vẫn không thể **khái quát hóa** giữa các từ có nghĩa gần nhau ("con mèo" và "con mèo con" được coi là hoàn toàn khác biệt). Đây chính là động lực để chuyển sang mô hình nơ-ron.

---

# 4. Mô hình ngôn ngữ (language model)

Một **mô hình ngôn ngữ (language model)** chính xác là một mô hình tự hồi quy áp dụng cho văn bản: nó học cách gán xác suất cho chuỗi token, hay tương đương, dự đoán **token kế tiếp**:

$$P(x_t \mid x_{<t})$$

Khi đã có $P$ này, mô hình ngôn ngữ làm được nhiều việc:

* **Sinh văn bản** — lấy mẫu lần lượt từng token từ $P(x_t \mid x_{<t})$, rồi nối token vừa sinh vào điều kiện cho bước sau (autoregressive decoding).
* **Đánh giá độ "tự nhiên"** — câu nào có $P$ cao hơn thì trôi chảy hơn, dùng cho sửa lỗi chính tả, dịch máy, nhận dạng tiếng nói.
* **Tiền huấn luyện (pretraining)** — mục tiêu dự đoán token kế tiếp là nền tảng của toàn bộ các LLM hiện đại.

Điểm tuyệt vời: dữ liệu huấn luyện là **tự giám sát (self-supervised)**. Văn bản tự nó vừa là đầu vào vừa là nhãn — token kế tiếp chính là đáp án. Không cần con người gán nhãn, nên ta có thể huấn luyện trên gần như toàn bộ văn bản của nhân loại.

---

# 5. Tokenization, vocabulary và tần suất từ

Trước khi mô hình hóa, văn bản thô phải được biến thành dãy số. Quy trình **chuyển văn bản thô (converting raw text)** gồm các bước.

## 5.1. Tách token (tokenization)

**Tách token (tokenization)** là cắt một chuỗi ký tự thành các đơn vị rời rạc gọi là **token**. Có vài mức độ hạt:

* **Mức từ (word-level)** — mỗi từ là một token; bộ từ vựng lớn, dễ gặp từ lạ (out-of-vocabulary).
* **Mức ký tự (character-level)** — mỗi ký tự là một token; từ vựng nhỏ nhưng chuỗi rất dài.
* **Mức từ con (subword, ví dụ BPE)** — dung hòa hai cực trên; là lựa chọn của hầu hết LLM hiện nay.

## 5.2. Xây từ vựng (vocabulary)

**Từ vựng (vocabulary)** là tập hợp tất cả các token duy nhất, mỗi token được gán một chỉ số nguyên. Ta thường thêm các token đặc biệt như `<unk>` (token chưa biết), `<bos>`/`<eos>` (đầu/cuối chuỗi), `<pad>` (đệm). Mô hình làm việc trên các chỉ số này, không phải trên chữ.

```python
# Đếm tần suất, dựng từ vựng từ kho ngữ liệu
from collections import Counter
tokens = " ".join(corpus).split()          # tách token mức từ
counter = Counter(tokens)                    # đếm tần suất
vocab = ["<unk>"] + [w for w, c in counter.most_common() if c >= 5]
stoi  = {w: i for i, w in enumerate(vocab)}  # token -> chỉ số
ids   = [stoi.get(w, 0) for w in tokens]     # mã hóa văn bản
```

## 5.3. Định luật Zipf

Khi xếp các từ theo tần suất giảm dần, ta quan sát một quy luật nổi tiếng: **định luật Zipf (Zipf's law)**. Tần suất $f$ của từ ở hạng $r$ giảm xấp xỉ tỉ lệ nghịch với hạng:

$$f(r) \;\propto\; \frac{1}{r^{\alpha}}, \qquad \alpha \approx 1$$

Hệ quả thực tế rất quan trọng: một số ít từ ("là", "của", "và") chiếm phần lớn số lần xuất hiện, trong khi **cái đuôi dài (long tail)** gồm vô số từ hiếm chỉ xuất hiện một vài lần. Chính cái đuôi dài này là nguyên nhân khiến n-gram bị **thưa** trầm trọng (mục 3.3): hầu hết các tổ hợp từ đều cực hiếm hoặc chưa từng thấy.

---

# 6. Chia chuỗi để huấn luyện

Một kho văn bản thường là một chuỗi token rất dài, không thể nạp cả vào mô hình. Ta phải cắt nó thành các **chuỗi con (subsequences)** độ dài cố định $n$ để tạo các minibatch.

Với mục tiêu dự đoán token kế tiếp, mỗi mẫu huấn luyện là một cặp **(đầu vào, nhãn)** lệch nhau một bước:

$$\text{input} = (x_1, \dots, x_n), \qquad \text{label} = (x_2, \dots, x_{n+1})$$

Nhãn chính là đầu vào dịch sang phải một vị trí, vì tại mỗi vị trí ta muốn dự đoán token **tiếp theo**. Có hai cách cắt:

| Chiến lược | Cách làm | Đặc điểm |
| --- | --- | --- |
| Phân vùng tuần tự | Cắt liền mạch, các batch nối tiếp nhau | Giữ ngữ cảnh dài giữa các batch |
| Lấy mẫu ngẫu nhiên | Chọn điểm bắt đầu ngẫu nhiên | Đa dạng, giảm tương quan giữa các mẫu |

Để tránh mô hình luôn nhìn cùng một cách cắt, người ta thường **dịch điểm bắt đầu ngẫu nhiên** một lượng nhỏ ở mỗi epoch trước khi phân vùng. Hàm mất mát huấn luyện là **cross-entropy** trung bình trên mọi vị trí, đo độ lệch giữa phân phối dự đoán và token thật — nối thẳng tới phần độ đo dưới đây.

---

# 7. Perplexity — đo chất lượng mô hình ngôn ngữ

## 7.1. Từ cross-entropy trung bình

Ta huấn luyện mô hình ngôn ngữ bằng cách cực tiểu hóa **cross-entropy** trung bình trên mỗi token. Với một chuỗi độ dài $T$, mất mát trung bình là:

$$\ell = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

Đại lượng này có đơn vị "nat" (nếu dùng $\log$ tự nhiên) và khó diễn giải trực giác. Ta muốn một con số dễ hình dung hơn.

## 7.2. Định nghĩa perplexity

**Perplexity (độ bối rối)** đơn giản là **lũy thừa $\exp$** của cross-entropy trung bình:

$$\text{Perplexity} = \exp(\ell) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)$$

Tương đương, đó là **trung bình nhân nghịch đảo** của xác suất gán cho từng token thật:

$$\text{Perplexity} = \left(\prod_{t=1}^{T} \frac{1}{P(x_t \mid x_{<t})}\right)^{1/T}$$

## 7.3. Diễn giải trực giác

Perplexity trả lời câu hỏi: **"trung bình mô hình đang phân vân giữa bao nhiêu lựa chọn cho token kế tiếp?"**

* **Perplexity = 1** — mô hình hoàn hảo, luôn gán xác suất $1$ cho token đúng (không hề phân vân).
* **Perplexity = $|\mathcal{V}|$** — mô hình tệ nhất, đoán mò đều nhau trên toàn bộ từ vựng (như tung xúc xắc $|\mathcal{V}|$ mặt).
* **Perplexity = $k$** — mô hình "bối rối" như thể đang chọn đều giữa $k$ khả năng.

Vì thế perplexity càng **thấp** càng tốt, và nó cho một thước đo có ý nghĩa vật lý: số nhánh phân vân hiệu dụng. Đây là độ đo chuẩn để so sánh các mô hình ngôn ngữ với nhau.

---

# 8. Hướng tới mô hình có trạng thái

Hãy nhìn lại bức tranh. Mô hình n-gram **cắt bớt lịch sử** thành một cửa sổ cố định, nên không thể bắt phụ thuộc xa, lại bùng nổ tổ hợp và thưa khi tăng $n$ (mục 3.3). Vấn đề gốc rễ: lịch sử có **độ dài thay đổi**, mà n-gram lại nhồi nó vào một cửa sổ **độ dài cố định**.

Giải pháp thanh lịch là hướng số 2 ở mục 2.2: thay vì giữ nguyên lịch sử thô, hãy **nén nó vào một vector trạng thái ẩn (hidden state)** $h_t$ tóm tắt mọi thứ đã thấy cho đến thời điểm $t$:

$$h_t = f(h_{t-1}, x_t), \qquad P(x_{t+1} \mid x_{\le t}) = g(h_t)$$

Trạng thái $h_t$ đóng vai trò "bộ nhớ": nó cập nhật theo từng token và mang theo thông tin của **toàn bộ** quá khứ, dù chuỗi dài bao nhiêu, mà số tham số vẫn cố định. Đây chính là ý tưởng của **mô hình có trạng thái (latent / state-space model)**.

Cách hiện thực hóa $f$ và $g$ bằng các phép biến đổi học được (xây trên nền [mạng nơ-ron](#/mang-no-ron)) dẫn thẳng tới **mạng nơ-ron hồi quy** — chủ đề của bài tiếp theo: [Mạng nơ-ron hồi quy](#/rnn).

---

# 9. Tổng kết

Mô hình chuỗi bắt đầu từ một phép phân rã **luôn đúng** — quy tắc nhân xác suất:

$$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

Phép phân rã này biến bài toán mô hình hóa cả chuỗi thành bài toán **dự đoán token kế tiếp** lặp lại, định nghĩa nên mô hình **tự hồi quy** và **mô hình ngôn ngữ**.

Khó khăn là lịch sử dài vô hạn. Cách cổ điển — **giả định Markov** và mô hình **n-gram** — cắt lịch sử thành cửa sổ cố định và ước lượng bằng đếm tần suất, nhưng vướng **bùng nổ tổ hợp** và **tính thưa** (mà định luật Zipf khiến trầm trọng hơn). Để dùng mô hình, ta cần **tách token**, dựng **từ vựng**, **chia chuỗi** thành các mẫu (đầu vào, nhãn) lệch một bước, và đo chất lượng bằng **perplexity** — chính là $\exp$ của cross-entropy trung bình, cho biết số nhánh phân vân hiệu dụng.

Hạn chế của cửa sổ cố định dẫn ta tới một ý tưởng mạnh hơn: nén toàn bộ lịch sử vào một **trạng thái ẩn** cập nhật theo thời gian.

> Bài tiếp theo — [Mạng nơ-ron hồi quy](#/rnn) — hiện thực hóa chính ý tưởng trạng thái ẩn $h_t = f(h_{t-1}, x_t)$ bằng mạng nơ-ron, cho phép mô hình mang theo bộ nhớ qua chuỗi dài tùy ý mà không bùng nổ tham số.
