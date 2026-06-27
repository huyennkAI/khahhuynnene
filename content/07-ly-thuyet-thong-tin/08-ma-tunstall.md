# Bài 8: Mã Tunstall

> Cho đến nay, tất cả mã ta xét đều giải quyết bài toán: **từ ký hiệu nguồn → chuỗi bit độ dài biến**. Mã Tunstall lật ngược bài toán: **phân tích nguồn thành các cụm (phrase) độ dài biến → mỗi cụm được gán một codeword nhị phân độ dài cố định**. Thay vì "mỗi ký hiệu tốn bao nhiêu bit?", Tunstall hỏi "bao nhiêu ký hiệu nguồn ta có thể nén vào một khối cố định $m$ bit?".
>
> $$\text{Huffman: } \mathcal{X}^1 \to \{0,1\}^{l_i} \qquad \text{Tunstall: } \mathcal{X}^{n_j} \to \{0,1\}^m \quad (\text{cùng } m \text{ bit, } n_j \text{ thay đổi})$$

---

# 1. Bài toán đối ngẫu

## 1.1. Hai hướng mã hóa

**Hướng thứ nhất (Huffman, Shannon, Arithmetic):**
* **Đầu vào:** Nguồn $X$ phát 1 ký hiệu (hoặc $n$ ký hiệu cùng nhau).
* **Đầu ra:** Codeword nhị phân với độ dài biến $l_i$ tùy theo xác suất.
* **Mục tiêu:** Tối thiểu hóa độ dài trung bình $\mathbb{E}[l] = \sum p_i l_i$.

**Hướng thứ hai (Tunstall):**
* **Đầu ra:** Mọi codeword đều có cùng độ dài $m$ bit — tức codebook có $M = 2^m$ codeword.
* **Đầu vào:** Nguồn phát ra một **cụm (phrase)** độ dài biến $n_j$ ký hiệu.
* **Mục tiêu:** Tối đa hóa số ký hiệu nguồn trung bình mỗi codeword $\mathbb{E}[n]$.

## 1.2. Tại sao cần độ dài cố định?

Trong nhiều ứng dụng, ta cần codeword **độ dài cố định**:

* **Phần cứng:** Bộ vi xử lý xử lý từ nhớ (word) kích thước cố định.
* **Truyền dẫn:** Các gói (packet) trong mạng có kích thước cố định.
* **Xử lý song song:** Dễ dàng chia công việc đồng đều.

Với Huffman code, codeword độ dài biến gây khó khăn cho phần cứng: cần biết vị trí bit bắt đầu của codeword tiếp theo.

## 1.3. Tỉ lệ nén của Tunstall

Nếu mỗi cụm gồm trung bình $\bar{n}$ ký hiệu nguồn và được biểu diễn bằng $m$ bit, thì tỉ lệ nén là:

$$R = \frac{m}{\bar{n}} \text{ bit/ký hiệu nguồn}$$

Mã Tunstall tối ưu **tối đa hóa $\bar{n}$** (hay tương đương, tối thiểu hóa $R$) cho $m$ cố định.

---

# 2. Thuật toán Tunstall

## 2.1. Cấu trúc từ điển (dictionary)

Mã Tunstall duy trì một **tập cụm (dictionary)** $\mathcal{D}$ gồm các chuỗi hữu hạn trên $\mathcal{X}$. Hai điều kiện:

1. **Prefix-free:** Không cụm nào là prefix của cụm khác trong $\mathcal{D}$.
2. **Hoàn chỉnh (complete):** Mọi chuỗi vô hạn đều có duy nhất một cụm là prefix của nó.

Điều kiện 2 đảm bảo mọi chuỗi nguồn đều được phân tích thành chuỗi các cụm.

## 2.2. Số cụm cố định $M = D^m$

Với $m$ bit codeword và bảng chữ nhị phân ($D=2$), ta có $M = 2^m$ codeword. Vậy $|\mathcal{D}| = M$.

Mỗi cụm $s \in \mathcal{D}$ được gán một codeword nhị phân $m$-bit.

## 2.3. Thuật toán

**Khởi tạo:** $\mathcal{D} = \{x_1, x_2, \ldots, x_{|\mathcal{X}|}\}$ — bộ từ điển gồm $|\mathcal{X}|$ cụm đơn ký tự.

Điều này hoạt động khi $|\mathcal{X}| \le M$. Thường ta cần $M \ge |\mathcal{X}|$; trong trường hợp nhị phân $M = 2^m \ge 2 = |\mathcal{X}|$ với $m \ge 1$.

**Lặp lại:** Khi $|\mathcal{D}| < M$:
1. Tìm cụm $s^* \in \mathcal{D}$ có xác suất lớn nhất: $s^* = \arg\max_{s \in \mathcal{D}} p(s)$.
2. **Mở rộng:** Xóa $s^*$ khỏi $\mathcal{D}$, thêm vào $|\mathcal{X}|$ cụm con $\{s^* \cdot x : x \in \mathcal{X}\}$ (ghép $s^*$ với mỗi ký hiệu).

**Kết quả:** Khi $|\mathcal{D}| = M$, gán codeword $m$-bit cho mỗi cụm.

## 2.4. Tại sao mở rộng cụm xác suất cao nhất?

Bằng cách mở rộng cụm $s^*$, ta tạo ra $|\mathcal{X}|$ cụm mới, mỗi cụm dài hơn $s^*$ một ký hiệu. Số cụm tăng từ $|\mathcal{D}|$ lên $|\mathcal{D}| + |\mathcal{X}| - 1$.

Mỗi lần mở rộng cụm $s$, ta tăng tổng độ dài kỳ vọng thêm $p(s)$ (vì cụm $s$ nay tốn thêm 1 ký hiệu). Để tối đa hóa tổng độ dài kỳ vọng (= $\bar{n}$), ta nên mở rộng cụm có $p(s)$ lớn nhất — đúng như thuật toán làm.

---

# 3. Ví dụ chi tiết

## 3.1. Bài toán

Nguồn nhị phân: $P(0) = 0.8$, $P(1) = 0.2$. Chọn $m = 2$ bit, vậy $M = 4$ cụm.

$H(X) = H_b(0.8) \approx 0.722$ bit. Mã Tunstall tối ưu phải đạt $\bar{n} \ge m/H = 2/0.722 \approx 2.77$ ký hiệu/cụm.

## 3.2. Quá trình xây dựng

**Khởi tạo:** $\mathcal{D} = \{0, 1\}$ (2 cụm).

| Cụm | Xác suất |
|-----|---------|
| $0$ | $0.8$ |
| $1$ | $0.2$ |

**Lần 1:** $|\mathcal{D}| = 2 < 4$. Mở rộng $s^* = 0$ (xác suất cao nhất $0.8$).

Xóa $0$, thêm $00$ và $01$:

| Cụm | Xác suất |
|-----|---------|
| $1$ | $0.2$ |
| $00$ | $0.64$ |
| $01$ | $0.16$ |

**Lần 2:** $|\mathcal{D}| = 3 < 4$. Mở rộng $s^* = 00$ (xác suất cao nhất $0.64$).

Xóa $00$, thêm $000$ và $001$:

| Cụm | Xác suất |
|-----|---------|
| $1$ | $0.2$ |
| $01$ | $0.16$ |
| $000$ | $0.512$ |
| $001$ | $0.128$ |

Bây giờ $|\mathcal{D}| = 4 = M$. Dừng.

## 3.3. Gán codeword và tính $\bar{n}$

| Cụm | Xác suất | Codeword | Độ dài cụm |
|-----|---------|---------|----------|
| $000$ | $0.512$ | $00$ | $3$ |
| $1$ | $0.200$ | $01$ | $1$ |
| $01$ | $0.160$ | $10$ | $2$ |
| $001$ | $0.128$ | $11$ | $3$ |

Kiểm tra hoàn chỉnh: mọi chuỗi nhị phân đều bắt đầu bằng đúng một trong $\{000, 1, 01, 001\}$ ✓ (vì đây là prefix-free và complete).

**Độ dài kỳ vọng:**
$$\bar{n} = 0.512\cdot 3 + 0.2\cdot 1 + 0.16\cdot 2 + 0.128\cdot 3 = 1.536 + 0.2 + 0.32 + 0.384 = 2.44 \text{ ký hiệu/cụm}$$

**Tỉ lệ nén:**
$$R = \frac{m}{\bar{n}} = \frac{2}{2.44} = 0.820 \text{ bit/ký hiệu nguồn}$$

So sánh với $H(X) = 0.722$ bit. Mã Tunstall dùng $0.820$ bit/ký hiệu — kém hơn entropy $0.098$ bit. Với $m$ lớn hơn sẽ tốt hơn.

---

# 4. Phân tích khi $M \to \infty$

## 4.1. Định lý hội tụ

**Định lý:** Khi $M \to \infty$ (hay $m \to \infty$), tỉ lệ nén của Tunstall tối ưu tiến về $H(X)$:

$$\frac{m}{\bar{n}} \to H(X) \quad \text{khi } m \to \infty$$

## 4.2. Phân tích tại điểm dừng

Khi thuật toán dừng, tất cả $M$ cụm có xác suất "gần đều". Cụ thể, tại điểm dừng:

* Cụm xác suất lớn nhất: $p_{\max}$
* Cụm xác suất nhỏ nhất: $p_{\min}$

Thuật toán dừng khi ta không thể mở rộng thêm mà vẫn giữ $M$ cụm. Khi $M$ lớn, các xác suất cụm trở nên đồng đều hơn: $p(s) \approx 1/M$ với mọi $s$.

## 4.3. Kết nối với tập điển hình

Khi $M = 2^{nH+o(n)}$, các cụm trong từ điển Tunstall tương ứng gần đúng với các **chuỗi điển hình** (typical sequences) từ bài 4. Tập điển hình $A_\epsilon^{(n)}$ có $\approx 2^{nH}$ phần tử, mỗi phần tử xác suất $\approx 2^{-nH}$ — đây chính xác là cấu trúc mà Tunstall cố gắng tạo ra.

## 4.4. Tốc độ hội tụ

Cụ thể hơn, với Tunstall tối ưu:

$$\bar{n} = \frac{\log M}{H(X)} + O(1/H(X)) \quad \text{khi } M \to \infty$$

Sai số $O(1/H)$ là hằng số cộng, không phụ thuộc $M$.

---

# 5. Tính tối ưu của thuật toán Tunstall

## 5.1. Định lý tối ưu

**Định lý:** Trong tất cả các mã dùng $M$ codeword độ dài cố định, thuật toán Tunstall tối đa hóa độ dài kỳ vọng của cụm $\bar{n}$.

## 5.2. Chứng minh bằng greedy argument

Mỗi bước mở rộng tăng $\bar{n}$ thêm $p(s^*)$ (xác suất cụm được mở rộng). Vì ta muốn tối đa hóa tổng tăng trưởng $\bar{n}$, và mỗi bước được tự do chọn cụm nào để mở rộng, ta nên luôn mở rộng cụm có $p(s)$ lớn nhất — đây là quyết định greedy cục bộ và có thể chứng minh là tối ưu toàn cục.

**Chứng minh đầy đủ (exchange argument):**

Giả sử tại bước $t$, ta mở rộng cụm $s$ thay vì $s'$ với $p(s) < p(s')$. Đổi thứ tự (mở rộng $s'$ trước rồi $s$) cho $\bar{n}$ lớn hơn hoặc bằng. Chi tiết:

Sau khi mở rộng $s$: tăng $\bar{n}$ thêm $p(s)$.
Sau đó mở rộng $s'$: tăng $\bar{n}$ thêm $p(s')$ (nếu $s'$ không phải con của $s$, xác suất $p(s')$ không đổi).

Tổng tăng theo thứ tự này: $p(s) + p(s')$.

Đổi thứ tự: tổng tăng $p(s') + p(s)$ — bằng nhau! Nhưng nếu $s'$ là con của $s$ (tức $s$ đã mở rộng tạo ra $s'$), thì xác suất của con $s'$ nhỏ hơn $p(s')$ ban đầu. Vậy mở rộng $s'$ trước (khi xác suất còn cao) cho tăng lớn hơn.

## 5.3. Mức độ tối ưu

Mã Tunstall tối ưu trong lớp "prefix-free, complete, $M$ cụm". Không có cách nào dùng $M$ codeword cố định mà đạt $\bar{n}$ lớn hơn.

---

# 6. Kết nối với bài toán phân rã Lempel-Ziv

## 6.1. Lempel-Ziv (LZ78/LZW) là Tunstall thích nghi

Thuật toán Lempel-Ziv nổi tiếng có cấu trúc tương tự Tunstall:
* Duy trì một từ điển các cụm.
* Mỗi khi gặp cụm mới, thêm vào từ điển.

Khác biệt: LZ78 **không biết phân phối** trước, từ điển được xây dựng **thích nghi** khi đọc dữ liệu.

## 6.2. Tính phổ quát

LZ78 cũng tiến về $H(X)$ khi chuỗi dài, nhưng tốc độ hội tụ chậm hơn Tunstall tối ưu.

---

# 7. So sánh Tunstall với Huffman và Arithmetic Coding

## 7.1. Bảng so sánh

| Đặc điểm | Huffman | Arithmetic Coding | Tunstall |
|----------|---------|-------------------|---------|
| Kiểu mã | Variable-length | Variable-length | Fixed-length |
| Đơn vị mã | 1 symbol → $l_i$ bits | $n$ symbols → $m$ bits | Cụm $n_j$ symbols → $m$ bits |
| Tỉ lệ nén | $[H, H+1)$ bit/sym | $[H, H+1/n)$ bit/sym | $[H, H+?)$ bit/sym |
| Codebook size | $O(|\mathcal{X}|)$ | $O(|\mathcal{X}|)$ | $M = 2^m$ |
| Giải mã | Bit-by-bit | Stream | Tra bảng (M codeword) |
| Độ phức tạp giải mã | $O(1)$ (bảng) | $O(1)$ mỗi bit | $O(1)$ tra bảng |

## 7.2. Ưu điểm của Tunstall

* **Giải mã cực nhanh:** Tra bảng $M$ phần tử, không cần bit-by-bit.
* **Độ dài codeword cố định:** Tốt cho phần cứng và truyền song song.
* **Đơn giản hóa xử lý lỗi:** Biết chính xác mỗi codeword chiếm $m$ bit.

## 7.3. Nhược điểm của Tunstall

* **Codeword dài $m$ bit cần:** Từ điển kích thước $M = 2^m$ — tăng theo hàm mũ.
* **Kém hiệu quả hơn Arithmetic Coding:** Với cùng độ phức tạp, AC tốt hơn.

---

# 8. Ví dụ nâng cao: nguồn ternary

## 8.1. Bài toán

Nguồn $\mathcal{X} = \{a, b, c\}$ với $P(a) = 0.5$, $P(b) = 0.3$, $P(c) = 0.2$.

$H(X) = 0.5\log 2 + 0.3\log(10/3) + 0.2\log 5 = 0.5 + 0.521 + 0.464 = 1.485$ bit.

Chọn $m = 3$, $M = 8$ codeword. Kỳ vọng $\bar{n} \ge 3/1.485 \approx 2.02$ ký hiệu/cụm.

## 8.2. Xây dựng từ điển

**Khởi tạo:** $\mathcal{D} = \{a, b, c\}$ (3 cụm). Cần thêm $8-3=5$ cụm qua các bước mở rộng, mỗi bước thêm $3-1=2$ cụm. Cần $5/2 = 2.5$ → 3 bước? Không nguyên. Thực ra:

$|\mathcal{D}|$ ban đầu $= 3$. Mỗi bước mở rộng: $3 \to 3 + 3 - 1 = 5 \to 7 \to 9 > 8$. Vậy sau 2 bước ta có 7 cụm, cần 1 bước cuối để từ 7 → cần đúng 8. Nhưng mỗi bước thêm $|\mathcal{X}|-1=2$ cụm. Từ 3 → 5 → 7 → 9. Ta không thể đạt chính xác 8 với $|\mathcal{X}|=3$.

Tunstall thực ra hoạt động: ta dừng khi $|\mathcal{D}| \ge M$ và chấp nhận thêm cụm "trống".

Điều kiện: $(M - 1) \equiv 0 \pmod{|\mathcal{X}|-1}$, tức $(M-1) \equiv 0 \pmod{2}$, tức $M$ lẻ. $M = 8$ là chẵn nên không hoàn hảo.

**Giải pháp:** Dùng $M = 7$ ($m = \lceil\log_2 7\rceil = 3$ bit, lãng phí 1 codeword) hoặc $M = 9$ ($m = 4$ bit).

Với $M = 7$:

**Bước 0:** $\{a:0.5, b:0.3, c:0.2\}$ — 3 cụm.

**Bước 1:** Mở rộng $a$ → $\{b:0.3, c:0.2, aa:0.25, ab:0.15, ac:0.10\}$ — 5 cụm.

**Bước 2:** Mở rộng $b$ → $\{c:0.2, aa:0.25, ab:0.15, ac:0.10, ba:0.15, bb:0.09, bc:0.06\}$ — 7 cụm. Dừng.

| Cụm | Xác suất | Độ dài |
|-----|---------|------|
| $aa$ | $0.25$ | $2$ |
| $c$ | $0.20$ | $1$ |
| $ab$ | $0.15$ | $2$ |
| $ba$ | $0.15$ | $2$ |
| $ac$ | $0.10$ | $2$ |
| $bb$ | $0.09$ | $2$ |
| $bc$ | $0.06$ | $2$ |

$\bar{n} = 0.25\cdot 2 + 0.2\cdot 1 + 0.15\cdot 2 + 0.15\cdot 2 + 0.1\cdot 2 + 0.09\cdot 2 + 0.06\cdot 2$
$= 0.5 + 0.2 + 0.3 + 0.3 + 0.2 + 0.18 + 0.12 = 1.8$ ký hiệu/cụm.

Tỉ lệ nén: $\lceil\log_2 7\rceil / 1.8 = 3/1.8 = 1.667$ bit/ký hiệu nguồn.

So với $H = 1.485$ bit — còn khoảng cách do $M$ chưa đủ lớn.

---

# 9. Tổng kết toàn series

## 9.1. Nhìn lại hành trình

Chuỗi 8 bài đã xây dựng hoàn chỉnh lý thuyết mã hóa nguồn:

**Bài 1-2:** Nền tảng — Entropy $H(X)$ là lượng thông tin trung bình, $I(X;Y)$ là thông tin chung, $D_{KL} \ge 0$ dẫn ra mọi bất đẳng thức.

**Bài 3:** Phân loại mã — Prefix-free là đủ (McMillan), Kraft constraint là điều kiện cần và đủ.

**Bài 4:** AEP — Hầu hết chuỗi dài là "điển hình", chỉ cần $2^{nH}$ codeword.

**Bài 5:** Mã tối ưu — $H/\log D \le L^* < H/\log D + 1$, block coding đạt $H + 1/n$.

**Bài 6:** Mã Shannon — Xây dựng cụ thể dùng xác suất tích lũy.

**Bài 7:** Huffman (tối ưu cho variable-length) và Arithmetic Coding (tiệm cận $H$ cho bất kỳ $n$).

**Bài 8 (này):** Tunstall — Bài toán đối ngẫu: fixed-length codeword, variable-length source parsing.

## 9.2. Bảng tổng hợp tất cả phương pháp

| Phương pháp | Đầu vào | Đầu ra | Tỉ lệ nén | Ưu điểm |
|-------------|---------|---------|----------|---------|
| Shannon code | 1 symbol | Bits (variable) | $[H, H+1)$ bit/sym | Đơn giản |
| Huffman | 1 symbol | Bits (variable) | $[H, H+1)$ bit/sym | Tối ưu sym-by-sym |
| Huffman (block $n$) | $n$ symbols | Bits (variable) | $[H, H+1/n)$ | Rất tốt nhưng codebook lớn |
| Arithmetic Coding | $n$ symbols | Bits (variable) | $[H, H+1/n)$ | Không cần codebook lớn |
| Tunstall ($M$ codewords) | Cụm (variable) | $\log M$ bits (fixed) | $\approx H$ khi $M \to\infty$ | Codeword cố định |

## 9.3. Giới hạn cuối cùng

Dù dùng phương pháp nào, định lý Shannon nguồn nói rằng:

$$\boxed{R \ge H(X) \text{ bit/ký hiệu nguồn}}$$

Đây là cận tuyệt đối. Entropy $H(X)$ là "chi phí thông tin" không thể tránh khỏi của nguồn. Tất cả các mã đều chỉ có thể tiến về giới hạn này, không bao giờ phá vỡ nó.

> **Kết luận series:** Lý thuyết thông tin Shannon cho ta bức tranh hoàn chỉnh về giới hạn nén dữ liệu. Từ self-information $I(x) = -\log p(x)$ đến Arithmetic Coding đạt hiệu suất $H+1/n$, tất cả các công cụ đều quy về một câu hỏi: làm thế nào để khai thác tối đa cấu trúc xác suất của nguồn tin? Entropy là câu trả lời — nó đo chính xác bao nhiêu cấu trúc đó có thể được khai thác.
