# Bài 7: Mã Huffman và mã số học (Arithmetic Coding)

> Bài này giải quyết hai câu hỏi thực tiễn. **Thứ nhất:** mã prefix-free tối ưu nhìn như thế nào và làm thế nào để xây dựng nó? Đó là **mã Huffman** — xây cây từ dưới lên, gộp hai nút xác suất thấp nhất mỗi bước. **Thứ hai:** làm thế nào đạt hiệu suất của block coding mà không cần codebook khổng lồ? Đó là **Arithmetic Coding** — biểu diễn toàn bộ chuỗi bằng một số trong $[0,1)$.
>
> $$L_{\text{Huffman}} = L^* \le L_{\text{Shannon}} < \frac{H(X)}{\log D} + 1 \qquad L_{\text{Arithmetic}} \approx \frac{H(X^n)}{n} + \frac{2}{n}$$

---

# 1. Mã Huffman: thuật toán

## 1.1. Ý tưởng cốt lõi

Bài trước chỉ ra rằng trong mã tối ưu: hai ký hiệu xác suất thấp nhất luôn có cùng độ dài lớn nhất và là anh em trong cây. Huffman code khai thác tính chất này bằng cách xây cây **từ dưới lên**: bắt đầu từ các lá (ký hiệu nguồn), gộp dần hai nút xác suất thấp nhất, cho tới khi còn một nút gốc.

## 1.2. Thuật toán Huffman (nhị phân)

**Đầu vào:** Phân phối $(p_1, \ldots, p_m)$.

**Thuật toán:**
1. Tạo $m$ nút lá, mỗi nút có trọng số bằng xác suất.
2. Lặp lại cho đến khi còn 1 nút:
   a. Tìm hai nút có trọng số nhỏ nhất, gọi là $A$ và $B$.
   b. Tạo nút cha $C$ với trọng số $w_C = w_A + w_B$.
   c. Gán nhánh $0$ cho $A$, nhánh $1$ cho $B$ (hoặc ngược lại).
   d. Thay $A, B$ bằng $C$ trong danh sách.
3. Nút còn lại là gốc cây.

**Đầu ra:** Cây mã. Codeword của mỗi ký hiệu là chuỗi bit trên đường từ gốc đến lá.

## 1.3. Ví dụ đơn giản

Nguồn $\mathcal{X} = \{a, b, c, d, e\}$ với $p = (0.4, 0.2, 0.2, 0.1, 0.1)$.

**Bước 1:** Danh sách ban đầu: $\{a:0.4, b:0.2, c:0.2, d:0.1, e:0.1\}$

**Bước 2:** Gộp $d(0.1)$ và $e(0.1)$ thành $de(0.2)$:
$$\{a:0.4, b:0.2, c:0.2, de:0.2\}$$

**Bước 3:** Gộp $b(0.2)$ và $c(0.2)$ thành $bc(0.4)$:
$$\{a:0.4, bc:0.4, de:0.2\}$$

**Bước 4:** Gộp $de(0.2)$ và $a(0.4)$ → thực ra ta gộp hai nhỏ nhất là $de(0.2)$ và ... chỉ còn ba nút với giá trị $0.4, 0.4, 0.2$. Gộp $0.4+0.2 = 0.6$ hoặc nhập nhằng. Thường ta sắp xếp:

Gộp $de(0.2)$ và chọn theo quy tắc (ưu tiên nút lớn hơn làm nhánh $0$):

Chọn gộp hai nhỏ nhất: $de(0.2)$ và một trong $a$ hoặc $bc$ (đều $0.4$). Ta gộp $de(0.2)$ với $a(0.4)$ thành $ade(0.6)$:
$$\{bc:0.4, ade:0.6\}$$

**Bước 5:** Gộp $bc(0.4)$ và $ade(0.6)$ thành gốc $(1.0)$:

Cây hoàn chỉnh và codeword:

```
           root(1.0)
          /         \
       bc(0.4)     ade(0.6)
       /    \      /       \
     b(0.2) c(0.2) a(0.4)  de(0.2)
                           /     \
                         d(0.1) e(0.1)
```

| Ký hiệu | Codeword | Độ dài |
|---------|---------|-------|
| $b$ | $00$ | $2$ |
| $c$ | $01$ | $2$ |
| $a$ | $10$ | $2$ |
| $d$ | $110$ | $3$ |
| $e$ | $111$ | $3$ |

$L = 0.4\cdot 2 + 0.2\cdot 2 + 0.2\cdot 2 + 0.1\cdot 3 + 0.1\cdot 3 = 0.8+0.4+0.4+0.3+0.3 = 2.2$ bit.

$H(X) = 0.4\log(1/0.4) + 0.2\log(1/0.2)\cdot 2 + 0.1\log(1/0.1)\cdot 2 = 0.529+0.929+0.664 = 2.122$ bit.

$L = 2.2 > H = 2.122$, sai số $0.078$ bit — rất nhỏ.

---

# 2. Chứng minh tính tối ưu của Huffman

## 2.1. Exchange argument

**Định lý:** Huffman code là mã prefix-free tối ưu (độ dài trung bình nhỏ nhất).

**Chứng minh bằng quy nạp trên $m$ (số ký hiệu):**

**Cơ sở ($m=2$):** Chỉ có một mã prefix-free: codeword độ dài 1. Huffman cho đúng mã này.

**Bước quy nạp:** Giả sử Huffman tối ưu cho $m-1$ ký hiệu. Ta chứng minh tối ưu cho $m$ ký hiệu.

**Bổ đề 1:** Trong mã tối ưu $C^*$, hai ký hiệu xác suất nhỏ nhất $p_m \le p_{m-1}$ có codeword dài nhất và là anh em.

**Bổ đề 2:** Nếu ta gộp hai ký hiệu anh em $x_{m-1}, x_m$ thành ký hiệu "siêu" $z$ với $p_z = p_{m-1} + p_m$, thì mã tối ưu cho bài toán $m-1$ ký hiệu mới là mã thu được từ $C^*$ bằng cách rút ngắn codeword của $x_{m-1}, x_m$ đi 1 bit (tức là dùng codeword chung của cha chúng).

Khi đó:
$$L(C^*) = L(C^*_{m-1}) + p_{m-1} + p_m = L(C^*_{m-1}) + p_z$$

Trong đó $C^*_{m-1}$ là mã tối ưu cho bài toán $m-1$ ký hiệu.

**Theo giả thuyết quy nạp:** Huffman tối ưu cho $m-1$ ký hiệu, nên $L(\text{Huffman}_{m-1}) = L(C^*_{m-1})$. Suy ra:
$$L(\text{Huffman}_m) = L(\text{Huffman}_{m-1}) + p_z = L(C^*_{m-1}) + p_z = L(C^*_m)$$

Vậy Huffman tối ưu cho $m$ ký hiệu.

## 2.2. Tính không duy nhất

Huffman code không duy nhất: có thể có nhiều mã tối ưu với cùng độ dài trung bình. Nguyên nhân:
* Có thể gán $0$ hay $1$ cho nhánh trái/phải tùy ý.
* Khi có nhiều nút cùng trọng số, thứ tự gộp có thể thay đổi.

Tất cả đều có cùng $L^*$, nhưng phân phối độ dài codeword có thể khác.

---

# 3. Huffman cho bảng chữ D-ary

## 3.1. Mở rộng

Với bảng chữ mã $D$-ary, mỗi bước gộp **$D$ nút** (thay vì 2) thành một nút cha.

**Vấn đề:** Số ký hiệu phải thỏa $(m - 1) \equiv 0 \pmod{D-1}$ để cây $D$-ary hoàn chỉnh. Nếu không, thêm ký hiệu giả với xác suất $0$.

**Điều kiện thêm ký hiệu giả:** Cần thêm $k$ ký hiệu giả sao cho $m + k \equiv 1 \pmod{D-1}$, tức $k = (D-1) - ((m-1) \mod (D-1)) \mod (D-1)$.

## 3.2. Ví dụ D=3 (ternary Huffman)

Nguồn $\mathcal{X} = \{a, b, c, d, e\}$ với $p = (0.4, 0.2, 0.2, 0.1, 0.1)$. $m=5$, $D=3$.

Điều kiện: $(5-1) \mod (3-1) = 4 \mod 2 = 0$ — không cần thêm ký hiệu giả.

**Bước 1:** Gộp 3 nút nhỏ nhất: $d(0.1), e(0.1), c(0.2)$ → $dec(0.4)$.
$$\{a:0.4, b:0.2, dec:0.4\}$$

**Bước 2:** Gộp 3 nút còn lại: $b(0.2), a(0.4), dec(0.4)$ → gốc $(1.0)$.

| Ký hiệu | Codeword |
|---------|---------|
| $b$ | $0$ |
| $a$ | $1$ |
| $d$ | $20$ |
| $e$ | $21$ |
| $c$ | $22$ |

$L = 0.4\cdot 1 + 0.2\cdot 1 + 0.1\cdot 2 + 0.1\cdot 2 + 0.2\cdot 2 = 0.4+0.2+0.2+0.2+0.4 = 1.4$ trit.

---

# 4. Arithmetic Coding: ý tưởng

## 4.1. Vấn đề của Huffman

Huffman code phải gán độ dài nguyên cho mỗi ký hiệu. Nếu $p_i = 0.9$, ký hiệu "xứng đáng" $0.152$ bit nhưng ta phải dùng tối thiểu 1 bit. Sai số $0.848$ bit rất lớn.

Arithmetic Coding giải quyết bằng cách mã hóa **chuỗi dài** mà không cần block của $n$ ký hiệu hoặc codebook khổng lồ.

## 4.2. Ý tưởng cơ bản

Thay vì gán một codeword cho mỗi ký hiệu, Arithmetic Coding mã hóa **toàn bộ chuỗi** $x^n$ thành một số thực trong khoảng $[0,1)$.

**Nguyên tắc:** Mỗi chuỗi $x^n$ tương ứng với một khoảng con $[l, l+p(x^n))$ trong $[0,1)$. Codeword là biểu diễn nhị phân ngắn nhất của một điểm trong khoảng đó.

---

# 5. Arithmetic Coding: xây dựng

## 5.1. Xác suất tích lũy

Sắp xếp $\mathcal{X} = \{x_1, \ldots, x_m\}$ và tính:
$$F(x_k) = \sum_{j < k} p(x_j) = P(X < x_k)$$

## 5.2. Khoảng cho một ký hiệu

Ký hiệu $x_k$ ứng với khoảng $[F(x_k), F(x_k) + p(x_k))$.

## 5.3. Khoảng cho chuỗi

Với chuỗi $x^n = (x_1^{(1)}, x_2^{(2)}, \ldots)$ (chú ý ký hiệu index trên là thứ tự trong chuỗi, index dưới là nhãn ký hiệu), khoảng thu hẹp dần:

**Khởi tạo:** $[l_0, u_0) = [0, 1)$.

**Bước $i$:** Khi nhận ký hiệu $x^{(i)}$:
$$l_i = l_{i-1} + (u_{i-1} - l_{i-1}) \cdot F(x^{(i)})$$
$$u_i = l_{i-1} + (u_{i-1} - l_{i-1}) \cdot (F(x^{(i)}) + p(x^{(i)}))$$

Tức là khoảng hiện tại $[l_{i-1}, u_{i-1})$ bị chia theo xác suất và ta chỉ giữ phần ứng với $x^{(i)}$.

## 5.4. Ví dụ cụ thể

Nguồn $\mathcal{X} = \{a, b, c\}$ với $p(a) = 0.5$, $p(b) = 0.3$, $p(c) = 0.2$.

Xác suất tích lũy: $F(a) = 0$, $F(b) = 0.5$, $F(c) = 0.8$.

**Mã hóa chuỗi $abc$:**

**Bước 1 — ký hiệu $a$:**
$$l_1 = 0 + 1 \cdot 0 = 0, \quad u_1 = 0 + 1 \cdot 0.5 = 0.5$$
Khoảng: $[0, 0.5)$

**Bước 2 — ký hiệu $b$:**
$$l_2 = 0 + 0.5 \cdot 0.5 = 0.25, \quad u_2 = 0 + 0.5 \cdot 0.8 = 0.4$$
Khoảng: $[0.25, 0.4)$

**Bước 3 — ký hiệu $c$:**
$$l_3 = 0.25 + (0.4-0.25)\cdot 0.8 = 0.25 + 0.12 = 0.37$$
$$u_3 = 0.25 + (0.4-0.25)\cdot 1.0 = 0.25 + 0.15 = 0.40$$
Khoảng: $[0.37, 0.40)$

**Codeword:** Tìm biểu diễn nhị phân ngắn nhất của một số trong $[0.37, 0.40)$.

$0.375 = 0.011_2$ nằm trong $[0.37, 0.40)$. Codeword: $011$ (3 bit).

**Kiểm tra:** $p(abc) = 0.5 \cdot 0.3 \cdot 0.2 = 0.03$. Độ dài codeword lý thuyết: $\log_2(1/0.03) \approx 5.06$ bit. Mã $011$ chỉ 3 bit — nhưng đây là vì ta may mắn chọn được số nhỏ. Thực ra cần $\lceil\log_2(1/p)\rceil + 1$ bit để đảm bảo.

---

# 6. Phân tích độ dài của Arithmetic Coding

## 6.1. Codeword của Arithmetic Coding

Để chứa một điểm trong khoảng $[l, l+p)$ của chuỗi $x^n$, ta cần codeword độ dài:
$$L_{AC}(x^n) = \left\lceil \log_2 \frac{1}{p(x^n)} \right\rceil + 1$$

**Chứng minh:** Khoảng rộng $p(x^n) = 2^{-\lceil\log_2(1/p)\rceil} \cdot 2^{\lceil\log_2(1/p)\rceil-\log_2(1/p)} \ge 2^{-\lceil\log_2(1/p)\rceil}$. Ta cần bit thêm (+ 1) để đảm bảo điểm ta chọn rơi trong khoảng đúng, không bị nhầm sang khoảng bên cạnh.

## 6.2. Độ dài trung bình

$$\mathbb{E}[L_{AC}(X^n)] \le H(X^n) + 1 = nH(X) + 1$$

Tốc độ bit/ký hiệu:
$$\frac{\mathbb{E}[L_{AC}(X^n)]}{n} \le H(X) + \frac{1}{n}$$

Cận dưới (từ bài trước): $\ge H(X)$.

## 6.3. So sánh với Shannon code và Huffman

| Phương pháp | Độ dài bit/ký hiệu | Nhận xét |
|-------------|-------------------|---------|
| Huffman ($n=1$) | $[H, H+1)$ | Tối ưu cho symbol-by-symbol |
| Shannon code ($n=1$) | $[H, H+1)$ | Không tối ưu |
| Huffman ($n$ block) | $[H, H+1/n)$ | Tốt nhưng codebook $|\mathcal{X}|^n$ |
| Arithmetic Coding | $[H, H+1/n)$ | Không cần codebook lớn! |

Arithmetic Coding đạt hiệu suất của Huffman block coding mà **không cần codebook kích thước $|\mathcal{X}|^n$**.

---

# 7. Giải mã Arithmetic Coding

## 7.1. Thuật toán giải mã

**Đầu vào:** Codeword nhị phân $c$ và độ dài chuỗi $n$.

1. Chuyển $c$ thành số thực $v = 0.c_1 c_2 \ldots$
2. Lặp $n$ lần:
   a. Tìm ký hiệu $x_k$ sao cho $F(x_k) \le v < F(x_k) + p(x_k)$ (tìm khoảng chứa $v$).
   b. Output ký hiệu $x_k$.
   c. Cập nhật $v = (v - F(x_k)) / p(x_k)$.

## 7.2. Ví dụ giải mã

Codeword $011$ = $0.375$ trong $[0,1)$. Nguồn như trên với $p(a)=0.5, p(b)=0.3, p(c)=0.2$.

**Bước 1:** $v = 0.375$. $F(a)=0 \le 0.375 < 0.5 = F(a)+p(a)$. Ký hiệu: $a$.
$v = (0.375 - 0)/0.5 = 0.75$.

**Bước 2:** $v = 0.75$. $F(b)=0.5 \le 0.75 < 0.8 = F(b)+p(b)$. Ký hiệu: $b$.
$v = (0.75 - 0.5)/0.3 = 0.833$.

**Bước 3:** $v = 0.833$. $F(c)=0.8 \le 0.833 < 1.0 = F(c)+p(c)$. Ký hiệu: $c$.

Kết quả: $abc$ ✓

---

# 8. Arithmetic Coding trong thực tế

## 8.1. Vấn đề số học dấu phẩy động

Với chuỗi dài, khoảng $[l, u)$ có thể cực kỳ nhỏ (nhỏ hơn $2^{-n}$), vượt quá độ chính xác của số học dấu phẩy động.

**Giải pháp:** "Rescaling" (tỉ lệ lại) khi khoảng thu hẹp về một nửa khoảng nhất định:
* Nếu $[l, u) \subset [0, 0.5)$: output bit $0$ và tỉ lệ gấp đôi.
* Nếu $[l, u) \subset [0.5, 1)$: output bit $1$ và tỉ lệ gấp đôi trừ 1.
* Nếu $[l, u) \subset [0.25, 0.75)$: xử lý "convergence" — bit tiếp theo có thể là $0$ hoặc $1$.

## 8.2. Arithmetic Coding với phân phối thích nghi

Arithmetic Coding có thể hoạt động với phân phối **không biết trước** bằng cách cập nhật xác suất theo tần suất thực nghiệm trong quá trình mã hóa. Gọi là **adaptive arithmetic coding** — là nền tảng của nhiều bộ nén hiện đại (bzip2, 7-zip).

## 8.3. Huffman thích nghi

Tương tự, có Huffman thích nghi (adaptive Huffman): cây được cập nhật sau mỗi ký hiệu. Nhưng overhead cao hơn và không đạt hiệu suất của Arithmetic Coding.

---

# 9. Ứng dụng: DEFLATE (zip, gzip)

## 9.1. DEFLATE kết hợp LZ77 và Huffman

Thuật toán nén DEFLATE (dùng trong zip, gzip, PNG) gồm hai bước:
1. **LZ77:** Tìm và thay thế chuỗi lặp bằng tham chiếu.
2. **Huffman:** Nén output của LZ77 bằng static Huffman code.

## 9.2. Vì sao không dùng Arithmetic Coding trong DEFLATE?

Lý do lịch sử và patent: Arithmetic Coding có patent trong những năm 1980-90. DEFLATE dùng Huffman để tránh vấn đề pháp lý. Hiện tại, các thuật toán mới hơn (Zstandard, Brotli) dùng ANS (Asymmetric Numeral Systems) — biến thể của Arithmetic Coding.

---

# 10. So sánh tổng thể

## 10.1. Bảng so sánh toàn diện

| Đặc điểm | Huffman | Arithmetic Coding |
|----------|---------|-------------------|
| Độ phức tạp xây dựng | $O(m \log m)$ | $O(n)$ |
| Độ phức tạp mã hóa | $O(n)$ | $O(n)$ |
| Codebook size | $O(m)$ | $O(m)$ (chỉ cần phân phối) |
| Độ dài bit/symbol | $[H, H+1)$ | $[H, H+1/n)$ |
| Thích nghi (adaptive) | Phức tạp | Tự nhiên |
| Chia theo symbol | Có | Không (mã hóa toàn chuỗi) |
| Patent issue | Không | Có (trước 2002) |
| Sử dụng phổ biến | DEFLATE, JPEG | bzip2, LZMA, ANS |

## 10.2. Khi nào dùng cái nào?

**Dùng Huffman khi:**
* Phân phối đã biết và cố định (static Huffman)
* Cần tốc độ giải mã cực nhanh (tra bảng)
* Triển khai đơn giản

**Dùng Arithmetic Coding khi:**
* Cần hiệu suất tiệm cận $H(X)$
* Phân phối thay đổi (adaptive)
* Kết hợp mô hình ngữ cảnh phức tạp

---

# 11. Tổng kết

Hai phương pháp mã hóa tối ưu:

**Huffman:**
* Xây từ dưới lên, gộp hai nút nhỏ nhất
* Tối ưu trong lớp prefix-free code ($L = L^*$)
* Độ dài trung bình $L^* \in [H/\log D, H/\log D + 1)$
* D-ary: gộp $D$ nút mỗi bước

**Arithmetic Coding:**
* Mã hóa toàn chuỗi bằng khoảng trong $[0,1)$
* Đạt hiệu suất $[H, H+1/n)$ bit/symbol
* Không cần codebook kích thước $|\mathcal{X}|^n$
* Nền tảng của các bộ nén hiện đại

Cả hai đều bị giới hạn từ dưới bởi $H(X)$ — entropy là rào cản tuyệt đối.

> Bài tiếp theo — **Mã Tunstall** — khám phá bài toán mã hóa **đối ngẫu**: thay vì mã hóa từng ký hiệu thành chuỗi bit độ dài biến, ta phân tích chuỗi ký hiệu thành các **cụm (phrase) có xác suất đồng đều** để mã hóa với **độ dài cố định**. Đây là góc nhìn mới hoàn toàn về bài toán nén.
