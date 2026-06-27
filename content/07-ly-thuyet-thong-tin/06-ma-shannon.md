# Bài 6: Mã Shannon

> Mã Shannon trả lời câu hỏi xây dựng: **làm thế nào để tường minh gán codeword cho từng ký hiệu?** Ý tưởng đơn giản và đẹp: mỗi ký hiệu $x_i$ "xứng đáng" được $\log_D(1/p_i)$ chữ số. Ta làm tròn lên số nguyên và dùng xác suất tích lũy làm địa chỉ trong cây mã. Kết quả đảm bảo prefix-free và độ dài trung bình trong khoảng $[H/\log D, H/\log D + 1)$.
>
> $$l_i = \left\lceil \log_D \frac{1}{p_i} \right\rceil, \qquad \frac{H(X)}{\log_2 D} \le L_{\text{Shannon}} < \frac{H(X)}{\log_2 D} + 1$$

---

# 1. Xây dựng mã Shannon

## 1.1. Ý tưởng

Từ bài trước, ta biết rằng bộ độ dài $l_i = \lceil\log_D(1/p_i)\rceil$ thỏa bất đẳng thức Kraft. Nhưng làm thế nào để gán codeword cụ thể cho từng độ dài đó?

Phương pháp Shannon dùng **xác suất tích lũy (cumulative probability)** để tạo địa chỉ độc nhất và tự nhiên cho mỗi ký hiệu.

## 1.2. Thuật toán (nhị phân, $D=2$)

**Bước 1:** Sắp xếp các ký hiệu theo thứ tự bất kỳ (thường theo xác suất giảm dần).

**Bước 2:** Tính xác suất tích lũy:
$$F_i = \sum_{j < i} p_j \quad \text{(xác suất của tất cả ký hiệu trước } x_i\text{)}$$

Với $x_1$ là ký hiệu đầu tiên: $F_1 = 0$.

**Bước 3:** Gán codeword bằng cách lấy $l_i = \lceil\log_2(1/p_i)\rceil$ chữ số nhị phân đầu của $F_i$ (biểu diễn nhị phân của $F_i$ trong $[0,1)$).

## 1.3. Biểu diễn nhị phân của số thực

Số thực $F_i \in [0,1)$ được biểu diễn nhị phân:
$$F_i = \sum_{k=1}^{\infty} b_k 2^{-k} = 0.b_1 b_2 b_3 \ldots \text{ (nhị phân)}$$

Codeword của $x_i$ là $l_i$ bit đầu tiên: $c_i = b_1 b_2 \ldots b_{l_i}$.

---

# 2. Chứng minh Shannon code là prefix-free

## 2.1. Điều kiện prefix-free

Hai codeword $c_i$ và $c_j$ (với $i \ne j$) không phải là prefix của nhau khi và chỉ khi các khoảng $[F_i, F_i + 2^{-l_i})$ và $[F_j, F_j + 2^{-l_j})$ rời nhau.

## 2.2. Chứng minh rời nhau

Không mất tính tổng quát giả sử $i < j$ (theo thứ tự sắp xếp), nên $F_j = F_i + \sum_{i \le k < j} p_k \ge F_i + p_i$.

Khoảng của $x_i$ là $[F_i, F_i + 2^{-l_i})$. Ta cần $F_j \ge F_i + 2^{-l_i}$.

Vì $l_i = \lceil\log_2(1/p_i)\rceil \ge \log_2(1/p_i)$:
$$2^{-l_i} \le 2^{-\log_2(1/p_i)} = p_i$$

Nên $F_j \ge F_i + p_i \ge F_i + 2^{-l_i}$ — hai khoảng không chồng lên nhau.

Vì các khoảng rời nhau, hai codeword không thể là prefix của nhau (nếu $c_i$ là prefix của $c_j$, thì $F_j$ rơi vào khoảng của $c_i$).

---

# 3. Ví dụ tính toán chi tiết

## 3.1. Bài toán

Nguồn $\mathcal{X} = \{a, b, c, d\}$ với phân phối:

| Ký hiệu | $p_i$ | $-\log_2 p_i$ | $l_i = \lceil -\log_2 p_i \rceil$ |
|---------|-------|----------------|-----------------------------------|
| $a$ | $0.5$ | $1.0$ | $1$ |
| $b$ | $0.25$ | $2.0$ | $2$ |
| $c$ | $0.125$ | $3.0$ | $3$ |
| $d$ | $0.125$ | $3.0$ | $3$ |

## 3.2. Tính xác suất tích lũy và codeword

| Ký hiệu | $p_i$ | $F_i$ (thập phân) | $F_i$ (nhị phân) | Codeword ($l_i$ bit đầu) |
|---------|-------|----------|---------|---------|
| $a$ | $0.5$ | $0.000$ | $0.0000\ldots$ | $0$ |
| $b$ | $0.25$ | $0.500$ | $0.1000\ldots$ | $10$ |
| $c$ | $0.125$ | $0.750$ | $0.1100\ldots$ | $110$ |
| $d$ | $0.125$ | $0.875$ | $0.1110\ldots$ | $111$ |

**Kiểm tra Kraft:** $2^{-1} + 2^{-2} + 2^{-3} + 2^{-3} = 0.5 + 0.25 + 0.125 + 0.125 = 1$ ✓

**Độ dài trung bình:**
$$L = 0.5\cdot 1 + 0.25\cdot 2 + 0.125\cdot 3 + 0.125\cdot 3 = 0.5+0.5+0.375+0.375 = 1.75 \text{ bit}$$

**Entropy:** $H(X) = 0.5\cdot 1 + 0.25\cdot 2 + 0.125\cdot 3 + 0.125\cdot 3 = 1.75$ bit.

Trường hợp đặc biệt: phân phối là lũy thừa của $1/2$ nên Shannon code đạt đúng $H(X)$!

## 3.3. Ví dụ với phân phối không đều

Nguồn $\mathcal{X} = \{a, b, c\}$ với $p = (0.5, 0.3, 0.2)$:

$H(X) = -0.5\log 0.5 - 0.3\log 0.3 - 0.2\log 0.2 = 0.5 + 0.521 + 0.464 = 1.485$ bit.

| Ký hiệu | $p_i$ | $-\log_2 p_i$ | $l_i$ | $F_i$ (thập phân) | Nhị phân $F_i$ | Codeword |
|---------|-------|----------------|-------|----------|---------|---------|
| $a$ | $0.5$ | $1.0$ | $1$ | $0.0$ | $0.0000\ldots$ | $0$ |
| $b$ | $0.3$ | $1.737$ | $2$ | $0.5$ | $0.1000\ldots$ | $10$ |
| $c$ | $0.2$ | $2.322$ | $3$ | $0.8$ | $0.1100\ldots$ | $110$ |

Kiểm tra Kraft: $2^{-1} + 2^{-2} + 2^{-3} = 0.5 + 0.25 + 0.125 = 0.875 < 1$ ✓ (không đầy)

$L = 0.5\cdot 1 + 0.3\cdot 2 + 0.2\cdot 3 = 0.5 + 0.6 + 0.6 = 1.7$ bit.

$H(X) = 1.485 \le L = 1.7 < H(X) + 1 = 2.485$ ✓

---

# 4. Tại sao $l_i = \lceil\log_D(1/p_i)\rceil$ là cận?

## 4.1. Ý nghĩa trực giác

Ký hiệu xác suất $p_i$ "nên được" mã bằng $\log_D(1/p_i)$ chữ số. Đây là số chữ số đủ để "lưu trữ" lượng thông tin $I(x_i) = \log(1/p_i)$. Làm tròn lên cho ra số nguyên và mất thêm tối đa 1 đơn vị.

## 4.2. Ký tự "xứng đáng" nhận bao nhiêu bit?

Xét từ góc độ người nhận: nếu ta nhận được codeword độ dài $l_i$, ta biết đây là một trong $D^{l_i}$ khả năng. Chi phí "định danh" ký hiệu $x_i$ trong bảng chữ $D$-ary là $\log_D(1/p_i)$ chữ số — chính xác bằng lượng thông tin.

Mã Shannon hiểu theo nghĩa: **mỗi ký hiệu được mã theo đúng "giá trị thông tin" của nó** (làm tròn lên).

---

# 5. So sánh Shannon code với Huffman code

## 5.1. Huffman luôn tốt hơn hoặc bằng

Cả Shannon code và Huffman code đều thỏa bất đẳng thức $H/\log D \le L < H/\log D + 1$. Nhưng Huffman code là **tối ưu** trong lớp prefix-free codes, còn Shannon code chỉ thỏa cận trên.

**Ví dụ cho thấy sự khác biệt:**

Nguồn $\mathcal{X} = \{a, b\}$ với $p = (0.9, 0.1)$, $H(X) = H_b(0.9) = 0.469$ bit.

| | Shannon code | Huffman code |
|---|---|---|
| $l_a$ | $\lceil\log_2(1/0.9)\rceil = \lceil 0.152 \rceil = 1$ | $1$ |
| $l_b$ | $\lceil\log_2(1/0.1)\rceil = \lceil 3.322 \rceil = 4$ | $1$ |
| $L$ | $0.9\cdot 1 + 0.1\cdot 4 = 1.3$ bit | $0.9\cdot 1 + 0.1\cdot 1 = 1.0$ bit |

Shannon code: 1.3 bit vs Huffman: 1.0 bit — Huffman tốt hơn đáng kể cho phân phối lệch này.

## 5.2. Khi nào Shannon code bằng Huffman?

Khi $p_i$ đều là lũy thừa của $1/D$: $p_i = D^{-k_i}$ cho $k_i \in \mathbb{Z}_{>0}$. Khi đó $\log_D(1/p_i)$ đã là số nguyên, và Shannon code = Huffman code.

---

# 6. Shannon-Fano code (khác với Shannon code)

## 6.1. Định nghĩa

**Shannon-Fano code** (Fano 1949, độc lập với Shannon) là phương pháp chia đệ quy:

1. Sắp xếp ký hiệu theo xác suất giảm dần.
2. Chia thành hai nhóm sao cho tổng xác suất hai nhóm gần bằng nhau nhất.
3. Gán bit $0$ cho nhóm đầu, bit $1$ cho nhóm sau.
4. Đệ quy cho mỗi nhóm.

## 6.2. Ví dụ

$p = (0.4, 0.2, 0.2, 0.1, 0.1)$. Sắp xếp: $a(0.4), b(0.2), c(0.2), d(0.1), e(0.1)$.

**Bước 1:** Chia $\{a, b\}$ (tổng $0.6$) và $\{c, d, e\}$ (tổng $0.4$) — không cân bằng.

Hoặc chia $\{a\}$ ($0.4$) và $\{b, c, d, e\}$ ($0.6$) — tốt hơn.

Thực ra Shannon-Fano thường không tối ưu và kém hơn Huffman. Ví dụ này cho thấy việc chia không phải lúc nào cũng rõ ràng.

## 6.3. Phân biệt Shannon code và Shannon-Fano code

| | Shannon code | Shannon-Fano code |
|---|---|---|
| Tác giả | Claude Shannon | Robert Fano (theo gợi ý của Shannon) |
| Phương pháp | Dựa trên xác suất tích lũy | Chia đệ quy theo tổng xác suất |
| Tối ưu? | Không | Không |
| Hiệu suất | Xấp xỉ nhau, đôi khi Shannon-Fano tốt hơn |

---

# 7. Shannon code cho bảng chữ D-ary

## 7.1. Mở rộng tự nhiên

Với bảng chữ mã $\mathcal{D} = \{0, 1, \ldots, D-1\}$, thuật toán Shannon tương tự:

$$l_i = \lceil\log_D(1/p_i)\rceil$$

Codeword: lấy $l_i$ "chữ số" cơ số $D$ của xác suất tích lũy $F_i$.

## 7.2. Ví dụ D=3 (ternary)

Nguồn $\mathcal{X} = \{a, b, c, d\}$ với $p = (1/3, 1/3, 1/6, 1/6)$.

$H(X) = 2\cdot(1/3)\log_3 3 + 2\cdot(1/6)\log_3 6 = 2/3 + (1/3)\log_3 6 \approx 1.19$ trit.

| Ký hiệu | $p_i$ | $F_i$ | $l_i = \lceil\log_3(1/p_i)\rceil$ | Codeword |
|---------|-------|-------|------|---------|
| $a$ | $1/3$ | $0$ | $1$ | $0$ |
| $b$ | $1/3$ | $1/3$ | $1$ | $1$ |
| $c$ | $1/6$ | $2/3$ | $2$ | $20$ |
| $d$ | $1/6$ | $5/6$ | $2$ | $21$ |

$F_c = 2/3$ trong cơ số 3: $2/3 = 2 \cdot 3^{-1}$, nên hai chữ số đầu là $20$.
$F_d = 5/6 = 0.2\overline{1}_3$, hai chữ số đầu là $21$.

Kraft: $3^{-1} + 3^{-1} + 3^{-2} + 3^{-2} = 1/3 + 1/3 + 1/9 + 1/9 = 8/9 < 1$ ✓

$L = 1/3\cdot 1 + 1/3\cdot 1 + 1/6\cdot 2 + 1/6\cdot 2 = 2/3 + 2/3 = 4/3$ trit.

$H/\log_2 D = 1.19/\log_2 3 = 1.19/1.585 = 0.75$ trit — so với $L=1.33$ trit. Còn khoảng trống, nhưng trong cận cho phép.

---

# 8. Mã Shannon như là mã tiệm cận

## 8.1. Khi phân phối không đều nhiều

Shannon code không tốt cho phân phối rất lệch (ví dụ $p_1 = 0.999$, còn lại chia đều). Trong trường hợp này:

* $l_1 = \lceil\log_2(1/0.999)\rceil = \lceil 0.00145 \rceil = 1$ — hợp lý
* Nhưng $l_j$ cho các ký hiệu hiếm có thể rất dài

Block coding giải quyết vấn đề này, nhưng Shannon code cho từng ký hiệu đã đủ tốt về mặt lý thuyết.

## 8.2. Tại sao vẫn quan trọng?

Shannon code quan trọng vì:
1. **Lịch sử:** Là phương pháp xây dựng đầu tiên, chứng minh sự tồn tại của mã tốt.
2. **Nền tảng lý thuyết:** Xác suất tích lũy là nền tảng của Arithmetic Coding (bài 7).
3. **Đơn giản:** Dễ implement và phân tích.

---

# 9. Liên hệ với Arithmetic Coding

## 9.1. Shannon code là phiên bản "thô" của Arithmetic Coding

Arithmetic Coding (bài 7) mở rộng ý tưởng của Shannon code: thay vì mã hóa từng ký hiệu riêng lẻ với độ dài $l_i = \lceil\log(1/p_i)\rceil$, nó mã hóa **toàn bộ chuỗi** bằng cách dùng một khoảng trong $[0,1)$ có độ dài chính xác $= p(x^n)$.

Kết quả: Arithmetic Coding đạt hiệu suất của block coding (tốt hơn $1/n$) với chi phí $O(n)$ thay vì codebook kích thước $|\mathcal{X}|^n$.

## 9.2. Khoảng xác suất

Trong Shannon code, ký hiệu $x_i$ được biểu diễn bởi khoảng $[F_i, F_i + p_i)$. Codeword là xấp xỉ nhị phân của một điểm trong khoảng này.

Arithmetic Coding thu hẹp khoảng dần theo từng ký hiệu mới: khoảng cho chuỗi $(x_1, x_2)$ là $[F_{x_1} + p(x_1)F_{x_2}, F_{x_1} + p(x_1)(F_{x_2}+p(x_2)))$.

---

# 10. Tổng kết

Mã Shannon là cầu nối giữa lý thuyết (entropy là cận dưới) và thực hành (cách gán codeword cụ thể):

**Thuật toán:**
1. Gán $l_i = \lceil\log_D(1/p_i)\rceil$
2. Tính $F_i = \sum_{j<i} p_j$
3. Codeword = $l_i$ chữ số đầu tiên của $F_i$ trong cơ số $D$

**Tính chất:**
* Luôn là prefix-free (chứng minh qua sự rời nhau của các khoảng)
* $H/\log D \le L_{\text{Shannon}} < H/\log D + 1$
* Không nhất thiết tối ưu (Huffman code tốt hơn)
* Xác suất tích lũy là nền tảng của Arithmetic Coding

**Ý nghĩa triết học:** Mã Shannon "lắng nghe" phân phối — mỗi ký hiệu nhận độ dài codeword tỉ lệ với lượng thông tin tự nhiên của nó. Không ký hiệu nào được đối xử bất công.

> Bài tiếp theo — **Mã Huffman và Arithmetic Coding** — sẽ xây dựng mã tối ưu thực sự (Huffman) và mã đạt hiệu suất tiệm cận $H(X)$ mà không cần block coding (Arithmetic Coding). Huffman giải quyết vấn đề tối ưu hóa, còn Arithmetic Coding giải quyết vấn đề kích thước codebook.
