# Bài 3: Giới thiệu về mã (Codes)

> Ta đã biết entropy $H(X)$ là cận dưới lý thuyết cho việc biểu diễn thông tin. Bài này hỏi: **làm thế nào để thiết kế một phương án mã hóa thực sự đạt đến cận đó?** Câu trả lời bắt đầu bằng việc phân loại mã: mã nào có thể giải mã được duy nhất? Mã nào có thể giải mã ngay tức thì? Bất đẳng thức Kraft cho ta điều kiện cần và đủ cho sự tồn tại của mã tiền tố.
>
> $$\sum_{i} D^{-l_i} \le 1 \quad \text{(Kraft)} \qquad H(X)/\log D \le L^* \quad \text{(cận dưới Shannon)}$$

---

# 1. Bài toán mã hóa nguồn

## 1.1. Thiết lập bài toán

Ta có một nguồn tin phát ra các ký hiệu từ bảng chữ nguồn $\mathcal{X} = \{x_1, x_2, \ldots, x_m\}$ với phân phối $p(x_i) = p_i$. Ta muốn mã hóa từng ký hiệu thành một chuỗi từ bảng chữ mã $\mathcal{D} = \{0, 1, \ldots, D-1\}$ (thường $D=2$ cho mã nhị phân).

Một **mã** là ánh xạ $C: \mathcal{X} \to \mathcal{D}^*$ (trong đó $\mathcal{D}^*$ là tập tất cả các chuỗi hữu hạn trên $\mathcal{D}$). Ảnh $C(x_i)$ gọi là **codeword** (từ mã) của $x_i$, có độ dài $l_i = |C(x_i)|$.

## 1.2. Mục tiêu

Thiết kế mã sao cho:
1. **Có thể giải mã** — nhận được một chuỗi mã, ta phục hồi được dãy ký hiệu gốc.
2. **Ngắn** — độ dài trung bình $L = \sum_i p_i l_i$ nhỏ nhất có thể.

Hai mục tiêu này tạo nên sự đánh đổi: mã ngắn hơn có thể dẫn đến nhập nhằng.

## 1.3. Ví dụ khởi động

Nguồn $\mathcal{X} = \{a, b, c, d\}$ với $p = (1/2, 1/4, 1/8, 1/8)$.

**Mã A:**

| Ký hiệu | Xác suất | Codeword |
|---------|---------|---------|
| $a$ | $1/2$ | $0$ |
| $b$ | $1/4$ | $10$ |
| $c$ | $1/8$ | $110$ |
| $d$ | $1/8$ | $111$ |

Độ dài trung bình: $L = \frac{1}{2}\cdot 1 + \frac{1}{4}\cdot 2 + \frac{1}{8}\cdot 3 + \frac{1}{8}\cdot 3 = 0.5 + 0.5 + 0.375 + 0.375 = 1.75$ bit.

**Mã B (ngắn hơn nhưng có vấn đề):**

| Ký hiệu | Codeword |
|---------|---------|
| $a$ | $0$ |
| $b$ | $0$ |
| $c$ | $1$ |
| $d$ | $10$ |

Mã B có codeword trùng nhau ($a$ và $b$ đều được mã hóa thành $0$) — không thể giải mã!

---

# 2. Phân loại mã

## 2.1. Mã không nhập nhằng (Non-singular codes)

**Định nghĩa:** Mã $C$ là **không nhập nhằng** nếu ánh xạ $C: \mathcal{X} \to \mathcal{D}^*$ là đơn ánh (injective), tức là $x \ne x' \implies C(x) \ne C(x')$.

Điều này đảm bảo ta có thể phân biệt từng ký hiệu riêng lẻ, nhưng chưa đủ để giải mã một chuỗi dài.

**Ví dụ vấn đề:** $C(a) = 0$, $C(b) = 00$. Đây là mã không nhập nhằng, nhưng chuỗi mã $00$ có thể là $aa$ hay $b$?

## 2.2. Mở rộng mã (Extension of a code)

Với mã $C$, ta định nghĩa **mở rộng** $C^*: \mathcal{X}^* \to \mathcal{D}^*$ bằng cách ghép nối:
$$C^*(x_1 x_2 \cdots x_n) = C(x_1) C(x_2) \cdots C(x_n)$$

## 2.3. Mã giải mã duy nhất (Uniquely decodable codes)

**Định nghĩa:** Mã $C$ là **giải mã duy nhất** nếu mở rộng $C^*$ là đơn ánh — mọi chuỗi hữu hạn các ký hiệu nguồn đều có ảnh khác nhau.

Đây là điều kiện tối thiểu: nhận được bất kỳ chuỗi mã hóa nào, ta đều phục hồi được duy nhất một chuỗi ký hiệu nguồn.

## 2.4. Mã tiền tố (Prefix-free codes)

**Định nghĩa:** Mã $C$ là **prefix-free** (hay **instantaneous** — tức thời) nếu không có codeword nào là tiền tố (prefix) của codeword khác. Tức là với mọi $x \ne x'$: $C(x)$ không phải là tiền tố của $C(x')$.

**Tại sao "tức thời"?** Khi đọc luồng bit, ta có thể giải mã ngay khi nhận đủ bits của một codeword mà không cần đọc thêm bits tiếp theo. Không cần "nhìn trước" (lookahead).

**Mối quan hệ:** Prefix-free $\implies$ Uniquely decodable $\implies$ Non-singular.

Chiều ngược lại không nhất thiết đúng (xem ví dụ bên dưới).

## 2.5. Ví dụ phân loại

| Tên | Codewords | Loại |
|-----|-----------|------|
| Mã A | $0, 10, 110, 111$ | Prefix-free |
| Mã B | $0, 10, 110, 10$ | Không hợp lệ (trùng codeword) |
| Mã C | $10, 00, 11, 110, 011$ | Uniquely decodable nhưng không prefix-free ($10$ là prefix của... không có, nhưng $11$ là prefix của $110$) |
| Mã D | $0, 01, 011, 0111$ | Không uniquely decodable |

Kiểm tra mã D: $C^*(01) = C(0)C(1)?$ — nhưng $1 \notin \mathcal{X}$. Hơn nữa $C(01)$ có thể là $C(0)$ nếu $01$ cũng là codeword... Ta cần xét kỹ hơn.

**Ví dụ mã uniquely decodable nhưng không prefix-free:**

| Ký hiệu | Codeword |
|---------|---------|
| $a$ | $10$ |
| $b$ | $00$ |
| $c$ | $11$ |
| $d$ | $110$ |
| $e$ | $100$ |

Codeword $11$ là prefix của $110$, nên mã không prefix-free. Nhưng mã vẫn uniquely decodable (có thể kiểm tra bằng giải thuật Sardinas–Patterson).

---

# 3. Bất đẳng thức Kraft

## 3.1. Phát biểu định lý

**Định lý Kraft (1949):** Với bảng chữ mã kích thước $D$, tồn tại mã **prefix-free** với độ dài codeword $(l_1, l_2, \ldots, l_m)$ khi và chỉ khi:

$$\sum_{i=1}^{m} D^{-l_i} \le 1 \quad \text{(bất đẳng thức Kraft)}$$

## 3.2. Chứng minh: hình học hóa

Ta tưởng tượng **cây D-ary** (cây có D nhánh) vô hạn chiều sâu. Mỗi nút ở độ sâu $d$ tương ứng với một chuỗi $D$-ary độ dài $d$.

Với mỗi codeword $C(x_i)$ ở độ sâu $l_i$, ta "đánh dấu" nút tương ứng và **cắt bỏ** tất cả các con cháu của nó (vì codeword prefix-free không được xuất hiện như là con cháu của nhau).

Phần "đất" bị chiếm bởi codeword $i$ (tính theo phân số các nút ở độ sâu $L = \max l_i$) là $D^{L-l_i}$ nút. Tổng số nút ở độ sâu $L$ là $D^L$.

Vì các phần đất không chồng lên nhau (tính chất prefix-free):
$$\sum_i D^{L-l_i} \le D^L \implies \sum_i D^{-l_i} \le 1$$

**Chiều ngược:** Nếu $\sum D^{-l_i} \le 1$, ta có thể xây dựng mã prefix-free với độ dài đó: gán các nút của cây D-ary từ trái sang phải theo độ sâu.

## 3.3. Ví dụ minh họa

**Mã A** với $l = (1, 2, 3, 3)$ (nhị phân, $D=2$):
$$\sum 2^{-l_i} = 2^{-1} + 2^{-2} + 2^{-3} + 2^{-3} = 0.5 + 0.25 + 0.125 + 0.125 = 1 \le 1 \checkmark$$

**Bộ độ dài** $l = (1, 1, 2, 2)$:
$$\sum 2^{-l_i} = 0.5 + 0.5 + 0.25 + 0.25 = 1.5 > 1$$

Không tồn tại mã prefix-free với bộ độ dài này (có thể tồn tại mã uniquely decodable nhưng không prefix-free).

---

# 4. Định lý McMillan

## 4.1. Phát biểu

**Định lý McMillan (1956):** Nếu mã $C$ là **uniquely decodable** với bảng chữ $D$, thì bộ độ dài $(l_1, \ldots, l_m)$ cũng thỏa bất đẳng thức Kraft:

$$\sum_{i=1}^{m} D^{-l_i} \le 1$$

## 4.2. Ý nghĩa

McMillan nói rằng: **bất kỳ mã giải mã duy nhất nào cũng phải trả cùng "cái giá" về độ dài như mã prefix-free**. Vì vậy ta không thể thu được lợi thế về độ dài bằng cách dùng mã uniquely decodable phức tạp hơn — prefix-free code đã là tốt nhất trong lớp.

**Hệ quả:** Khi thiết kế mã tối ưu, ta chỉ cần xét prefix-free codes mà không mất tính tổng quát.

## 4.3. Chứng minh ý tưởng

Xét $K = \sum_i D^{-l_i}$ và tính $K^n$:
$$K^n = \left(\sum_i D^{-l_i}\right)^n = \sum_{(i_1,\ldots,i_n)} D^{-(l_{i_1}+\cdots+l_{i_n})}$$

Đặt $l_{\max} = \max_i l_i$. Ta nhóm theo tổng độ dài $s = l_{i_1}+\cdots+l_{i_n}$:

$$K^n = \sum_{s=n}^{nl_{\max}} a_s D^{-s}$$

trong đó $a_s$ = số chuỗi ký hiệu độ dài $n$ có tổng độ dài mã bằng $s$.

Vì $C^*$ là đơn ánh (uniquely decodable), các chuỗi mã tương ứng phân biệt nhau, nên $a_s \le D^s$ (số chuỗi $D$-ary độ dài $s$):

$$K^n \le \sum_{s=n}^{nl_{\max}} D^s \cdot D^{-s} = nl_{\max} - n + 1 \le nl_{\max}$$

$$\implies K \le (nl_{\max})^{1/n} \xrightarrow{n\to\infty} 1$$

Vậy $K \le 1$.

---

# 5. Độ dài trung bình và cận dưới Shannon

## 5.1. Độ dài trung bình

Với mã $C$ có độ dài codeword $(l_1, \ldots, l_m)$ và nguồn $X$ với phân phối $(p_1, \ldots, p_m)$:

$$L(C) = \sum_{i=1}^{m} p_i l_i = \mathbb{E}[l(X)]$$

## 5.2. Cận dưới: độ dài trung bình không thể nhỏ hơn entropy

**Định lý:** Với mọi mã giải mã duy nhất $C$ trên bảng chữ $D$:

$$L(C) \ge \frac{H(X)}{\log_2 D}$$

**Chứng minh:**

Đặt $r_i = D^{-l_i} / \sum_j D^{-l_j}$. Vì Kraft: $\sum_j D^{-l_j} \le 1$, đặt $\sum_j D^{-l_j} = c \le 1$.

$$L - \frac{H}{\log D} = \sum_i p_i l_i - \sum_i p_i \log_D \frac{1}{p_i}$$
$$= \sum_i p_i \left(l_i - \log_D \frac{1}{p_i}\right) = \sum_i p_i \log_D(p_i D^{l_i})$$

Bây giờ dùng bất đẳng thức $\ln t \le t-1$:
$$\sum_i p_i \log_D(p_i D^{l_i}) \ge 0$$

Vì $\sum_i p_i \log_D(p_i D^{l_i}) = -D_{KL}(p \| q)$ với $q_i = D^{-l_i}/c \le D^{-l_i}$... Cách chứng minh sạch hơn:

$$L - \frac{H}{\log D} = \sum_i p_i \log_D \frac{p_i}{D^{-l_i}} \ge \sum_i p_i \log_D \frac{p_i}{\sum_j p_j D^{-l_j+l_i}} $$

Thực ra ta dùng trực tiếp $D_{KL} \ge 0$: đặt $q_i \propto D^{-l_i}$, chuẩn hóa thành phân phối. Thì $D_{KL}(p\|q) \ge 0$ cho ra $H(p,q) \ge H(p)$, tức $\sum p_i l_i \log D \ge H(X)$.

## 5.3. Cận dưới không đạt được khi nào?

Cận dưới $H(X)/\log D$ đạt được khi và chỉ khi $l_i = \log_D(1/p_i)$ với mọi $i$ đều là số nguyên. Điều này đòi hỏi $p_i = D^{-l_i}$, tức phân phối là lũy thừa của $1/D$. Trong trường hợp tổng quát, ta phải làm tròn lên và chấp nhận dư ra tối đa 1 bit.

---

# 6. Ví dụ tính toán cụ thể

## 6.1. Bài toán thiết kế mã

Nguồn $\mathcal{X} = \{a, b, c, d, e\}$ với phân phối:

| Ký hiệu | Xác suất |
|---------|---------|
| $a$ | $0.5$ |
| $b$ | $0.25$ |
| $c$ | $0.125$ |
| $d$ | $0.0625$ |
| $e$ | $0.0625$ |

**Entropy:**
$$H(X) = -0.5\log 0.5 - 0.25\log 0.25 - 0.125\log 0.125 - 2\cdot 0.0625\log 0.0625$$
$$= 0.5 + 0.5 + 0.375 + 2\cdot 0.25 = 1.875 \text{ bit}$$

**Cận dưới độ dài trung bình:** $L^* \ge 1.875$ bit.

**Mã prefix-free tối ưu (có thể kiểm tra Kraft):**

| Ký hiệu | Xác suất | Codeword | Độ dài |
|---------|---------|---------|--------|
| $a$ | $0.5$ | $0$ | $1$ |
| $b$ | $0.25$ | $10$ | $2$ |
| $c$ | $0.125$ | $110$ | $3$ |
| $d$ | $0.0625$ | $1110$ | $4$ |
| $e$ | $0.0625$ | $1111$ | $4$ |

**Kiểm tra Kraft:** $2^{-1}+2^{-2}+2^{-3}+2^{-4}+2^{-4} = 0.5+0.25+0.125+0.0625+0.0625 = 1 \le 1$ ✓

**Độ dài trung bình:**
$$L = 0.5\cdot 1 + 0.25\cdot 2 + 0.125\cdot 3 + 0.0625\cdot 4 + 0.0625\cdot 4 = 0.5+0.5+0.375+0.25+0.25 = 1.875 \text{ bit}$$

Đây là trường hợp hiếm có: $p_i$ đều là lũy thừa của $1/2$, nên ta đạt đúng $H(X)$!

## 6.2. Phân phối tổng quát

Nguồn $\mathcal{X} = \{a, b, c\}$ với $p = (0.5, 0.3, 0.2)$:

$$H(X) = -0.5\log 0.5 - 0.3\log 0.3 - 0.2\log 0.2 = 0.5 + 0.521 + 0.464 = 1.485 \text{ bit}$$

Thử mã: $l=(1,2,2)$. Kraft: $2^{-1}+2^{-2}+2^{-2} = 0.5+0.25+0.25 = 1 \le 1$ ✓.

$L = 0.5\cdot 1 + 0.3\cdot 2 + 0.2\cdot 2 = 0.5 + 0.6 + 0.4 = 1.5$ bit.

$1.5 \ge H(X) = 1.485$ ✓ (cách đúng 0.015 bit — rất gần tối ưu).

---

# 7. Mối quan hệ giữa các loại mã

## 7.1. Phân cấp mã

```
Tất cả mã
    └── Non-singular (không nhập nhằng)
            └── Uniquely decodable (giải mã duy nhất)
                    └── Prefix-free (tiền tố / tức thời)
```

## 7.2. Khi nào dùng mã không prefix-free?

Trong lý thuyết, mã uniquely decodable cho ta thêm tự do trong thiết kế. Nhưng định lý McMillan nói rằng ta không thu được lợi gì về độ dài trung bình. Trong thực tế, prefix-free codes được ưa dùng vì:

* **Giải mã tức thời:** Không cần buffer toàn bộ chuỗi trước khi giải mã.
* **Đơn giản:** Cấu trúc cây nhị phân dễ implement.
* **Tối ưu:** Huffman code là prefix-free và tối ưu trong lớp prefix-free (= tối ưu trong lớp uniquely decodable).

---

# 8. Ý nghĩa của bất đẳng thức Kraft

## 8.1. Kraft như là ràng buộc ngân sách

Hãy nghĩ đến $D^{-l_i}$ như là "chi phí" của codeword độ dài $l_i$. Kraft nói rằng tổng chi phí phải $\le 1$. Muốn codeword ngắn ($l_i$ nhỏ = chi phí cao), thì phải đánh đổi bằng codeword dài hơn ở chỗ khác.

## 8.2. Đẳng thức Kraft khi nào đạt?

Khi $\sum D^{-l_i} = 1$, cây mã "đầy" — mọi nhánh đều bị chiếm dụng, không còn không gian trống. Đây là mã tiền tố **hoàn chỉnh (complete)**. Mã Huffman luôn cho $\sum D^{-l_i} = 1$ với phân phối tổng quát.

## 8.3. Kết nối với entropy

Nếu $l_i = \log_D(1/p_i)$ (không nhất thiết nguyên), Kraft inequality trở thành đẳng thức:
$$\sum D^{-l_i} = \sum D^{-\log_D(1/p_i)} = \sum p_i = 1$$

Và độ dài trung bình bằng entropy: $L = \sum p_i \log_D(1/p_i) = H(X)/\log D$.

Đây là lý do ta muốn gán $l_i \approx \log_D(1/p_i)$ — codeword ngắn cho ký hiệu xác suất cao, codeword dài cho ký hiệu xác suất thấp.

---

# 9. Tổng kết

Bài này xây dựng nền tảng lý thuyết mã hóa:

* **Phân cấp:** Prefix-free $\subsetneq$ Uniquely decodable $\subsetneq$ Non-singular
* **Kraft:** $\sum D^{-l_i} \le 1$ là điều kiện cần và đủ để tồn tại prefix-free code với bộ độ dài $(l_1,\ldots,l_m)$
* **McMillan:** Mọi uniquely decodable code cũng thỏa Kraft — không có lợi thế khi dùng mã phức tạp hơn prefix-free
* **Cận dưới Shannon:** $L \ge H(X)/\log D$ — entropy là rào cản không thể vượt qua
* **Hướng tối ưu:** Gán $l_i \approx \log_D(1/p_i)$ để tiệm cận cận dưới

> Bài tiếp theo — **Dãy điển hình (Typical Sequences)** — sẽ cho ta công cụ mạnh nhất để *chứng minh* rằng ta có thể nén xuống $H(X)$ bit/ký hiệu khi mã hóa nhiều ký hiệu cùng nhau. Định lý AEP (Asymptotic Equipartition Property) — phiên bản thông tin của Luật số lớn — là trái tim của chứng minh này.
