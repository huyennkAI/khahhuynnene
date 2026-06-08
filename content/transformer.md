# Kiến trúc Transformer

> Transformer, giới thiệu năm 2017 trong bài báo *"Attention Is All You Need"*, là kiến trúc nền tảng của hầu hết mô hình ngôn ngữ lớn hiện nay như GPT, BERT hay Claude.

## Vấn đề với mô hình tuần tự

Các mạng RNN/LSTM xử lý chuỗi **từng bước một**, gây ra hai hạn chế:

- Khó học phụ thuộc xa giữa các từ.
- Không song song hóa được, huấn luyện chậm.

Transformer giải quyết bằng cách nhìn toàn bộ chuỗi cùng lúc qua cơ chế **attention**.

## Self-Attention

Ý tưởng: với mỗi từ, mô hình tính xem nó nên "chú ý" đến những từ nào khác trong câu. Mỗi từ được chiếu thành ba vector:

- **Query** $Q$ — "tôi đang tìm gì".
- **Key** $K$ — "tôi chứa thông tin gì".
- **Value** $V$ — "thông tin thực sự".

Trọng số chú ý được tính bằng:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{QK^\top}{\sqrt{d_k}} \right) V$$

Hệ số $\sqrt{d_k}$ giúp ổn định gradient khi số chiều lớn.

## Multi-Head Attention

Thay vì một phép attention, Transformer dùng nhiều "đầu" song song, mỗi đầu học một kiểu quan hệ khác nhau (cú pháp, ngữ nghĩa, vị trí...):

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

## Các thành phần khác

- **Positional Encoding** — vì attention không có khái niệm thứ tự, ta cộng thêm tín hiệu vị trí vào embedding.
- **Feed-Forward Network** — mạng MLP áp dụng độc lập cho từng vị trí.
- **Residual + Layer Norm** — giúp huấn luyện mạng rất sâu ổn định.

```text
Đầu vào → Embedding + Positional Encoding
        → [ Multi-Head Attention → Add & Norm
            → Feed-Forward → Add & Norm ] × N lớp
        → Đầu ra
```

## Vì sao Transformer thống trị?

1. **Song song hóa hoàn toàn** — tận dụng tối đa GPU.
2. **Bắt phụ thuộc xa** — mọi từ kết nối trực tiếp với mọi từ.
3. **Mở rộng cực tốt** — hiệu năng tăng đều khi tăng dữ liệu và tham số (scaling laws).

> Từ kiến trúc này, chỉ cần thay đổi cách huấn luyện, ta có cả các mô hình hiểu ngôn ngữ (BERT) lẫn sinh ngôn ngữ (GPT). Đây là viên gạch nền của kỷ nguyên LLM.
