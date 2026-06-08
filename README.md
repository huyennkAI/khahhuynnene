# Chuyên đề Lý thuyết AI

Website tĩnh chia sẻ các chuyên đề lý thuyết Trí tuệ nhân tạo, tổ chức theo chuyên mục. Xây bằng HTML/CSS/JS thuần, nội dung viết bằng Markdown, hỗ trợ công thức toán (KaTeX) và tô màu code (highlight.js). Triển khai trên Cloudflare Pages.

## Cấu trúc

```
.
├── index.html              Khung trang và nạp thư viện
├── assets/
│   ├── css/style.css       Giao diện kiểu tài liệu (docs)
│   └── js/app.js           Định tuyến, dựng menu, render Markdown
└── content/
    ├── manifest.json       Danh sách chuyên mục và bài viết
    └── *.md                Mỗi file là một chuyên đề
```

## Thêm một chuyên đề mới

1. Tạo file `content/ten-bai.md` và viết nội dung bằng Markdown.
2. Mở `content/manifest.json`, thêm một mục vào chuyên mục phù hợp:

```json
{
  "slug": "ten-bai",
  "title": "Tên hiển thị",
  "desc": "Mô tả ngắn",
  "file": "content/ten-bai.md",
  "tags": ["tu-khoa"]
}
```

Muốn tạo chuyên mục mới thì thêm một phần tử vào mảng `groups` với `title` và `items` riêng.

## Công thức toán

Dùng cú pháp LaTeX: `$...$` cho inline và `$$...$$` cho block.

## Chạy thử cục bộ

Vì trang dùng `fetch` để nạp Markdown, cần một web server (mở trực tiếp bằng `file://` sẽ bị chặn):

```bash
python3 -m http.server 8080
```

Rồi mở http://localhost:8080

## Triển khai lên Cloudflare Pages (qua GitHub)

1. Đẩy thư mục này lên một repository GitHub.
2. Vào https://dash.cloudflare.com → Workers & Pages → Create → Pages → Connect to Git.
3. Chọn repository, đặt **Project name** là `khahhuynday` (để có địa chỉ `khahhuynday.pages.dev`).
4. Build settings: để **trống** lệnh build, **Build output directory** đặt là `/` (thư mục gốc).
5. Save and Deploy.

Mỗi lần push lên nhánh chính, Cloudflare sẽ tự động deploy lại.
