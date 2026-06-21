(function () {
  "use strict";

  var article = document.getElementById("article");
  var nav = document.getElementById("nav");
  var toc = document.getElementById("toc");
  var search = document.getElementById("search");
  var prevLink = document.getElementById("prevLink");
  var nextLink = document.getElementById("nextLink");
  var sidebar = document.getElementById("sidebar");
  var overlay = document.getElementById("overlay");
  var menuToggle = document.getElementById("menuToggle");
  var themeToggle = document.getElementById("themeToggle");

  var manifest = null;
  var flat = [];

  marked.setOptions({
    highlight: function (code, lang) {
      try {
        if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
        return hljs.highlightAuto(code).value;
      } catch (e) { return code; }
    }
  });

  // Toán học: cho marked bỏ qua nội dung trong $...$ và $$...$$ rồi render thẳng
  // bằng KaTeX. Nếu không, marked sẽ xử lý "_" (subscript) thành <em>, nuốt "\\",
  // làm hỏng công thức (KaTeX không nhận ra delimiter -> hiện mã thô).
  function renderTex(tex, displayMode) {
    try {
      return window.katex.renderToString(tex.trim(), {
        displayMode: displayMode,
        throwOnError: false
      });
    } catch (e) {
      return '<code class="math-error">' + escapeHtml(tex) + "</code>";
    }
  }

  marked.use({
    extensions: [
      {
        name: "blockMath",
        level: "block",
        start: function (src) { var i = src.indexOf("$$"); return i < 0 ? undefined : i; },
        tokenizer: function (src) {
          var m = /^\$\$([\s\S]+?)\$\$/.exec(src);
          if (m) return { type: "blockMath", raw: m[0], text: m[1] };
        },
        renderer: function (token) { return renderTex(token.text, true); }
      },
      {
        name: "inlineMath",
        level: "inline",
        start: function (src) { var i = src.indexOf("$"); return i < 0 ? undefined : i; },
        tokenizer: function (src) {
          var m = /^\$([^\$\n]+?)\$/.exec(src);
          if (m) return { type: "inlineMath", raw: m[0], text: m[1] };
        },
        renderer: function (token) { return renderTex(token.text, false); }
      }
    ]
  });

  function slugify(text) {
    return text.toLowerCase().trim()
      .replace(/[àáạảãâầấậẩẫăằắặẳẵ]/g, "a")
      .replace(/[èéẹẻẽêềếệểễ]/g, "e")
      .replace(/[ìíịỉĩ]/g, "i")
      .replace(/[òóọỏõôồốộổỗơờớợởỡ]/g, "o")
      .replace(/[ùúụủũưừứựửữ]/g, "u")
      .replace(/[ỳýỵỷỹ]/g, "y")
      .replace(/đ/g, "d")
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");
  }

  function initTheme() {
    var saved = localStorage.getItem("theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches)
      document.documentElement.setAttribute("data-theme", "dark");
  }
  themeToggle.addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", cur);
    localStorage.setItem("theme", cur);
  });

  function openSidebar() { sidebar.classList.add("open"); overlay.classList.add("show"); }
  function closeSidebar() { sidebar.classList.remove("open"); overlay.classList.remove("show"); }
  menuToggle.addEventListener("click", function () {
    if (sidebar.classList.contains("open")) closeSidebar(); else openSidebar();
  });
  overlay.addEventListener("click", closeSidebar);

  function buildFlat() {
    flat = [];
    manifest.groups.forEach(function (g) {
      g.items.forEach(function (it) { flat.push(it); });
    });
  }

  function renderNav(filter) {
    nav.innerHTML = "";
    var q = (filter || "").toLowerCase().trim();
    manifest.groups.forEach(function (g) {
      var matched = g.items.filter(function (it) {
        if (!q) return true;
        return (it.title + " " + (it.desc || "") + " " + (it.tags || []).join(" ")).toLowerCase().indexOf(q) !== -1;
      });
      if (!matched.length) return;
      var group = document.createElement("div");
      group.className = "nav-group";
      var title = document.createElement("div");
      title.className = "nav-group-title";
      title.textContent = g.title;
      group.appendChild(title);
      matched.forEach(function (it) {
        var a = document.createElement("a");
        a.className = "nav-link";
        a.href = "#/" + it.slug;
        a.dataset.slug = it.slug;
        var idx = flat.findIndex(function (f) { return f.slug === it.slug; });
        var num = (idx >= 0) ? String(idx + 1).padStart(2, '0') + '.' : '';
        var spanNum = document.createElement('span');
        spanNum.className = 'nav-num';
        spanNum.textContent = num;
        var spanTitle = document.createElement('span');
        spanTitle.className = 'nav-title';
        spanTitle.textContent = it.title;
        a.appendChild(spanNum);
        a.appendChild(spanTitle);
        group.appendChild(a);
      });
      nav.appendChild(group);
    });
    highlightActive();
  }

  function highlightActive() {
    var current = currentSlug();
    var links = nav.querySelectorAll(".nav-link");
    for (var i = 0; i < links.length; i++)
      links[i].classList.toggle("active", links[i].dataset.slug === current);
  }

  function currentSlug() {
    return location.hash.replace(/^#\/?/, "");
  }

  function buildToc() {
    toc.innerHTML = "";
    var heads = article.querySelectorAll("h2, h3");
    var used = {};
    heads.forEach(function (h) {
      var base = slugify(h.textContent) || "muc";
      var id = base, n = 1;
      while (used[id]) { id = base + "-" + (++n); }
      used[id] = true;
      h.id = id;
      var a = document.createElement("a");
      a.href = "#" + id;
      a.textContent = h.textContent;
      a.className = h.tagName.toLowerCase();
      a.addEventListener("click", function (e) {
        e.preventDefault();
        var el = document.getElementById(id);
        if (el) el.scrollIntoView();
        history.replaceState(null, "", "#/" + currentSlug());
      });
      toc.appendChild(a);
    });
  }

  var tocObserver = null;
  function observeHeadings() {
    if (tocObserver) tocObserver.disconnect();
    var links = {};
    toc.querySelectorAll("a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    tocObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          Object.keys(links).forEach(function (k) { links[k].classList.remove("active"); });
          if (links[en.target.id]) links[en.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-70px 0px -75% 0px" });
    article.querySelectorAll("h2, h3").forEach(function (h) { tocObserver.observe(h); });
  }

  function renderMath() {
    if (window.renderMathInElement) {
      window.renderMathInElement(article, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ],
        throwOnError: false
      });
    }
  }

  function setPager() {
    var slug = currentSlug();
    var idx = flat.findIndex(function (it) { return it.slug === slug; });
    prevLink.innerHTML = ""; nextLink.innerHTML = "";
    prevLink.removeAttribute("href"); nextLink.removeAttribute("href");
    if (idx > 0) {
      prevLink.href = "#/" + flat[idx - 1].slug;
      prevLink.innerHTML = "← Trước<b>" + escapeHtml(flat[idx - 1].title) + "</b>";
    }
    if (idx !== -1 && idx < flat.length - 1) {
      nextLink.href = "#/" + flat[idx + 1].slug;
      nextLink.innerHTML = "Tiếp →<b>" + escapeHtml(flat[idx + 1].title) + "</b>";
    }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function renderHome() {
    var html = '<h1>Chuyên đề Lý thuyết AI</h1>';
    html += '<p class="lead">Tổng hợp các chuyên đề nền tảng về Trí tuệ nhân tạo và Học máy, được sắp xếp theo chuyên mục. Chọn một chuyên mục bên dưới để bắt đầu.</p>';
    toc.innerHTML = "";
    manifest.groups.forEach(function (g) {
      var items = g.items.filter(function (it) { return it.slug !== manifest.home; });
      if (!items.length) return;
      html += '<section class="home-section">';
      html += '<h2 id="' + slugify(g.title) + '">' + escapeHtml(g.title) + '</h2>';
      html += '<div class="cards">';
      items.forEach(function (it) {
        var idx = flat.findIndex(function (f) { return f.slug === it.slug; });
        var num = (idx >= 0) ? String(idx + 1).padStart(2, '0') + '. ' : '';
        html += '<a class="card" href="#/' + it.slug + '">';
        html += '<div class="card-kicker">' + escapeHtml(g.title) + "</div>";
        html += '<div class="card-title"><span class="card-num">' + escapeHtml(num) + '</span>' + escapeHtml(it.title) + "</div>";
        html += '<p class="card-desc">' + escapeHtml(it.desc || "") + "</p>";
        html += "</a>";
      });
      html += "</div></section>";
    });
    article.innerHTML = html;
    buildToc();
    observeHeadings();
    setPager();
  }

  function loadDoc(slug) {
    article.innerHTML = '<div class="loading">Đang tải...</div>';
    var item = flat.find(function (it) { return it.slug === slug; });
    var file = item ? item.file : ("content/" + slug + ".md");
    fetch(file, { cache: "no-cache" })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then(function (md) {
        article.innerHTML = marked.parse(md);
        buildToc();
        renderMath();
        observeHeadings();
        setPager();
        window.scrollTo(0, 0);
      })
      .catch(function () {
        article.innerHTML = '<h1>Không tìm thấy chuyên đề</h1><p class="lead">Nội dung bạn tìm chưa tồn tại. <a href="#/">Về trang chủ</a>.</p>';
        toc.innerHTML = "";
        setPager();
      });
  }

  function route() {
    closeSidebar();
    var slug = currentSlug();
    document.title = "Chuyên đề Lý thuyết AI";
    if (!slug) {
      renderHome();
    } else {
      var item = flat.find(function (it) { return it.slug === slug; });
      if (item) document.title = item.title + " · Chuyên đề Lý thuyết AI";
      loadDoc(slug);
    }
    highlightActive();
  }

  search.addEventListener("input", function () { renderNav(search.value); });

  window.addEventListener("hashchange", route);

  initTheme();
  fetch("content/manifest.json", { cache: "no-cache" })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      manifest = data;
      manifest.groups.forEach(function (g) {
        g.items.forEach(function (it) { it.group = g.title; });
      });
      buildFlat();
      renderNav("");
      route();
    })
    .catch(function () {
      article.innerHTML = '<h1>Lỗi tải dữ liệu</h1><p>Không đọc được <code>content/manifest.json</code>.</p>';
    });
})();
