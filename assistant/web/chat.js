// Простой чат поверх POST /ask. Без библиотек и без innerHTML: весь текст
// вставляется через textContent, поэтому HTML из ответа модели не выполнится.
"use strict";

const REASONS = {
  model_not_found: "Не нашёл ответа в документации и коде",
  filtered_out: "Не нашёл ответа, который подтверждают документация и код",
  no_sources: "Не нашёл ничего по вопросу в документации и коде",
  blacklist: "GigaChat не стал отвечать на этот вопрос (тематические ограничения)",
  length: "Не получилось собрать ответ: он не уместился в лимит модели",
  format_error: "Не получилось собрать ответ: модель ответила не по формату",
};

const log = document.getElementById("log");
const form = document.getElementById("ask");
const input = document.getElementById("question");
const send = document.getElementById("send");
let answers = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// `PAPERLESS_X` в утверждении — имя параметра: показываем его как код.
function withCode(text) {
  const fragment = document.createDocumentFragment();
  text.split("`").forEach((part, index) => {
    fragment.append(index % 2 ? element("code", "", part) : document.createTextNode(part));
  });
  return fragment;
}

function safeLink(url, text) {
  // Только https: ссылки строит сервер, но лишняя проверка ничего не стоит.
  if (!/^https:\/\//.test(url)) return element("span", "", text);
  const link = element("a", "", text);
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function lines(source) {
  return source.start_line === source.end_line
    ? `${source.start_line}`
    : `${source.start_line}-${source.end_line}`;
}

function renderAnswer(answer) {
  const card = element("article", "message");
  const prefix = `a${++answers}-`;
  if (answer.status === "answer") {
    const list = element("ol", "statements");
    for (const statement of answer.statements) {
      const item = element("li");
      item.append(withCode(statement.text));
      for (const number of statement.sources) {
        const ref = element("a", "ref", `[${number}]`);
        ref.href = `#${prefix}${number}`;
        item.append(ref);
      }
      list.append(item);
    }
    card.append(list);
  } else {
    const reason = REASONS[answer.reason] || "Не нашёл ответа";
    card.append(element("p", "notice", `${reason} Paperless-ngx ${answer.paperless_ref}.`));
    if (answer.sources.length) card.append(element("p", "notice", "Вот что может пригодиться:"));
  }
  const sources = element("ul", "sources");
  for (const source of answer.sources) {
    const item = element("li");
    item.id = `${prefix}${source.number}`;
    item.append(`[${source.number}] `);
    item.append(safeLink(source.url, `${source.path}:${lines(source)}`));
    item.append(element("span", "title", ` — ${source.title}`));
    if (source.site_url) {
      item.append(" · ");
      item.append(safeLink(source.site_url, "сайт"));
    }
    sources.append(item);
  }
  card.append(sources);
  if (answer.model) {
    card.append(element("p", "meta",
      `Ответ сгенерирован GigaChat (${answer.model}) по документации и коду Paperless-ngx ${answer.paperless_ref}.`));
  }
  return card;
}

async function ask(question) {
  log.append(element("div", "message question", question));
  const pending = element("div", "message pending", "Ищу в документации и коде, проверяю ссылки…");
  log.append(pending);
  pending.scrollIntoView({ behavior: "smooth" });
  send.disabled = true;
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (response.status === 401) {
      // Сессия истекла или пароль сменили: на страницу входа.
      window.location.href = "/login";
      return;
    }
    const data = await response.json().catch(() => ({}));
    if (response.ok) {
      pending.replaceWith(renderAnswer(data));
    } else if (response.status === 422) {
      pending.replaceWith(element("div", "message error", "Вопрос пустой или длиннее 2000 символов."));
    } else {
      pending.replaceWith(element("div", "message error",
        `Сейчас не могу ответить: ${data.detail || `ошибка ${response.status}`}.`));
    }
  } catch {
    pending.replaceWith(element("div", "message error", "Ассистент не отвечает: проверьте, что он запущен."));
  } finally {
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;
  input.value = "";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

fetch("/health")
  .then((response) => response.json())
  .then((health) => {
    document.getElementById("about").textContent =
      `Ответы по документации и коду Paperless-ngx ${health.paperless_ref}. Каждая ссылка проверена кодом.`;
  })
  .catch(() => {});

// Кнопка «Выйти» видна, только когда вход включён (при NO_AUTH=1 user — null).
const logout = document.getElementById("logout");
fetch("/session")
  .then((response) => response.json())
  .then((session) => {
    if (!session.user) return;
    logout.textContent = `Выйти (${session.user})`;
    logout.hidden = false;
  })
  .catch(() => {});
logout.addEventListener("click", async () => {
  await fetch("/logout", { method: "POST" }).catch(() => {});
  window.location.href = "/login";
});
