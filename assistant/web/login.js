// Вход: POST /login с JSON. При успехе сервер ставит cookie сессии.
"use strict";

const form = document.getElementById("login");
const error = document.getElementById("error");
const enter = document.getElementById("enter");

function show(text) {
  error.textContent = text;
  error.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.hidden = true;
  enter.disabled = true;
  try {
    const response = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        login: document.getElementById("user").value,
        password: document.getElementById("password").value,
      }),
    });
    if (response.ok) {
      window.location.replace("/");
    } else if (response.status === 401) {
      show("Неверный логин или пароль.");
    } else if (response.status === 429 || response.status === 503) {
      // 429/503 отдаёт nginx, когда попыток входа слишком много.
      show("Слишком много попыток. Подождите минуту и попробуйте снова.");
    } else {
      show(`Не получилось войти: ошибка ${response.status}.`);
    }
  } catch {
    show("Сервер не отвечает.");
  } finally {
    enter.disabled = false;
  }
});
