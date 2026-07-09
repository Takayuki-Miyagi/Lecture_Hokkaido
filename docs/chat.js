const API_URL = "https://lecture-chatbot-backend.vercel.app/api/chat";
const STORAGE_KEY = "lecture-chatbot-history";

function normalizeMathDelimiters(text) {
  return text
    .replace(/^\s*\[\s*$/gm, "\\[")
    .replace(/^\s*\]\s*$/gm, "\\]");
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function renderHistory() {
  const chatLog = document.getElementById("chat-log");
  const history = loadHistory();

  chatLog.innerHTML = "";

  for (const item of history) {
    const block = document.createElement("div");
    block.className = "chat-message";

    const q = document.createElement("div");
    q.className = "chat-question";
    q.textContent = item.question;

    const a = document.createElement("div");
    a.className = "chat-answer";
    a.textContent = normalizeMathDelimiters(item.answer);

    block.appendChild(q);
    block.appendChild(a);
    chatLog.appendChild(block);
  }

  if (window.MathJax) {
    window.MathJax.typesetPromise([chatLog]);
  }
}

async function askQuestion() {
  const questionBox = document.getElementById("question");
  const question = questionBox.value.trim();

  if (!question) {
    return;
  }

  const history = loadHistory();

  history.push({
    question,
    answer: "Thinking...",
    time: new Date().toISOString(),
  });
  saveHistory(history);
  renderHistory();

  questionBox.value = "";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: question }),
    });

    const data = await res.json();

    history[history.length - 1].answer =
      res.ok ? data.answer : data.error ?? "Error";

    saveHistory(history);
    renderHistory();
  } catch {
    history[history.length - 1].answer =
      "Failed to connect to the chatbot backend.";

    saveHistory(history);
    renderHistory();
  }
}

document.getElementById("ask").addEventListener("click", askQuestion);

document.getElementById("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    askQuestion();
  }
});

document.getElementById("clear").addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  renderHistory();
});

renderHistory();
