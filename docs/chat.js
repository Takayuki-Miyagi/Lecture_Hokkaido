const API_URL = "https://lecture-chatbot-backend.vercel.app/api/chat";
const STORAGE_KEY = "lecture-chatbot-history";

let messages = loadMessages();

function normalizeMathDelimiters(text) {
  return text
    .replace(/^\s*\[\s*$/gm, "\\[")
    .replace(/^\s*\]\s*$/gm, "\\]");
}

function loadMessages() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

function saveMessages() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

function renderMessages() {
  const chatLog = document.getElementById("chat-log");
  chatLog.innerHTML = "";

  for (const message of messages) {
    const bubble = document.createElement("div");
    bubble.className = `message ${message.role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = message.role === "user" ? "You" : "AI";

    const content = document.createElement("div");
    content.className = "content";
    content.textContent =
      message.role === "assistant"
        ? normalizeMathDelimiters(message.content)
        : message.content;

    bubble.appendChild(avatar);
    bubble.appendChild(content);
    chatLog.appendChild(bubble);
  }

  chatLog.scrollTop = chatLog.scrollHeight;

  if (window.MathJax) {
    window.MathJax.typesetPromise([chatLog]);
  }
}

async function askQuestion() {
  const input = document.getElementById("question");
  const question = input.value.trim();

  if (!question) return;

  messages.push({
    role: "user",
    content: question,
  });

  messages.push({
    role: "assistant",
    content: "Thinking...",
  });

  input.value = "";
  saveMessages();
  renderMessages();

  try {
    const messagesForApi = messages
      .filter((m) => m.content !== "Thinking...")
      .slice(-12);

    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages: messagesForApi }),
    });

    const data = await res.json();

    messages[messages.length - 1] = {
      role: "assistant",
      content: res.ok ? data.answer : data.error ?? "Error",
    };

    saveMessages();
    renderMessages();
  } catch {
    messages[messages.length - 1] = {
      role: "assistant",
      content: "Failed to connect to the chatbot backend.",
    };

    saveMessages();
    renderMessages();
  }
}

document.getElementById("ask").addEventListener("click", askQuestion);

document.getElementById("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});

document.getElementById("clear").addEventListener("click", () => {
  messages = [];
  saveMessages();
  renderMessages();
});

renderMessages();
