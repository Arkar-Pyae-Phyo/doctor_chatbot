const chatForm = document.getElementById("chatForm");
const chatLog = document.getElementById("chatLog");
const questionInput = document.getElementById("questionInput");

const appendMessage = (content, role) => {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = content;
  chatLog.appendChild(message);
  chatLog.scrollTop = chatLog.scrollHeight;
};

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage(question, "user");
  questionInput.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      let message = "Unable to get a response from the API.";
      try {
        const error = await response.json();
        if (error?.detail) {
          message = error.detail;
        }
      } catch (parseError) {
        message = "Unable to get a response from the API.";
      }
      appendMessage(message, "bot");
      return;
    }
    const data = await response.json();
    appendMessage(data.answer || "I do not know.", "bot");
  } catch (error) {
    appendMessage("Unable to reach the API.", "bot");
  }
});
