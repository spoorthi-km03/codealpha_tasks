const messages = document.getElementById('messages');
const input = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const WELCOME = "Hello! 👋 I'm the College FAQ Assistant. Ask me about admissions, courses, fees, scholarships, hostel, library, exams, attendance, placements, timings or contact details.";

function addMessage(text, who, note) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  if (note) { const s = document.createElement('small'); s.textContent = note; div.appendChild(s); }
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}
function showTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot typing';
  div.id = 'typing';
  div.innerHTML = '<span></span><span></span><span></span>';
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}
function hideTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }

async function sendMessage(text) {
  text = (text !== undefined ? text : input.value).trim();
  if (!text) { input.focus(); return; }
  addMessage(text, 'user');
  input.value = '';
  sendBtn.disabled = true;
  showTyping();
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    await new Promise(r => setTimeout(r, 400));
    hideTyping();
    const note = data.matched ? 'Topic: ' + data.category + ' · Match: ' + Math.round(data.score * 100) + '%' : '';
    addMessage(data.answer || 'Something went wrong.', 'bot', note);
  } catch (e) {
    hideTyping();
    addMessage('Sorry, I could not reach the server. Please check that the app is running and try again.', 'bot');
  }
  sendBtn.disabled = false;
  input.focus();
}
function clearChat() { messages.innerHTML = ''; addMessage(WELCOME, 'bot'); input.focus(); }

sendBtn.addEventListener('click', () => sendMessage());
input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); sendMessage(); } });
document.getElementById('clearBtn').addEventListener('click', clearChat);
document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => sendMessage(c.textContent)));
clearChat();
