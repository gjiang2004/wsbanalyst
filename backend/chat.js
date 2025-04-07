import React, { useState, useEffect } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:5000";

const Chat = () => {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);

  // Fetch conversation history on component mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/history");
        const historyLines = response.data.history.split("\n").filter(line => line.trim() !== "");
        
        const formattedMessages = historyLines.map((line, index) => ({
          sender: index % 2 === 0 ? "user" : "bot",
          text: line,
        }));

        setMessages(formattedMessages);
      } catch (error) {
        console.error("Error fetching history:", error);
      }
    };

    fetchHistory();
  }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;

    // Add user's message to chat
    const newMessages = [...messages, { sender: "user", text: message }];
    setMessages(newMessages);
    setMessage("");

    try {
      const response = await axios.post("http://127.0.0.1:5000/chat", { message });

      // Add bot response to chat
      setMessages([...newMessages, { sender: "bot", text: response.data.response }]);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={msg.sender}>
            <strong>{msg.sender === "user" ? "You: " : "Bot: "}</strong> {msg.text}
          </div>
        ))}
      </div>

      <div className="input-container">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

export default Chat;
