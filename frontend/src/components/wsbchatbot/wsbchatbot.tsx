import React, { useState, useEffect, useRef } from 'react';
import './wsbchatbot.css';
import char from '../../assets/aigirl.jpg';
import char2 from '../../assets/wojak.jpg';
import Sidebar from '../sidebar/sidebar';

type Sender = 'user' | 'bot';

interface Message {
  sender: Sender;
  text: string;
}

export const WsbChatbot: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [message, setMessage]             = useState<string>('');
  const [messages, setMessages]           = useState<Message[]>([]);
  const [isStreaming, setIsStreaming]      = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatWindowRef  = useRef<HTMLDivElement | null>(null);

  const handleSidebarToggle = (isOpen: boolean | ((prevState: boolean) => boolean)) => {
    setIsSidebarOpen(isOpen);
  };

  // Load conversation history once on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/history')
      .then(r => r.json())
      .then(data => {
        setMessages(
          data.history.map((m: { role: string; text: string }) => ({
            sender: m.role === 'user' ? 'user' : 'bot',
            text:   m.text,
          }))
        );
      })
      .catch(e => console.error('Error fetching history:', e));
  }, []);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!message.trim() || isStreaming) return;
    const userText = message;
    setMessage('');
    setIsStreaming(true);

    // add user message and empty bot message to stream into
    setMessages(prev => [
      ...prev,
      { sender: 'user', text: userText },
      { sender: 'bot',  text: '' },
    ]);

    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: userText }),
      });

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') break;

          const token = JSON.parse(raw);
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              text: updated[updated.length - 1].text + token,
            };
            return updated;
          });
        }
      }
    } catch (e) {
      console.error('Error sending message:', e);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div className="page-container">
      <Sidebar onToggle={handleSidebarToggle} />
      <div className={`chat-container ${isSidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className="chat-box">
          <div className="chat-window" ref={chatWindowRef}>
            <div className="aiman">
              <div className="image-container" />
              <p>A personalized WSB AI companion to chat with!</p>
            </div>

            <div className="messages">
              {messages.map((msg, index) => {
                const isUser    = msg.sender === 'user';
                const avatarSrc = isUser ? char2 : char;
                const username  = isUser ? 'You' : 'WSB AI';

                return (
                  <div key={index} className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
                    <div className={`avatar-container ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
                      {isUser ? (
                        <>
                          <span className="username">{username}</span>
                          <img src={avatarSrc} alt="Avatar" className="avatar" />
                        </>
                      ) : (
                        <>
                          <img src={avatarSrc} alt="Avatar" className="avatar" />
                          <span className="username">{username}</span>
                        </>
                      )}
                    </div>

                    <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
                      <div className="message-content">
                        <p>
                          {msg.text}
                          {!isUser && isStreaming && index === messages.length - 1 && (
                            <span className="cursor">▍</span>
                          )}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="chat-input-container">
            <input
              type="text"
              className="chat-input"
              placeholder="Type your message..."
              value={message}
              onChange={e => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default WsbChatbot;