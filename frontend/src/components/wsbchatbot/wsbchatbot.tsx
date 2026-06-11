import React, { useState, useEffect, useRef } from 'react';
import './wsbchatbot.css';
import char from '../../assets/aigirl.jpg';
import char2 from '../../assets/wojak.jpg';
import { apiUrl } from '../../lib/api';
import Sidebar from '../sidebar/sidebar';

type Sender = 'user' | 'bot';

interface Message {
  sender: Sender;
  text: string;
}

const CHAT_ERROR_MESSAGE =
  'Chat is not responding right now. Check that the backend is running and the chat provider is configured.';

export const WsbChatbot: React.FC = () => {
  const [message, setMessage]             = useState<string>('');
  const [messages, setMessages]           = useState<Message[]>([]);
  const [isStreaming, setIsStreaming]      = useState(false);
  const [isSidebarOpen, setIsSidebarOpen]  = useState(true);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatWindowRef  = useRef<HTMLDivElement | null>(null);

  // Load conversation history once on mount
  useEffect(() => {
    fetch(apiUrl('/history'))
      .then(r => {
        if (!r.ok) throw new Error('History request failed: ' + r.status);
        return r.json();
      })
      .then(data => {
        const history = Array.isArray(data.history) ? data.history : [];
        setMessages(
          history.map((m: { role: string; text: string }) => ({
            sender: m.role === 'user' ? 'user' : 'bot',
            text:   m.text,
          }))
        );
      })
      .catch(e => {
        console.error('Error fetching history:', e);
        setMessages([{ sender: 'bot', text: CHAT_ERROR_MESSAGE }]);
      });
  }, []);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const replaceLastBotMessage = (text: string) => {
    setMessages(prev => {
      const updated = [...prev];
      const lastIndex = updated.length - 1;
      if (lastIndex >= 0 && updated[lastIndex].sender === 'bot') {
        updated[lastIndex] = { ...updated[lastIndex], text };
      }
      return updated;
    });
  };

  const appendToLastBotMessage = (token: string) => {
    setMessages(prev => {
      const updated = [...prev];
      const lastIndex = updated.length - 1;
      if (lastIndex >= 0 && updated[lastIndex].sender === 'bot') {
        updated[lastIndex] = {
          ...updated[lastIndex],
          text: updated[lastIndex].text + token,
        };
      }
      return updated;
    });
  };

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
      const res = await fetch(apiUrl('/chat'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: userText }),
      });

      if (!res.ok) {
        let detail = '';
        try {
          detail = await res.text();
        } catch {
          detail = '';
        }
        throw new Error('Chat request failed: ' + res.status + (detail ? ' ' + detail : ''));
      }
      if (!res.body) throw new Error('Chat response did not include a stream body.');

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') {
            finished = true;
            break;
          }

          appendToLastBotMessage(JSON.parse(raw));
        }

        if (done) break;
      }
    } catch (e) {
      console.error('Error sending message:', e);
      replaceLastBotMessage(CHAT_ERROR_MESSAGE);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div className="page-container">
      <Sidebar onToggle={setIsSidebarOpen} />
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