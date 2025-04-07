import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
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
  const [message, setMessage] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);

  // Ref to scroll to bottom automatically
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatWindowRef = useRef<HTMLDivElement | null>(null);

  const handleSidebarToggle = (isOpen: boolean | ((prevState: boolean) => boolean)) => {
    setIsSidebarOpen(isOpen);
  };

  // Load conversation history once
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/history');
        const historyLines = response.data.history
          .split('\n')
          .filter((line: string) => line.trim() !== '');
        const formatted: Message[] = historyLines.map((line: string, index: number) => ({
          sender: (index % 2 === 0 ? 'user' : 'bot') as Sender,
          text: line,
        }));
        setMessages(formatted);
      } catch (error) {
        console.error('Error fetching history:', error);
      }
    };
    fetchHistory();
  }, []);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!message.trim()) return;

    // Add user message to local state
    const newMessages = [...messages, { sender: 'user' as Sender, text: message }];
    setMessages(newMessages);
    setMessage('');

    try {
      // Send to backend
      const response = await axios.post('http://127.0.0.1:5000/chat', { message });
      const botResponse = response.data.response;

      // Add bot reply to local state
      setMessages([...newMessages, { sender: 'bot' as Sender, text: botResponse }]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
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
                const isUser = msg.sender === 'user';
                const avatarSrc = isUser ? char2 : char;
                const username = isUser ? 'You' : 'WSB AI';

                return (
                  <div key={index} className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
                    {/* Avatar + username */}
                    <div className={`avatar-container ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
                      {isUser ? (
                        <>
                          <span className="username">{username}</span>
                          <img
                            src={avatarSrc}
                            alt="Avatar"
                            className="avatar"
                          />
                        </>
                      ) : (
                        <>
                          <img
                            src={avatarSrc}
                            alt="Avatar"
                            className="avatar"
                          />
                          <span className="username">{username}</span>
                        </>
                      )}
                    </div>

                    {/* Message bubble */}
                    <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
                      <div className="message-content">
                        <p>{msg.text}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
              {/* Dummy div to anchor scroll at bottom */}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input + Send button */}
          <div className="chat-input-container">
            <input
              type="text"
              className="chat-input"
              placeholder="Type your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default WsbChatbot;