import React, { useState, useEffect } from 'react';
import './sidebar.css';
import wsbLogo from '../../assets/WSB.png';

interface SidebarProps {
  onToggle: (state: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ onToggle }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeItem, setActiveItem] = useState<string | null>(null);

  const toggleSidebar = () => {
    const newState = !isSidebarOpen;
    setIsSidebarOpen(newState);
    // Pass the new state back to parent
    if (onToggle) {
      onToggle(newState);
    }
  };

  // Initialize parent with sidebar state
  useEffect(() => {
    if (onToggle) {
      onToggle(isSidebarOpen);
    }
  }, [isSidebarOpen, onToggle]);

  const handleItemClick = (item: string) => {
    setActiveItem(item);
  };

  return (
    <>
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <a href ="/"> <div className="sidebar-header">
          <img src={wsbLogo} alt="wsb logo" />
        </div> </a>
        <div className="sidebar-content">
          <ul className="sidebar-menu">
            {/*<a href="/top-posts">
              <li className={`sidebar-item ${activeItem === 'trending' ? 'active' : ''}`} onClick={() => handleItemClick('trending')}>
                <span className="sidebar-icon">🚀</span>
                {isSidebarOpen && <span className="sidebar-label">Trending</span>}
              </li>
            </a>*/}
            <a href="/wsb-chatbot">
              <li className={`sidebar-item ${activeItem === 'chatbot' ? 'active' : ''}`} onClick={() => handleItemClick('chatbot')}>
                <span className="sidebar-icon">🤖</span>
                {isSidebarOpen && <span className="sidebar-label">WSB Chatbot</span>}
              </li>
            </a>
            <a href="/trading">
              <li className={`sidebar-item ${activeItem === 'simulation' ? 'active' : ''}`} onClick={() => handleItemClick('simulation')}>
                <span className="sidebar-icon">📈</span>
                {isSidebarOpen && <span className="sidebar-label">Simulation</span>}
              </li>
            </a>
          </ul>
        </div>

        <div className="sidebar-toggle-container">
          <button className="sidebar-toggle" onClick={toggleSidebar}>
            {isSidebarOpen ? '◀' : '▶'}
          </button>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
