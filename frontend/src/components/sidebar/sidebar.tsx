import { useEffect, useState } from 'react';
import { BarChart3, Bot, ChevronLeft, ChevronRight, LineChart } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import './sidebar.css';
import wsbLogo from '../../assets/WSB.png';
import { pageHome, pageTopPosts, pageTrading, pageWsbChatbot } from '../../router/router';

interface SidebarProps {
  onToggle: (state: boolean) => void;
}

const SIDEBAR_STORAGE_KEY = 'wsb-sidebar-open';

const navItems = [
  { to: pageTopPosts, label: 'Top Posts', icon: BarChart3 },
  { to: pageWsbChatbot, label: 'WSB Chatbot', icon: Bot },
  { to: pageTrading, label: 'Simulation', icon: LineChart },
];

export const Sidebar = ({ onToggle }: SidebarProps) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.localStorage.getItem(SIDEBAR_STORAGE_KEY) !== 'false');

  useEffect(() => {
    onToggle(isSidebarOpen);
    window.localStorage.setItem(SIDEBAR_STORAGE_KEY, String(isSidebarOpen));
  }, [isSidebarOpen, onToggle]);

  const toggleSidebar = () => {
    setIsSidebarOpen((current) => !current);
  };

  return (
    <aside className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`} aria-label="Primary navigation">
      <NavLink to={pageHome} className="sidebar-brand" aria-label="WSB Analyst home">
        <img src={wsbLogo} alt="WSB logo" />
        <span>
          <strong>WSB</strong>
          <small>Analyst</small>
        </span>
      </NavLink>

      <nav className="sidebar-content">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `sidebar-item ${isActive ? 'active' : ''}`}
            title={isSidebarOpen ? undefined : label}
          >
            <Icon className="sidebar-icon" size={20} strokeWidth={2.2} aria-hidden="true" />
            <span className="sidebar-label">{label}</span>
          </NavLink>
        ))}
      </nav>

      <button
        className="sidebar-toggle"
        type="button"
        onClick={toggleSidebar}
        aria-label={isSidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
      >
        {isSidebarOpen ? <ChevronLeft size={19} /> : <ChevronRight size={19} />}
      </button>
    </aside>
  );
};

export default Sidebar;
