import React, { useState } from 'react';
import './trading.css';
import Sidebar from '../sidebar/sidebar';
import Graph from '../graphs/Graph';

export const Trading = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const handleSidebarToggle = (isOpen: boolean | ((prevState: boolean) => boolean)) => {
    setIsSidebarOpen(isOpen);
  };
  
  return (
    <>
      <Sidebar onToggle={handleSidebarToggle} />
      <div className={`trading-container ${isSidebarOpen ? '' : 'sidebar-closed'}`}>
        <div className={`trading-content ${isSidebarOpen ? '' : 'sidebar-closed'}`}>
          <Graph />
        </div>
      </div>
    </>
  );
};

export default Trading;