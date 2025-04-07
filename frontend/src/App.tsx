import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import './App.css';
import RouterSwitch from './router/RouterSwitch';

function App() {
  return (
    <Router>
      <RouterSwitch />
    </Router>
  );
}

export default App;
