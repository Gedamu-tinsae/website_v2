import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import '../styles/Navbar.css';

const Navbar = () => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate('/');
    window.location.reload();
  };

  const handleHomeClick = () => {
    navigate('/');
    window.location.reload();
  };

  return (
    <nav className="navbar">
      <div className="logo" onClick={handleLogoClick}>VITS</div>
      <div className="nav-links">
        <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')} onClick={handleHomeClick}>HOME</NavLink>
        <NavLink to="/about" className={({ isActive }) => (isActive ? 'active' : '')}>ABOUT</NavLink>
        <NavLink to="/docs" className={({ isActive }) => (isActive ? 'active' : '')}>DOCS</NavLink>
        <NavLink to="/db" className={({ isActive }) => (isActive ? 'active' : '')}>DB</NavLink>
      </div>
    </nav>
  );
};

export default Navbar;