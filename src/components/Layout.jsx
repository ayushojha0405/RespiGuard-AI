import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import Footer from './Footer';
import './Layout.css';

const Layout = ({ children }) => {
  return (
    <div className="layout">
      <Header className="header" />
      <div className="main">
        <Sidebar className="sidebar" />
        <div className="content">{children}</div>
      </div>
      <Footer className="footer" />
    </div>
  );
};

export default Layout;
