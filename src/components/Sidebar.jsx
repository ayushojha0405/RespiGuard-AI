import React from "react";
import { NavLink } from "react-router-dom";
import "./../styles/Sidebar.css";

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="logo">
        RespiGuard AI
      </div>
      <nav className="nav-links">
        <NavLink exact="true" to="/dashboard" activeclassname="active">Dashboard</NavLink>
        <NavLink to="/reportscan" activeclassname="active">Report Scan</NavLink>
        <NavLink to="/history" activeclassname="active">History</NavLink>
        <NavLink to="/analytics" activeclassname="active">Analytics</NavLink>
        <NavLink to="/testlab" activeclassname="active">Test Lab</NavLink>
        <NavLink to="/helpdocs" activeclassname="active">Help & Docs</NavLink>
        <NavLink to="/settings" activeclassname="active">Settings</NavLink>
        <NavLink to="/feedback" activeclassname="active">Feedback</NavLink>
        <NavLink to="/myprofile" activeclassname="active">My Profile</NavLink>
        <NavLink to="/" activeclassname="active">Logout</NavLink>
      </nav>
    </div>
  );
};

export default Sidebar;
