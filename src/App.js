import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import LoginPage from './components/LoginPage';
import Dashboard from './components/Dashboard';
import Sidebar from "./components/Sidebar";
import ReportScan from "./components/ReportScan";
import History from "./components/History";
import TestLab from "./components/TestLab";
import HelpDocs from "./components/HelpDocs";
import Check from "./components/Check";
import FinalReport from "./components/FinalReport";
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/sidebar" element={<Sidebar />} />
        <Route path="/reportscan" element={<ReportScan />} />
        <Route path="/history" element={<History />} />
        <Route path="/testlab" element={<TestLab />} />
        <Route path="/Check" element={<Check />} />
        <Route path="/helpdocs" element={<HelpDocs />} />
        <Route path="/final-report" element={<FinalReport />} />

      </Routes>
    </BrowserRouter>
  );
}

export default App;
