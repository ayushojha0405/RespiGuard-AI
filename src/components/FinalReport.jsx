import React, { useEffect, useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";

const FinalReport = () => {
  const [report, setReport] = useState("");

  useEffect(() => {
    const savedReport = localStorage.getItem("finalReport");
    if (savedReport) {
      setReport(savedReport);
    } else {
      setReport("No report found. Please run a test first.");
    }
  }, []);

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>AI-Generated Final Report</h1>
          <pre style={{
            background: "#f4f4f4",
            padding: "1rem",
            borderRadius: "8px",
            whiteSpace: "pre-wrap",
            wordWrap: "break-word"
          }}>
            {report}
          </pre>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default FinalReport;
