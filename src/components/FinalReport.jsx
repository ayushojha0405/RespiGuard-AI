import React, { useEffect, useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";

const FinalReport = () => {
  const [report, setReport] = useState("");
  const [pdfUrl, setPdfUrl] = useState("");
  const [timestamp, setTimestamp] = useState("");
  const [patientId, setPatientId] = useState("");

  useEffect(() => {
    setReport(localStorage.getItem("finalReport") || "No report found.");
    setPdfUrl(localStorage.getItem("finalReportPdfUrl") || "");
    setTimestamp(localStorage.getItem("finalReportTimestamp") || "");
    setPatientId(localStorage.getItem("finalReportPatientId") || "Unknown");
  }, []);

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Final Report for Patient ID: {patientId}</h1>
          <p><strong>Generated At:</strong> {timestamp}</p>

          <pre
            style={{
              background: "#f4f4f4",
              padding: "1rem",
              borderRadius: "8px",
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
              marginBottom: "1.5rem"
            }}
          >
            {report}
          </pre>

          {pdfUrl && (
            <>
              <h2>Attached PDF Report</h2>
              <iframe
                src={pdfUrl}
                title="AI Medical Report"
                width="100%"
                height="600px"
                style={{
                  border: "1px solid #ccc",
                  borderRadius: "8px",
                  marginBottom: "1rem"
                }}
              />

              <a
                href={pdfUrl}
                download={`Report_${patientId}.pdf`}
                className="test-btn"
                style={{
                  display: "inline-block",
                  backgroundColor: "#28a745",
                  color: "white",
                  padding: "10px 16px",
                  borderRadius: "6px",
                  textDecoration: "none"
                }}
              >
                ⬇ Download PDF
              </a>
            </>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default FinalReport;
