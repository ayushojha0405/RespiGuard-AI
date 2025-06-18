import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/Check.css";

const Check = () => {
  const [reports, setReports] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const data = JSON.parse(localStorage.getItem("details.json")) || [];
    setReports(data);
  }, []);

  const handleDelete = (indexToDelete) => {
    const updatedReports = reports.filter((_, index) => index !== indexToDelete);
    setReports(updatedReports);
    localStorage.setItem("details.json", JSON.stringify(updatedReports));
  };

  const handleTest = async (report) => {
    try {
      const response = await fetch("http://localhost:5000/run-test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          imageData: report.scanImageData,
          imageName: report.scanFile || "scan.jpg",
          patientId: report.patientId || "Unknown",
        }),
      });

      const result = await response.json();

      if (result.error) {
        alert("Error: " + result.error);
      } else {
        localStorage.setItem("finalReport", result.output);
        navigate("/final-report");
      }
    } catch (error) {
      alert("Failed to connect to backend: " + error.message);
    }
  };

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Submitted Reports</h1>
          {reports.length === 0 ? (
            <p>No reports submitted yet.</p>
          ) : (
            reports.map((report, index) => (
              <div key={index} className="report-card">
                <p><strong>Patient ID:</strong> {report.patientId}</p>
                <p><strong>Name:</strong> {report.patientName}</p>
                <p><strong>Age:</strong> {report.age}</p>
                <p><strong>Gender:</strong> {report.gender}</p>
                <p><strong>Exam Date:</strong> {report.examDate}</p>
                <p><strong>Doctor:</strong> {report.doctorName}</p>
                <p><strong>File:</strong> {report.scanFile}</p>

                {report.scanImageData && (
                  <img
                    src={report.scanImageData}
                    alt="Scan Preview"
                    style={{ width: "200px", marginTop: "10px" }}
                  />
                )}

                <div style={{ marginTop: "10px" }}>
                  <button className="delete-btn" onClick={() => handleDelete(index)}>
                    Delete Report
                  </button>
                  <button className="test-btn" onClick={() => handleTest(report)}>
                    Start Testing
                  </button>
                </div>
                <hr />
              </div>
            ))
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Check;
