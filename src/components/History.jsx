import React, { useEffect, useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/History.css";

const History = () => {
  const [reports, setReports] = useState([]);

  useEffect(() => {
    const storedReports = JSON.parse(localStorage.getItem("reports")) || [];
    setReports(storedReports);
  }, []);

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Report History</h1>

          <table className="history-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Gender</th>
                <th>Date</th>
                <th>Doctor</th>
                <th>File</th>
              </tr>
            </thead>
            <tbody>
              {reports.length === 0 ? (
                <tr>
                  <td colSpan="7" style={{ textAlign: "center" }}>No reports found</td>
                </tr>
              ) : (
                reports.map((report, index) => (
                  <tr key={index}>
                    <td>{report.patientId}</td>
                    <td>{report.patientName}</td>
                    <td>{report.age}</td>
                    <td>{report.gender}</td>
                    <td>{report.examDate}</td>
                    <td>{report.doctorName}</td>
                    <td>{report.scanFile}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default History;
