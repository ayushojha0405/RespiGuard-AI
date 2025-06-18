import React from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/Dashboard.css";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

const Dashboard = () => {
  const doctorUsername = localStorage.getItem("doctorUsername");

  const lineData = [
    { month: "Jan", patients: 10 },
    { month: "Feb", patients: 30 },
    { month: "Mar", patients: 20 },
    { month: "Apr", patients: 50 },
    { month: "May", patients: 40 },
  ];

  const pieData = [
    { name: "Healthy", value: 60 },
    { name: "At Risk", value: 25 },
    { name: "Infected", value: 15 },
  ];

  const COLORS = ["#1ca6c9", "#36bfe3", "#105c70"];

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Hello, Dr. {doctorUsername}</h1>

          <div className="summary-cards">
            <div className="card">
              <h3>Total Patients</h3>
              <p>450</p>
            </div>
            <div className="card">
              <h3>Active Scans</h3>
              <p>120</p>
            </div>
            <div className="card">
              <h3>Critical Cases</h3>
              <p>15</p>
            </div>
            <div className="card">
              <h3>Recovered</h3>
              <p>310</p>
            </div>
          </div>

          <div className="charts-section">
            <div className="chart-card">
              <h3>Monthly Patient Scans</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={lineData}>
                  <Line type="monotone" dataKey="patients" stroke="#16819c" strokeWidth={3} />
                  <CartesianGrid stroke="#ccc" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3>Respiratory Health Status</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Dashboard;
