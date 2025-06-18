## 🚀 Getting Started

### 1️⃣ Backend (Flask + PyTorch)

```bash
cd flask-backend
python -m venv venv
venv/Scripts/activate       # Windows
# or
source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
python app.py
```

📍 Flask runs at: `http://localhost:5000`

---

### 2️⃣ Frontend (React)

```bash
npm install
npm start
```

🌐 React runs at: `http://localhost:3000`

---

## 📦 Python Requirements

Inside `flask-backend/requirements.txt`:

```text
flask
flask-cors
torch
numpy
opencv-python
matplotlib
scipy
```

---

## ✨ Features

* 📋 Patient form with ID, name, age, gender, and scan date
* 📤 Upload `.jpg` scan images
* 🧠 AI model detects 14+ chest diseases (YOLO-like CNN)
* 🖼 Annotated heatmap and bounding boxes for findings
* 📝 Generates detailed diagnostic reports
* 💾 Stores report in browser `localStorage`
* 🔍 View report on `/final-report`
* 🗑 Delete entries from `/check`
* 📁 Saves image/report to `temp_images/`

---

## 📄 Output Examples

* `123456_report.png` → Annotated heatmap
* `123456_report.txt` → AI-generated text report
* Accessible from `/final-report`

---

## 📌 Notes

* Only `.jpg` images are supported.
* React and Flask must both be running locally.
* For best results, use a GPU-enabled PyTorch setup.

---


