# 🛡️ Network Intrusion Detection System (NIDS)

A sophisticated machine learning-powered web application for detecting network intrusions and security threats in real-time.

[![Live Demo](https://img.shields.io/badge/Live_Demo-🚀-success)](https://web-production-7c370.up.railway.app)
![NIDS Dashboard](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange)

## 🌟 Features

- **🛡️ Advanced ML Detection** - XGBoost-powered threat classification with 99.76% accuracy
- **📊 Interactive Dashboard** - Real-time analytics with charts and visualizations
- **📈 Performance Analytics** - Comprehensive model evaluation metrics
- **🔒 Secure Authentication** - User management with password protection
- **📜 Prediction History** - Complete audit trail with CSV export functionality
- **🎯 Threat Analysis** - Detailed insights into network security patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nids-system.git
   cd nids-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app/app.py
   ```

4. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.76% |
| **Precision** | 99.68% |
| **Recall** | 99.84% |
| **F1-Score** | 99.76% |
| **ROC-AUC** | 99.99% |

## 🏗️ Tech Stack

### Backend
- **Flask** - Lightweight WSGI web application framework
- **SQLite** - Embedded database for user management
- **Werkzeug** - WSGI utility library

### Machine Learning
- **XGBoost** - Gradient boosting framework
- **Scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data manipulation and analysis

### Frontend
- **HTML5/CSS3** - Modern web standards
- **JavaScript** - Interactive functionality
- **Bootstrap** - Responsive design framework
- **Chart.js** - Data visualization

### Deployment
- **Railway** - Modern app deployment platform (Currently Live)
- **Gunicorn** - Production WSGI server
- **Git** - Version control and deployment

## 📁 Project Structure

```
nids-system/
├── wsgi.py              # WSGI entry point for production
├── requirements.txt     # Python dependencies
├── Procfile            # Railway deployment configuration
├── runtime.txt         # Python version specification
├── app/                # Flask application
│   ├── templates/      # Jinja2 HTML templates
│   │   ├── base.html   # Main layout with responsive design
│   │   ├── dashboard.html
│   │   ├── login.html
│   │   ├── about.html
│   │   └── ...
│   ├── static/         # CSS, JS, images
│   │   ├── css/
│   │   └── images/
│   └── app.py          # Main Flask application
├── artifact/           # ML model artifacts
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── metadata.json
├── utils/              # Database utilities
├── data/               # Raw data files
└── README.md          # Project documentation
```

## 🎯 Key Components

### 🔍 Network Traffic Analysis
- Real-time packet inspection
- Feature extraction from network flows
- Anomaly detection algorithms

### 🤖 Machine Learning Pipeline
- Data preprocessing and feature engineering
- Model training and validation
- Performance evaluation and metrics

### 🌐 Web Interface
- User-friendly dashboard
- Interactive prediction forms
- Historical data visualization

## 🚀 Deployment

### Railway (Currently Live)
The application is currently deployed and live on Railway:

🌐 **Live Demo**: https://web-production-7c370.up.railway.app

**Railway Deployment Setup**:
1. **Connect Repository**: Link your GitHub repo to Railway
2. **Automatic Detection**: Railway automatically detects Python apps
3. **Build Process**: Installs dependencies from `requirements.txt`
4. **Start Command**: Uses `Procfile` with Gunicorn WSGI server
5. **Deploy**: Auto-deploy on git push

### Local Development
```bash
pip install -r requirements.txt
python app/app.py
# Visit: http://localhost:5000
```

### Alternative Platforms
- **Render**: Similar setup with Python runtime
- **Fly.io**: Container-native deployment
- **Heroku**: Traditional PaaS deployment

## 📈 Usage

1. **Register/Login** to access the system
2. **Navigate to Predict** page to analyze network traffic
3. **Input network features** or use default safe values
4. **View results** with confidence scores
5. **Monitor dashboard** for analytics and insights
6. **Export history** as CSV for further analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is developed as a **Final Year Project** for academic purposes at the Federal University of Petroleum Resources, Effurun (FUPRE).

## 🙏 Acknowledgments

- **Supervisor**: For guidance and mentorship
- **FUPRE**: For providing the academic environment
- **Open Source Community**: For the amazing tools and libraries

## 📞 Contact

**Project Developer**: [Your Name]
**Institution**: Federal University of Petroleum Resources, Effurun
**Course**: Computer Science (Final Year Project)

---

⭐ **Star this repository** if you find it helpful!
