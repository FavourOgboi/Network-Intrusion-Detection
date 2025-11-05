# 🚀 Render Deployment Guide for NIDS

## Why Render?

Render is the **perfect platform** for your Network Intrusion Detection System because:

- ✅ **Native Python/Flask support** - No complex configurations needed
- ✅ **SQLite database support** - Your database works as-is
- ✅ **Free tier** - 750 hours/month for development
- ✅ **Git integration** - Auto-deploy on push
- ✅ **Managed infrastructure** - No server maintenance
- ✅ **Custom domains** - Professional URLs

## 🛠️ Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code pushed to GitHub

## 🚀 Quick Deployment

### Step 1: Connect Repository
1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Select the main branch

### Step 2: Configure Service
```
Name: nids-app (or your choice)
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: python app/app.py
```

### Step 3: Environment Variables
Add in Render dashboard:
```
FLASK_ENV = production
```

### Step 4: Deploy
- Render will auto-deploy when you push to GitHub
- Your site will be live at: `https://your-app-name.onrender.com`

## 📁 Configuration Files

- **`render.yaml`** - Render service configuration
- **`requirements.txt`** - Python dependencies
- **`Procfile`** - Alternative process definition

## ⚠️ **Build Error Solutions:**

### Error: "scikit-learn build failed"
**Solution:** Create `runtime.txt` to force Python 3.11:
```txt
# runtime.txt
python-3.11.4
```

**And update requirements.txt:**
```txt
Flask==2.3.3
Werkzeug==2.3.7
scikit-learn==1.2.2  # Use 1.2.2 instead of 1.3.0
pandas==1.5.3        # Use 1.5.3 instead of 2.0.3
numpy==1.24.3
xgboost==1.7.6
joblib==1.3.2
python-dotenv==1.0.0
```

### Error: "Build timeout"
**Solution:** Reduce build time by using pre-compiled packages

### Error: "Memory limit exceeded"
**Solution:** Use lighter dependencies or upgrade Netlify plan

## ⚠️ **If You Get Errors:**

### Error: "launch manifest was created for a app, but this is a app"
**Solution:**
```bash
# Delete conflicting files
rm fly.toml
rm -rf .fly

# Try again
fly launch
```

### Error: "App name already taken"
**Solution:**
- Choose a different app name when prompted
- Or use: `fly launch --name your-unique-app-name`

### Error: "Docker build failed"
**Solution:**
- Check your Dockerfile syntax
- Ensure all dependencies are in requirements.txt
- Try: `fly deploy --local-only` for testing

## 📁 Configuration Files Created

- **`fly.toml`** - Fly.io app configuration
- **`Dockerfile`** - Container build instructions
- **`.dockerignore`** - Files excluded from build

## 🔧 Customization

### Change App Name
Edit `fly.toml`:
```toml
app = "your-custom-name"
```

### Adjust Resources
Edit `fly.toml`:
```toml
[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024  # Increase for better performance
```

### Environment Variables
```bash
fly secrets set SECRET_KEY="your-secret-key"
fly secrets set FLASK_ENV="production"
```

## 📊 Monitoring & Management

### View Logs
```bash
fly logs
```

### Check Status
```bash
fly status
```

### Scale Resources
```bash
fly scale memory 2048  # Increase RAM
fly scale vm 2         # Add more instances
```

### Update Deployment
```bash
fly deploy  # After making code changes
```

## 🗂️ File Structure

```
nids-system/
├── fly.toml           # Fly.io configuration
├── Dockerfile         # Container definition
├── .dockerignore      # Build exclusions
├── requirements.txt   # Python dependencies
├── app/              # Flask application
│   ├── app.py        # Main application
│   ├── templates/    # HTML templates
│   └── static/       # CSS, JS, images
├── artifact/         # ML model files
├── utils/            # Database utilities
└── users.db          # SQLite database
```

## 🐛 Troubleshooting

### Build Issues
```bash
fly logs --app your-app-name
```

### Database Issues
- SQLite database is included in container
- Data persists between deployments
- For production, consider external database

### Port Issues
- App runs on port 8080 internally
- Fly.io handles external routing

## 💰 Pricing

- **Free Tier**: 3 shared CPUs, 256MB RAM, 1GB storage
- **Pay-as-you-go**: $0.02/hour for usage beyond free tier
- **Hobby Plan**: $5/month for consistent usage

## 🔄 Alternative Deployments

If Fly.io doesn't meet your needs:

### Heroku (Alternative)
```bash
heroku create your-app
git push heroku main
```

### Railway (Alternative)
- Connect GitHub repository
- Automatic deployment
- Built-in database support

## 📞 Support

- **Fly.io Docs**: https://fly.io/docs
- **Community**: https://community.fly.io
- **Status Page**: https://status.fly.io

---

**Ready to deploy?** Run `fly launch` and your NIDS will be live in minutes! 🌐
