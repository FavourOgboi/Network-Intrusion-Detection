# 🚀 Fly.io Deployment Guide for NIDS

## Why Fly.io?

Fly.io is the **perfect platform** for your Network Intrusion Detection System because:

- ✅ **Container-native** - Perfect for Flask applications
- ✅ **SQLite support** - Your database works without changes
- ✅ **Global deployment** - Fast access worldwide
- ✅ **Free tier** - Generous limits for development
- ✅ **Persistent storage** - Your data stays with the app
- ✅ **Simple scaling** - Easy to upgrade when needed

## 🛠️ Prerequisites

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **Fly CLI**: Install the command-line tool

### Install Fly CLI

**Windows (PowerShell):**
```powershell
# Using PowerShell
iwr https://fly.io/install.ps1 -useb | iex
```

**Or download from:** https://fly.io/docs/getting-started/installing-flyctl/

## 🚀 Quick Deployment

### Step 1: Authenticate
```bash
fly auth login
```

### Step 2: Launch Your App
```bash
fly launch
```
- Choose your app name (e.g., `nids-app`)
- Select region (choose closest to your users)
- Accept default configuration

### Step 3: Deploy
```bash
fly deploy
```

### Step 4: Open Your App
```bash
fly open
```

**Your NIDS will be live at:** `https://your-app-name.fly.dev`

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
