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

### Step 2: Clean Start (Important!)
```bash
# Remove any existing fly.toml to avoid conflicts
rm fly.toml
```

### Step 3: Launch Your App (Skip Auto-Update)
```bash
# If you get auto-update errors, skip it:
fly launch --no-update-check
```
**Or use the older version:**
```bash
fly launch
# When it asks to update, say "No"
```

**Follow the prompts:**
- **App Name**: Choose unique name (e.g., `your-nids-app-123`)
- **Region**: Select closest to your users (e.g., `lax` for US West)
- **Organization**: Choose your account
- **Dockerfile**: It will detect your Dockerfile automatically
- **Database**: No (we're using SQLite)

### Step 4: Deploy
```bash
fly deploy
```

### Step 5: Open Your App
```bash
fly open
```

**Your NIDS will be live at:** `https://your-app-name.fly.dev`

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
