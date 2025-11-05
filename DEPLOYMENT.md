# Netlify Deployment Guide for NIDS

## Important Note
This Flask application with database functionality cannot be directly deployed to Netlify as a traditional Flask app. Netlify is primarily designed for static sites and serverless functions.

## Recommended Deployment Options

### Option 1: Heroku (Recommended for Full Flask App)
1. Create a Heroku account
2. Install Heroku CLI
3. Create a `Procfile`:
   ```
   web: python app/app.py
   ```
4. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 2: Vercel
1. Create a Vercel account
2. Connect your GitHub repository
3. Vercel can handle Python/Flask apps with proper configuration

### Option 3: Railway
1. Create a Railway account
2. Connect your GitHub repository
3. Automatic deployment with database support

### Option 4: DigitalOcean App Platform
1. Create a DigitalOcean account
2. Use App Platform for containerized deployment
3. Supports Flask applications with databases

## Current Netlify Setup
The current setup includes:
- `netlify.toml` configuration
- Netlify Functions for prediction API
- Static file serving

However, for a full-featured web application with user authentication and database, consider the alternatives above.

## Quick Netlify Deploy (Limited Functionality)
If you still want to deploy a demo version to Netlify:

1. **Connect Repository**:
   - Go to Netlify dashboard
   - Connect your GitHub repository

2. **Build Settings**:
   - Build command: `echo 'No build required'`
   - Publish directory: `app/static`
   - Functions directory: `netlify/functions`

3. **Environment Variables**:
   - Add `PYTHON_VERSION = 3.9`

4. **Deploy**:
   - Push to main branch
   - Netlify will deploy automatically

## Limitations on Netlify
- No persistent database (SQLite won't work)
- No user sessions/authentication
- Limited to static content + serverless functions
- No background tasks or persistent state

## Recommended Action
For a production deployment with full functionality, use **Heroku** or **Railway** instead of Netlify.
