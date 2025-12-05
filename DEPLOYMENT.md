# 🚀 CrySense AI - Deployment Guide

This guide covers deploying the CrySense AI application to Render.com with both backend API and frontend static site.

## 📋 Prerequisites

- GitHub account with repository access
- Render.com account (free tier available)
- Git repository: `https://github.com/SeneKela/baby-cry-analyser.git`

## 🏗️ Architecture Overview

- **Backend**: FastAPI Python service (Render Web Service)
- **Frontend**: React + Vite static site (Render Static Site)
- **Model**: PyTorch model file (`cry_model.pth`) - 809KB, committed to Git

---

## 🔧 Deployment Steps

### 1️⃣ Push Code to GitHub

```bash
# Check current status
git status

# Add all deployment files
git add .

# Commit changes
git commit -m "Add Render deployment configuration"

# Push to GitHub
git push origin main
```

### 2️⃣ Deploy Backend to Render

#### Option A: Using Render Blueprint (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository: `SeneKela/baby-cry-analyser`
4. Render will detect `render.yaml` and create both services automatically
5. Review the configuration and click **"Apply"**

#### Option B: Manual Setup

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `baby-cry-analyser-backend`
   - **Runtime**: `Python 3`
   - **Root Directory**: `code/web/backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click **"Create Web Service"**

#### Backend Environment Variables

No additional environment variables are required. The backend will automatically:
- Use the model file at `../../../cry_model.pth` (relative path)
- Create a `temp/` directory for uploads and reports
- Listen on the port provided by Render via `$PORT`

#### Get Backend URL

After deployment, your backend will be available at:
```
https://baby-cry-analyser-backend.onrender.com
```

**Important**: Copy this URL - you'll need it for the frontend configuration.

### 3️⃣ Update Frontend Configuration

Before deploying the frontend, update the API URL:

1. Edit `code/web/frontend/.env.production`:
   ```env
   VITE_API_URL=https://baby-cry-analyser-backend.onrender.com
   ```

2. Commit and push the change:
   ```bash
   git add code/web/frontend/.env.production
   git commit -m "Update production API URL"
   git push origin main
   ```

### 4️⃣ Deploy Frontend to Render

#### Using Render Blueprint

If you used the Blueprint in step 2, the frontend is already deployed. Skip to step 5.

#### Manual Setup

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New"** → **"Static Site"**
3. Connect your GitHub repository
4. Configure the site:
   - **Name**: `baby-cry-analyser-frontend`
   - **Root Directory**: `code/web/frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
5. Click **"Create Static Site"**

#### Get Frontend URL

After deployment, your frontend will be available at:
```
https://baby-cry-analyser-frontend.onrender.com
```

### 5️⃣ Update Backend CORS Configuration

The backend needs to allow requests from your deployed frontend:

1. Edit `code/web/backend/main.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=[
           "http://localhost:5173",  # Development
           "https://senekela.github.io",  # GitHub Pages
           "https://baby-cry-analyser-frontend.onrender.com"  # Render frontend
       ],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. Commit and push:
   ```bash
   git add code/web/backend/main.py
   git commit -m "Add Render frontend to CORS origins"
   git push origin main
   ```

3. Render will automatically redeploy the backend with the new configuration.

---

## ✅ Verification

### Test Backend API

1. Visit your backend URL with `/docs` endpoint:
   ```
   https://baby-cry-analyser-backend.onrender.com/docs
   ```

2. You should see the FastAPI interactive documentation (Swagger UI)

### Test Frontend Application

1. Visit your frontend URL:
   ```
   https://baby-cry-analyser-frontend.onrender.com
   ```

2. Test the complete flow:
   - Upload a baby cry audio file (`.wav`, `.mp3`, or `.webm`)
   - Wait for analysis to complete
   - Verify prediction results are displayed
   - Check that confidence scores are shown
   - Confirm recommendations are provided
   - Test PDF report download (if applicable)

---

## 🔍 Troubleshooting

### Backend Issues

**Problem**: Backend fails to start
- **Check logs**: Go to Render dashboard → Backend service → Logs
- **Common causes**:
  - Missing dependencies in `requirements.txt`
  - Model file not found (check path: `../../../cry_model.pth`)
  - Port binding issues (ensure using `$PORT` environment variable)

**Problem**: Model file not found
- **Solution**: Verify `cry_model.pth` is committed to Git:
  ```bash
  git ls-files | grep cry_model.pth
  ```
- If missing, add it:
  ```bash
  git add code/cry_model.pth
  git commit -m "Add trained model file"
  git push origin main
  ```

### Frontend Issues

**Problem**: Frontend shows "Failed to fetch" or CORS errors
- **Solution**: Ensure backend CORS configuration includes frontend URL
- **Check**: Browser console for specific error messages
- **Verify**: Backend is running and accessible

**Problem**: API calls fail with 404
- **Solution**: Check `VITE_API_URL` in `.env.production` matches backend URL
- **Rebuild**: Trigger a manual deploy in Render dashboard

**Problem**: Blank page or build errors
- **Check logs**: Render dashboard → Frontend service → Deploy logs
- **Common causes**:
  - TypeScript compilation errors
  - Missing dependencies
  - Build command issues

### General Issues

**Problem**: Free tier services sleep after inactivity
- **Behavior**: Render free tier services sleep after 15 minutes of inactivity
- **Impact**: First request after sleep takes 30-60 seconds to wake up
- **Solution**: Upgrade to paid tier for always-on services, or accept the cold start delay

---

## 🔐 Security Considerations

### Environment Variables

Currently, no sensitive environment variables are required. If you add any in the future:

1. Go to Render Dashboard → Service → Environment
2. Add environment variables (they will be encrypted)
3. Never commit sensitive values to Git

### CORS Configuration

- Keep CORS origins as restrictive as possible
- Only add trusted domains
- Remove `http://localhost:5173` in production if not needed

### Model File Security

- The model file is currently public in the Git repository
- If you need to keep it private, consider:
  - Using a private GitHub repository
  - Storing the model in cloud storage (S3, GCS) and downloading on startup
  - Using Render disk storage

---

## 📊 Monitoring

### Render Dashboard

Monitor your services at:
- Backend: https://dashboard.render.com/web/[service-id]
- Frontend: https://dashboard.render.com/static/[site-id]

**Available metrics**:
- Request count
- Response times
- Error rates
- Build and deploy history
- Logs (real-time and historical)

### Logs

View logs in real-time:
```bash
# Using Render CLI (optional)
render logs -s baby-cry-analyser-backend
render logs -s baby-cry-analyser-frontend
```

---

## 🔄 Continuous Deployment

Render automatically deploys when you push to GitHub:

1. Make changes to your code
2. Commit and push to `main` branch
3. Render detects the push and triggers a new deployment
4. Monitor deployment progress in the dashboard

### Disable Auto-Deploy (Optional)

If you want manual control:
1. Go to Service Settings
2. Disable "Auto-Deploy"
3. Use "Manual Deploy" button when ready

---

## 💰 Cost Estimation

### Free Tier (Current)

- **Backend**: Free Web Service (750 hours/month)
- **Frontend**: Free Static Site (100GB bandwidth/month)
- **Limitations**:
  - Services sleep after 15 minutes of inactivity
  - Slower build times
  - Limited resources

### Paid Tier (Optional)

- **Starter**: $7/month per service
  - Always-on (no sleep)
  - Faster builds
  - More resources

---

## 📝 Next Steps

1. ✅ Deploy backend to Render
2. ✅ Deploy frontend to Render
3. ✅ Update CORS configuration
4. ✅ Test complete application flow
5. 🔄 Set up custom domain (optional)
6. 🔄 Configure monitoring and alerts (optional)
7. 🔄 Add CI/CD tests (optional)

---

## 🆘 Support

- **Render Documentation**: https://render.com/docs
- **Render Community**: https://community.render.com
- **GitHub Issues**: https://github.com/SeneKela/baby-cry-analyser/issues

---

**Last Updated**: December 5, 2024
