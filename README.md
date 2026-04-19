# Video Speech Enhancer — Render Deployment

## Deploy Steps

1. Push this folder to a new GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects the Dockerfile
5. Click **Deploy** — live in ~5 minutes

## Keep-Alive (prevent spin-down on free tier)

Render's free tier spins down after 15 min of inactivity.
To prevent this, use UptimeRobot (free):

1. Go to https://uptimerobot.com → Register (free)
2. New Monitor → HTTP(s)
3. URL: `https://your-app.onrender.com` (use `/` as the ping path)
4. Interval: **14 minutes**
5. Save — your app will never sleep again

## Notes
- Port: 10000 (Render's default)
- First request after deploy downloads ~30MB DeepFilterNet weights (one-time)
- CPU processing: ~1x realtime
