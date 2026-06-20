#!/bin/bash
# ============================================
# Traffic Violation Detection — GCP Deploy Script
# ============================================
#
# Run this in GCP Browser SSH:
#
#   curl -sSL https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/deploy.sh | bash
#
#   — OR —
#
#   wget -qO deploy.sh https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/deploy.sh
#   chmod +x deploy.sh
#   ./deploy.sh
#
# ============================================
set -euo pipefail

# ┌──────────────────────────────────────────┐
# │  ⚙️  CONFIGURE THESE                     │
# └──────────────────────────────────────────┘
GITHUB_REPO="https://github.com/Aj242005/fg-r2.git"
BRANCH="main"                                                # ← branch to deploy
APP_DIR="$HOME/fgr2-backend"

# ──────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

echo ""
echo "=========================================="
echo "🚦 Traffic Violation Detection — Deployer"
echo "=========================================="
echo ""

# ── Step 1: Install Docker ──
if command -v docker &> /dev/null; then
    log "Docker already installed: $(docker --version)"
else
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    log "Docker installed"
fi

# Ensure current user can run docker
if ! docker info &> /dev/null 2>&1; then
    warn "Adding current user to docker group (takes effect now via newgrp)"
    sudo usermod -aG docker "$USER"
    exec sg docker -c "$0 $*"
fi

# Verify docker compose
if docker compose version &> /dev/null; then
    log "Docker Compose: $(docker compose version --short)"
else
    err "Docker Compose v2 not found. Re-install Docker."
fi

# ── Step 2: Install git if missing ──
if ! command -v git &> /dev/null; then
    info "Installing git..."
    sudo apt-get update -qq && sudo apt-get install -y git > /dev/null
    log "Git installed"
fi

# ── Step 3: Clone or pull repo ──
if [ -d "$APP_DIR/.git" ]; then
    info "Repo already cloned — pulling latest..."
    cd "$APP_DIR"
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
    log "Code updated to latest $BRANCH"
else
    info "Cloning $GITHUB_REPO ($BRANCH)..."
    git clone --branch "$BRANCH" --single-branch --depth 1 "$GITHUB_REPO" "$APP_DIR"
    log "Repo cloned to $APP_DIR"
fi

# If the repo root IS the code/ dir, stay here.
# If code/ is a subdirectory, cd into it.
if [ -f "$APP_DIR/code/run.py" ]; then
    cd "$APP_DIR/code"
    info "Detected code/ subdirectory — deploying from there"
elif [ -f "$APP_DIR/run.py" ]; then
    cd "$APP_DIR"
else
    err "Could not find run.py in repo. Check your repo structure."
fi

DEPLOY_DIR=$(pwd)
log "Deploy directory: $DEPLOY_DIR"

# ── Step 4: Verify files ──
for f in Dockerfile docker-compose.yml Caddyfile run.py config.py requirements.txt; do
    [ -f "$f" ] || err "Missing: $f"
done
log "All deployment files verified"

# Model check
echo ""
if [ -f "models/violation_model.pt" ]; then
    log "violation_model.pt found ($(du -h models/violation_model.pt | cut -f1))"
else
    warn "models/violation_model.pt NOT found"
    warn "Upload it later: scp violation_model.pt VM_IP:$DEPLOY_DIR/models/"
    warn "Or use browser SSH upload → mv ~/violation_model.pt $DEPLOY_DIR/models/"
fi

if [ -f "models/plate_model.pt" ]; then
    log "plate_model.pt found ($(du -h models/plate_model.pt | cut -f1))"
else
    warn "models/plate_model.pt NOT found — plate OCR will be disabled"
fi

# ── Step 5: Open firewall ──
if command -v ufw &> /dev/null && sudo ufw status 2>/dev/null | grep -q "active"; then
    sudo ufw allow 80/tcp > /dev/null
    sudo ufw allow 443/tcp > /dev/null
    log "UFW: ports 80 & 443 opened"
fi

# ── Step 6: Build ──
echo ""
echo "=========================================="
echo "🔨 Building Docker images (first build takes ~3-5 min)..."
echo "=========================================="
echo ""

docker compose build

# ── Step 7: Start ──
echo ""
echo "=========================================="
echo "🚀 Starting services..."
echo "=========================================="
echo ""

docker compose up -d

# ── Step 8: Health check ──
echo ""
info "Waiting for app to start (first run downloads EasyOCR models, ~60-90s)..."
echo ""

MAX_WAIT=180
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -sf http://localhost:8000/api/health 2>/dev/null) && break
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    printf "   ⏳ %ds / %ds\r" "$ELAPSED" "$MAX_WAIT"
done
echo ""

if [ $ELAPSED -ge $MAX_WAIT ]; then
    warn "App hasn't responded yet. It may still be loading models."
    warn "Check: docker compose logs -f app"
else
    log "App is healthy! ✨"
fi

# ── Step 9: Summary ──
EXTERNAL_IP=$(curl -sf http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null || echo "<run: curl ifconfig.me>")

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "  📍 VM IP:          $EXTERNAL_IP"
echo "  🌐 Domain:         https://fgr2-backend.mooo.com"
echo "  📖 API Docs:       https://fgr2-backend.mooo.com/docs"
echo "  ❤️  Health:         https://fgr2-backend.mooo.com/api/health"
echo ""
echo "  ──────────────────────────────────────"
echo "  ⚠️  NEXT STEP: Point your FreeDNS A record"
echo "     fgr2-backend.mooo.com → $EXTERNAL_IP"
echo "  ──────────────────────────────────────"
echo ""
echo "  📋 Commands:"
echo "     docker compose logs -f        # all logs"
echo "     docker compose logs -f app    # app only"
echo "     docker compose logs -f caddy  # TLS/proxy"
echo "     docker compose restart app    # after model swap"
echo "     docker compose down           # stop all"
echo ""
echo "  📦 Upload models via browser SSH:"
echo "     Upload → gear icon (⚙️) → Upload file"
echo "     mv ~/violation_model.pt $DEPLOY_DIR/models/"
echo "     mv ~/plate_model.pt $DEPLOY_DIR/models/"
echo "     docker compose restart app"
echo ""
