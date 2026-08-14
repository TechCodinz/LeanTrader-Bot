#!/usr/bin/env bash
set -Eeuo pipefail

readonly APP_ROOT="/opt/leantrader"
readonly APP_DIR="${APP_ROOT}/app"
readonly REPOSITORY_URL="https://github.com/TechCodinz/LeanTrader-Bot.git"
readonly RELEASE_COMMIT="6f1a08687e5ee001df3c0f4647b9ef7a9e5ebd3c"
readonly RELEASE_TREE="b4e038c125b465a8631fd653d0255e5c16f8d423"
readonly LOG_FILE="/var/log/leantrader-bootstrap.log"

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run this bootstrap as root." >&2
  exit 1
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
  local exit_code=$?
  echo "ERROR: bootstrap stopped on line ${BASH_LINENO[0]} (exit ${exit_code})."
  echo "Review ${LOG_FILE}; do not paste passwords or tokens into support messages."
  exit "${exit_code}"
}
trap on_error ERR

echo "== LeanTrader verified VPS bootstrap =="

if [[ ! -r /etc/os-release ]]; then
  echo "ERROR: /etc/os-release is unavailable." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "ERROR: this release requires Ubuntu 24.04; found ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "[1/8] Installing security and operating-system updates"
apt-get update
apt-get -y full-upgrade
apt-get install -y \
  ca-certificates \
  curl \
  fail2ban \
  git \
  jq \
  unattended-upgrades \
  ufw

echo "[2/8] Configuring host firewall and SSH abuse protection"
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment "SSH"
ufw allow 80/tcp comment "HTTP"
ufw allow 443/tcp comment "HTTPS"
ufw --force enable

install -d -m 0755 /etc/fail2ban/jail.d
cat > /etc/fail2ban/jail.d/leantrader-sshd.local <<'EOF'
[sshd]
enabled = true
backend = systemd
banaction = ufw
findtime = 10m
maxretry = 5
bantime = 1h
EOF
systemctl enable --now fail2ban unattended-upgrades
systemctl restart fail2ban

echo "[3/8] Configuring time synchronization and a small emergency swap"
timedatectl set-ntp true
if ! swapon --show=NAME --noheadings | grep -q .; then
  if [[ ! -e /swapfile ]]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
fi
if ! grep -qE '^/swapfile[[:space:]]' /etc/fstab; then
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
cat > /etc/sysctl.d/60-leantrader-memory.conf <<'EOF'
vm.swappiness=10
EOF
sysctl --system >/dev/null

echo "[4/8] Installing Docker Engine and Compose"
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt-get update
  apt-get install -y \
    containerd.io \
    docker-buildx-plugin \
    docker-ce \
    docker-ce-cli \
    docker-compose-plugin
fi
systemctl enable --now docker
docker version --format 'Docker server: {{.Server.Version}}'
docker compose version

echo "[5/8] Fetching the exact audited LeanTrader release"
install -d -m 0755 "${APP_ROOT}"
if [[ -d "${APP_DIR}/.git" ]]; then
  if ! git -C "${APP_DIR}" diff --quiet || ! git -C "${APP_DIR}" diff --cached --quiet; then
    echo "ERROR: ${APP_DIR} contains uncommitted changes; refusing to overwrite them." >&2
    exit 1
  fi
  git -C "${APP_DIR}" fetch --prune origin
elif [[ -e "${APP_DIR}" ]]; then
  echo "ERROR: ${APP_DIR} exists but is not a Git checkout." >&2
  exit 1
else
  git clone --filter=blob:none "${REPOSITORY_URL}" "${APP_DIR}"
fi
git -C "${APP_DIR}" checkout --detach "${RELEASE_COMMIT}"

actual_tree="$(git -C "${APP_DIR}" rev-parse 'HEAD^{tree}')"
if [[ "${actual_tree}" != "${RELEASE_TREE}" ]]; then
  echo "ERROR: release tree mismatch: expected ${RELEASE_TREE}, got ${actual_tree}." >&2
  exit 1
fi
echo "Verified release tree: ${actual_tree}"

echo "[6/8] Creating the paper-authority configuration and persistent state"
cd "${APP_DIR}"
if [[ ! -e .env ]]; then
  cp .env.production.example .env
fi
chmod 600 .env

for required_setting in \
  'TRADING_MODE=paper' \
  'ENABLE_LIVE=false' \
  'ALLOW_LIVE=false' \
  'LIVE_CONFIRM=NO'; do
  if ! grep -qx "${required_setting}" .env; then
    echo "ERROR: .env is not the approved no-live configuration (${required_setting})." >&2
    exit 1
  fi
done

install -d -m 0750 runtime logs
install -d -m 0750 -o root -g 10001 secrets
chown -R 10001:10001 runtime logs

echo "[7/8] Building and starting LeanTrader"
docker compose config --quiet
docker compose up -d --build

container_id="$(docker compose ps -q leantrader)"
if [[ -z "${container_id}" ]]; then
  echo "ERROR: Docker Compose did not create the leantrader service." >&2
  exit 1
fi

echo "[8/8] Waiting for the supported runtime healthcheck"
health="starting"
for _ in $(seq 1 48); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  if [[ "${health}" == "unhealthy" || "${health}" == "missing" ]]; then
    break
  fi
  sleep 5
done

docker compose ps
if [[ "${health}" != "healthy" ]]; then
  echo "ERROR: LeanTrader health is ${health}. Recent logs follow:" >&2
  docker compose logs --tail=120 leantrader >&2
  exit 1
fi

echo
echo "LEANTRADER_BOOTSTRAP_OK"
echo "Mode: paper authority; optional bounded Bybit Testnet mirror"
echo "Commit: ${RELEASE_COMMIT}"
echo "Tree: ${RELEASE_TREE}"
echo "Health: ${health}"
echo "Firewall: $(ufw status | head -n 1)"
echo "Fail2ban: $(systemctl is-active fail2ban)"
echo "Heartbeat: ${APP_DIR}/runtime/vps_heartbeat.json"
if [[ -f /var/run/reboot-required ]]; then
  echo "Reboot: required after reviewing the running service"
else
  echo "Reboot: not currently required"
fi
echo "Never add production exchange credentials. Testnet keys belong only in root-owned secret files."
