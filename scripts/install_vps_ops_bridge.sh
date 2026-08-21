#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="TechCodinz/LeanTrader-Bot"
readonly RELEASE_REF="${OPS_RELEASE_REF:-}"
readonly TUNNEL_CLIENT_VERSION="v0.0.10"
readonly TUNNEL_CLIENT_BASE="https://persistent.oaistatic.com/tunnel-client/${TUNNEL_CLIENT_VERSION}"
readonly INSTALL_ROOT="/opt/leantrader-ops"
readonly CONFIG_ROOT="/etc/leantrader-ops"
readonly LOG_ROOT="/var/log/leantrader-ops"
readonly PROFILE_PATH="${CONFIG_ROOT}/leantrader.yaml"
readonly API_KEY_PATH="${CONFIG_ROOT}/runtime_api_key"
readonly WORK_DIR="$(mktemp -d)"

# These hashes are updated with the audited files in the same release commit.
readonly SERVER_SHA="37cb9f3c9dae507f05eb334bf286048cb58555a93b89bae1dadc8ad304846619"
readonly HELPER_SHA="5ad31aa1be843ccb9346a03111373a72fb3634268acd5c155bd4c83a3234d2ac"
readonly REQUIREMENTS_SHA="2ba90ca762c2825bcc60e50421fa8431e57ae64c9b56d3ed1159a3d2c65edfcf"
readonly SUDOERS_SHA="449cc4aaa3b81c4b5e337e81c7d53961ecab1383552b395741854da56803c277"
readonly SERVICE_SHA="f06e1147322643f2d0d7e90a1bc87a56ed21ea53c4ff37c7738af7ea15c80093"
readonly WRAPPER_SHA="d96d37c89ec7352810c8be219e09f5ae3b9c9c9d68dcc50589a3743a3cf0f8f1"
readonly BOOTSTRAP_SHA="0b3382121f8ce5bdac01c38ad57a2fe12d9df50a2b814d951ed9e8ca0656ed97"

cleanup() {
  rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "run this installer as root"
fi
if [[ -z "${RELEASE_REF}" || ! "${RELEASE_REF}" =~ ^[0-9a-f]{40}$ ]]; then
  fail "OPS_RELEASE_REF must be the audited 40-character Git commit supplied with the install command"
fi
if [[ ! -r /etc/os-release ]]; then
  fail "/etc/os-release is unavailable"
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  fail "this installer supports Ubuntu 24.04; found ${PRETTY_NAME:-unknown}"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl python3 python3-venv sudo unzip

download_verified() {
  local repo_path="$1"
  local expected_sha="$2"
  local destination="$3"
  curl -fsSL --proto '=https' --tlsv1.2 \
    "https://raw.githubusercontent.com/${REPOSITORY}/${RELEASE_REF}/${repo_path}" \
    -o "${destination}"
  printf '%s  %s\n' "${expected_sha}" "${destination}" | sha256sum --check --status || \
    fail "integrity check failed for ${repo_path}"
}

download_verified "ops/vps_bridge/server.py" "${SERVER_SHA}" "${WORK_DIR}/server.py"
download_verified "ops/vps_bridge/privileged_helper.py" "${HELPER_SHA}" "${WORK_DIR}/privileged_helper.py"
download_verified "ops/vps_bridge/requirements.txt" "${REQUIREMENTS_SHA}" "${WORK_DIR}/requirements.txt"
download_verified "ops/vps_bridge/leantrader-ops.sudoers" "${SUDOERS_SHA}" "${WORK_DIR}/leantrader-ops.sudoers"
download_verified "ops/vps_bridge/leantrader-tunnel.service" "${SERVICE_SHA}" "${WORK_DIR}/leantrader-tunnel.service"
download_verified "ops/vps_bridge/leantrader-mcp-server" "${WRAPPER_SHA}" "${WORK_DIR}/leantrader-mcp-server"
download_verified "scripts/bootstrap_verified_vps.sh" "${BOOTSTRAP_SHA}" "${WORK_DIR}/bootstrap_verified_vps.sh"

case "$(uname -m)" in
  x86_64)
    tunnel_archive="tunnel-client-${TUNNEL_CLIENT_VERSION}-linux-amd64.zip"
    tunnel_sha="b9e0388a343f2d7adeff3992f411a0bd3d916a64bc56534aac5fd15ac1b20cd5"
    ;;
  aarch64|arm64)
    tunnel_archive="tunnel-client-${TUNNEL_CLIENT_VERSION}-linux-arm64.zip"
    tunnel_sha="b842a9b2352eebd80514cf01a1fbb1c0d400a7d24a4015e85a7ea5f1aeaa5b30"
    ;;
  *) fail "unsupported CPU architecture: $(uname -m)" ;;
esac

curl -fsSL --proto '=https' --tlsv1.2 \
  "${TUNNEL_CLIENT_BASE}/${tunnel_archive}" -o "${WORK_DIR}/${tunnel_archive}"
printf '%s  %s\n' "${tunnel_sha}" "${WORK_DIR}/${tunnel_archive}" | \
  sha256sum --check --status || fail "tunnel-client integrity check failed"
unzip -q "${WORK_DIR}/${tunnel_archive}" -d "${WORK_DIR}/tunnel-client"
"${WORK_DIR}/tunnel-client/tunnel-client" --version
install -m 0555 -o root -g root \
  "${WORK_DIR}/tunnel-client/tunnel-client" /usr/local/bin/tunnel-client

id -u leanops >/dev/null 2>&1 || \
  useradd --system --create-home --home-dir /var/lib/leantrader-ops --shell /usr/sbin/nologin leanops
id -u leantunnel >/dev/null 2>&1 || \
  useradd --system --create-home --home-dir /var/lib/leantunnel --shell /usr/sbin/nologin leantunnel

install -d -m 0755 -o root -g root "${INSTALL_ROOT}"
install -d -m 0750 -o root -g leantunnel "${CONFIG_ROOT}"
install -d -m 0750 -o leanops -g leanops "${LOG_ROOT}"
python3 -m venv "${INSTALL_ROOT}/venv"
"${INSTALL_ROOT}/venv/bin/python" -m pip install --upgrade pip
"${INSTALL_ROOT}/venv/bin/python" -m pip install --requirement "${WORK_DIR}/requirements.txt"

install -m 0644 -o root -g root "${WORK_DIR}/server.py" "${INSTALL_ROOT}/server.py"
install -m 0555 -o root -g root "${WORK_DIR}/privileged_helper.py" /usr/local/sbin/leantrader-ops-helper
install -m 0555 -o root -g root "${WORK_DIR}/leantrader-mcp-server" /usr/local/bin/leantrader-mcp-server
install -m 0555 -o root -g root "${WORK_DIR}/bootstrap_verified_vps.sh" /usr/local/sbin/leantrader-bootstrap-verified
install -m 0440 -o root -g root "${WORK_DIR}/leantrader-ops.sudoers" /etc/sudoers.d/leantrader-ops
visudo -cf /etc/sudoers.d/leantrader-ops >/dev/null
install -m 0644 -o root -g root "${WORK_DIR}/leantrader-tunnel.service" /etc/systemd/system/leantrader-tunnel.service

profile_dir="/var/lib/leantunnel/profiles"
install -d -m 0700 -o leantunnel -g leantunnel "${profile_dir}"

if [[ -s "${PROFILE_PATH}" && -s "${API_KEY_PATH}"       && -z "${CONTROL_PLANE_TUNNEL_ID:-}"       && -z "${CONTROL_PLANE_API_KEY:-}" ]]; then
  echo "Preserving existing LeanTrader tunnel profile and Runtime API key."
else
  if [[ -z "${CONTROL_PLANE_TUNNEL_ID:-}" ]]; then
    read -r -p "Paste the tunnel ID (starts with tunnel_): " CONTROL_PLANE_TUNNEL_ID </dev/tty
  fi
  if [[ ! "${CONTROL_PLANE_TUNNEL_ID}" =~ ^tunnel_[A-Za-z0-9]+$ ]]; then
    fail "invalid tunnel ID"
  fi
  if [[ -z "${CONTROL_PLANE_API_KEY:-}" ]]; then
    read -r -s -p "Paste the Runtime API key (hidden): " CONTROL_PLANE_API_KEY </dev/tty
    echo >/dev/tty
  fi
  if [[ -z "${CONTROL_PLANE_API_KEY}" || "${CONTROL_PLANE_API_KEY}" =~ [[:space:]] ]]; then
    fail "runtime API key is empty or contains whitespace"
  fi

  install -m 0400 -o leantunnel -g leantunnel /dev/null "${API_KEY_PATH}"
  printf '%s\n' "${CONTROL_PLANE_API_KEY}" >"${API_KEY_PATH}"
  unset CONTROL_PLANE_API_KEY

  runuser -u leantunnel -- /usr/local/bin/tunnel-client init \
    --force \
    --profile leantrader \
    --profile-dir "${profile_dir}" \
    --tunnel-id "${CONTROL_PLANE_TUNNEL_ID}" \
    --control-plane-api-key-ref "file:${API_KEY_PATH}" \
    --health-listen-addr "127.0.0.1:8080" \
    --mcp-command "/usr/bin/sudo -n -u leanops /usr/local/bin/leantrader-mcp-server"
  install -m 0440 -o root -g leantunnel "${profile_dir}/leantrader.yaml" "${PROFILE_PATH}"
fi

runuser -u leantunnel -- /usr/local/bin/tunnel-client doctor \
  --profile-file "${PROFILE_PATH}" --explain
systemctl daemon-reload
systemctl enable --now leantrader-tunnel.service

for _ in $(seq 1 20); do
  if systemctl is-active --quiet leantrader-tunnel.service; then
    break
  fi
  sleep 1
done
systemctl is-active --quiet leantrader-tunnel.service || {
  journalctl -u leantrader-tunnel.service --no-pager --lines=80 >&2
  fail "tunnel service did not become active"
}

echo
echo "LEANTRADER_VPS_BRIDGE_OK"
echo "Tunnel service: $(systemctl is-active leantrader-tunnel.service)"
echo "Local health UI: http://127.0.0.1:8080/ui (VPS-local only)"
echo "Next: open ChatGPT Settings > Connectors and add the tunnel while this service is running."
echo "Do not paste the Runtime API key into ChatGPT, GitHub, or support messages."
