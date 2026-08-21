#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="TechCodinz/LeanTrader-Bot"
readonly RELEASE_REF="${BRIDGE_REF:-}"
readonly APP_DIR="/opt/leantrader/app"
readonly OPS_DIR="/opt/leantrader-ops"
readonly WORK_DIR="$(mktemp -d)"
readonly SERVER_SHA="37cb9f3c9dae507f05eb334bf286048cb58555a93b89bae1dadc8ad304846619"
readonly HELPER_SHA="968b8027010bdac49d2d839dd359d8388fb1a5ed720c70900f7f625d213f97ce"
readonly SUDOERS_SHA="449cc4aaa3b81c4b5e337e81c7d53961ecab1383552b395741854da56803c277"
readonly SELFTEST_SHA="58137fdd1b95efce25741d6f917ab9c8a09b7ff59dfbe489752b949c5f3477c7"
readonly DOC_SHA="1f5bfc478ed2de9f780b4a739996bf85b32068eedcfaab7bb29541a5a4d54617"

cleanup() {
  rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "run this upgrade as root"
fi
if [[ ! "${RELEASE_REF}" =~ ^[0-9a-f]{40}$ ]]; then
  fail "BRIDGE_REF must be the pinned 40-character recovery commit"
fi
if [[ ! -d "${APP_DIR}/.git" ]]; then
  fail "LeanTrader Git repository is unavailable at ${APP_DIR}"
fi
if [[ ! -x "${OPS_DIR}/venv/bin/python" ]]; then
  fail "the existing LeanTrader operations environment is unavailable"
fi

download_verified() {
  local filename="$1"
  local expected_sha="$2"
  curl -fsSL --proto '=https' --tlsv1.2 \
    "https://raw.githubusercontent.com/${REPOSITORY}/${RELEASE_REF}/ops/vps_bridge/recovery_v1_1/${filename}" \
    -o "${WORK_DIR}/${filename}"
  printf '%s  %s\n' "${expected_sha}" "${WORK_DIR}/${filename}" | \
    sha256sum --check --status || fail "integrity check failed for ${filename}"
}

download_verified "server.py" "${SERVER_SHA}"
download_verified "privileged_helper.py" "${HELPER_SHA}"
download_verified "leantrader-ops.sudoers" "${SUDOERS_SHA}"
download_verified "selftest.py" "${SELFTEST_SHA}"
download_verified "VPS_MCP_RECONCILIATION.md" "${DOC_SHA}"

python3 -m py_compile "${WORK_DIR}/server.py" "${WORK_DIR}/privileged_helper.py" "${WORK_DIR}/selftest.py"
python3 "${WORK_DIR}/selftest.py"
visudo -cf "${WORK_DIR}/leantrader-ops.sudoers" >/dev/null

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="${OPS_DIR}/backups/${timestamp}"
install -d -m 0700 -o root -g root "${backup_dir}"
cp -a "${OPS_DIR}/server.py" "${backup_dir}/server.py"
cp -a /usr/local/sbin/leantrader-ops-helper "${backup_dir}/leantrader-ops-helper"
cp -a /etc/sudoers.d/leantrader-ops "${backup_dir}/leantrader-ops.sudoers"
sha256sum "${backup_dir}/server.py" "${backup_dir}/leantrader-ops-helper" \
  "${backup_dir}/leantrader-ops.sudoers" >"${backup_dir}/SHA256SUMS"

install -m 0644 -o root -g root "${WORK_DIR}/server.py" "${OPS_DIR}/server.py"
install -m 0555 -o root -g root "${WORK_DIR}/privileged_helper.py" /usr/local/sbin/leantrader-ops-helper
install -m 0440 -o root -g root "${WORK_DIR}/leantrader-ops.sudoers" /etc/sudoers.d/leantrader-ops
visudo -cf /etc/sudoers.d/leantrader-ops >/dev/null

install -m 0644 -o root -g root "${WORK_DIR}/server.py" "${APP_DIR}/ops/vps_bridge/server.py"
install -m 0644 -o root -g root "${WORK_DIR}/privileged_helper.py" "${APP_DIR}/ops/vps_bridge/privileged_helper.py"
install -m 0644 -o root -g root "${WORK_DIR}/leantrader-ops.sudoers" "${APP_DIR}/ops/vps_bridge/leantrader-ops.sudoers"
install -m 0644 -o root -g root "${WORK_DIR}/selftest.py" "${APP_DIR}/ops/vps_bridge/reconciliation_selftest.py"
install -m 0644 -o root -g root "${WORK_DIR}/VPS_MCP_RECONCILIATION.md" "${APP_DIR}/VPS_MCP_RECONCILIATION.md"

installer="${APP_DIR}/scripts/install_vps_ops_bridge.sh"
if [[ -f "${installer}" ]]; then
  sed -i \
    -e 's/cfbe6b3a82cc94aa407a6c683410284a5a0eef644ee545a73afcb98bbf0e8cce/37cb9f3c9dae507f05eb334bf286048cb58555a93b89bae1dadc8ad304846619/g' \
    -e 's/addec96953d568aa4f96a969e98b7fc4872572927aeb6923637ebc635b368ff3/968b8027010bdac49d2d839dd359d8388fb1a5ed720c70900f7f625d213f97ce/g' \
    -e 's/8168dcfb21a15c378dbda36a536b1abe2ecd70d16e933e2aed32419a47d37e0c/449cc4aaa3b81c4b5e337e81c7d53961ecab1383552b395741854da56803c277/g' \
    "${installer}"
fi

python3 -m py_compile "${APP_DIR}/ops/vps_bridge/server.py" "${APP_DIR}/ops/vps_bridge/privileged_helper.py"
python3 "${APP_DIR}/ops/vps_bridge/reconciliation_selftest.py"

printf '%s' '{"operation":"inventory","path":"","ref":"HEAD","scope":"worktree","start_line":1,"end_line":20,"limit":5}' | \
  runuser -u leanops -- sudo -n /usr/local/sbin/leantrader-ops-helper repo-read \
  >"${WORK_DIR}/inventory.json"
python3 -m json.tool "${WORK_DIR}/inventory.json" >/dev/null

systemctl restart leantrader-tunnel.service
systemctl is-active --quiet leantrader-tunnel.service || fail "tunnel failed to restart"
systemctl is-active --quiet leantrader-evolution-sidecar.service || fail "Evolution Sidecar is not active"

container_id="$(docker compose -f "${APP_DIR}/docker-compose.yml" ps -q leantrader)"
if [[ -z "${container_id}" ]]; then
  fail "LeanTrader paper container is not present"
fi
container_health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_id}")"
if [[ "${container_health}" != "healthy" && "${container_health}" != "running" ]]; then
  fail "LeanTrader paper container is not healthy after bridge-only upgrade"
fi

echo "LEANTRADER_REPOSITORY_BRIDGE_V1_1_OK"
echo "Bridge backup: ${backup_dir}"
echo "Tunnel: $(systemctl is-active leantrader-tunnel.service)"
echo "Evolution Sidecar: $(systemctl is-active leantrader-evolution-sidecar.service)"
echo "LeanTrader container: ${container_health}"
echo "Next: refresh LeanTrader VPS Operations in ChatGPT Plugins."
