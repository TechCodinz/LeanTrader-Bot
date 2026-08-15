# LeanTrader Secure VPS Operations Bridge

This bridge connects the supported LeanTrader Ubuntu VPS to ChatGPT and Codex through OpenAI Secure MCP Tunnel. The VPS makes an outbound HTTPS connection; no MCP port or new inbound firewall rule is opened.

## Security boundary

The bridge deliberately does **not** offer a shell or filesystem browser. It cannot read `.env`, `/run/secrets`, exchange credentials, or the tunnel Runtime API key. It cannot enable live trading, change API keys, or remove `runtime/TESTNET_HALT`.

The exposed tools are limited to:

- bounded VPS and container health;
- allowlisted heartbeat projections;
- 20, 50, 100, or 200 redacted log lines;
- a confirmed service restart;
- a confirmed run of the pinned paper-authority bootstrap; and
- an idempotent Testnet emergency halt.

The tunnel daemon runs as `leantunnel`. The MCP server runs separately as `leanops`. A root-owned sudoers file permits only exact helper subcommands; it grants neither user Docker-group membership nor arbitrary root execution. Every MCP action is appended to `/var/log/leantrader-ops/audit.jsonl`.

## One-time VPS installation

Create the tunnel in OpenAI Platform first. Then create a **Runtime API key** with Tunnels Read + Use. Do not use an Admin API key and do not paste either key into chat.

Run the audited install command supplied with the release from a root-capable Termius session. The installer asks for the tunnel ID and Runtime API key through `/dev/tty`; the key stays hidden and is written to a root-controlled, tunnel-user-only file. It installs the checksum-pinned public `tunnel-client` release, validates the MCP profile with `doctor`, and starts `leantrader-tunnel.service`.

Useful local checks on the VPS:

```bash
sudo systemctl status leantrader-tunnel.service --no-pager
sudo journalctl -u leantrader-tunnel.service --no-pager --lines=80
curl -fsS http://127.0.0.1:8080/readyz
```

The local tunnel UI binds only to `127.0.0.1:8080`. Do not expose it through UFW or a public reverse proxy.

## Connector activation

While `leantrader-tunnel.service` is active, open ChatGPT **Settings → Connectors**, add the Secure MCP Tunnel, and select the tunnel created for this VPS. Once its tools appear in Work mode, start with `vps_health` and `leantrader_status`; use the verified deployment tool only after reviewing the reported state.
