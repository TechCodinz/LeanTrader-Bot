# Monitoring Runbook (Prometheus + Grafana)

## Prometheus

1) Install Prometheus (Ubuntu):
```bash
sudo useradd --no-create-home --shell /usr/sbin/nologin prometheus || true
sudo mkdir -p /etc/prometheus /var/lib/prometheus
sudo chown -R prometheus:prometheus /etc/prometheus /var/lib/prometheus
curl -fsSL https://github.com/prometheus/prometheus/releases/latest/download/prometheus-*-linux-amd64.tar.gz -o /tmp/prom.tgz
sudo tar -xzf /tmp/prom.tgz -C /tmp
cd /tmp/prometheus-*-linux-amd64
sudo cp prometheus promtool /usr/local/bin/
sudo cp -r consoles console_libraries /etc/prometheus/
```

2) Copy our scrape config and service:
```bash
sudo mkdir -p /etc/prometheus
sudo cp /opt/ultrabot/monitoring/prometheus.yml /etc/prometheus/prometheus.yml
cat <<'UNIT' | sudo tee /etc/systemd/system/prometheus.service
[Unit]
Description=Prometheus
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus --web.listen-address=0.0.0.0:9090
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

3) Verify: http://<server-ip>:9090

## Grafana

1) Install Grafana OSS (Ubuntu):
```bash
sudo apt-get update && sudo apt-get install -y apt-transport-https software-properties-common wget
sudo mkdir -p /etc/apt/keyrings
wget -qO- https://packages.grafana.com/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update && sudo apt-get install -y grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

2) Login: http://<server-ip>:3000 (admin/admin)
3) Add Prometheus as data source: URL http://localhost:9090
4) Import dashboard: upload `/opt/ultrabot/monitoring/grafana_ultra_dashboard.json`

## Exporter

- Ensure `tools/metrics_http.py` is running on port 9100 (we added a systemd unit earlier or run:
```bash
METRICS_PORT=9100 nohup /opt/ultrabot/.venv/bin/python /opt/ultrabot/tools/metrics_http.py >/opt/ultrabot/runtime/metrics_http.log 2>&1 &
```
- Prometheus scrapes `127.0.0.1:9100/metrics` per the provided config.