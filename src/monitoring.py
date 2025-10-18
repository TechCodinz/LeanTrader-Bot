"""
Comprehensive Monitoring System for Trading Bot
Real-time monitoring, alerting, and health checks
"""

import asyncio
import os
import psutil
from datetime import datetime
import sqlite3
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status data structure"""

    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]

@dataclass
class Alert:
    """Alert data structure"""

    level: str  # 'info', 'warning', 'error', 'critical'
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False

class SystemMonitor:
    """System resource monitoring"""

    def __init__(self):
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent,
        }

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percentage': (disk.used / disk.total) * 100,
        }

    def get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
        }

    def check_system_health(self) -> HealthStatus:
        """Check overall system health"""
        cpu_usage = self.get_cpu_usage()
        memory_info = self.get_memory_usage()
        disk_info = self.get_disk_usage()

        status = 'healthy'
        message = 'System is running normally'

        if cpu_usage > self.cpu_threshold:
            status = 'warning'
            message = f'High CPU usage: {cpu_usage:.1f}%'
        elif memory_info['percentage'] > self.memory_threshold:
            status = 'warning'
            message = f'High memory usage: {memory_info["percentage"]:.1f}%'
        elif disk_info['percentage'] > self.disk_threshold:
            status = 'critical'
            message = f'High disk usage: {disk_info["percentage"]:.1f}%'

        return HealthStatus(
            component='system',
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={
                'cpu_usage': cpu_usage,
                'memory_usage': memory_info['percentage'],
                'disk_usage': disk_info['percentage'],
                'memory_total': memory_info['total'],
                'memory_available': memory_info['available'],
                'disk_total': disk_info['total'],
                'disk_free': disk_info['free'],
            },
        )

class DatabaseMonitor:
    """Database monitoring and health checks"""

    def __init__(self, database_path: str):
        self.database_path = database_path

    def check_connection(self) -> bool:
        """Check database connection"""
        try:
            conn = sqlite3.connect(self.database_path)
            conn.execute('SELECT 1')
            conn.close()
            return True
        except Exception:
            return False

    def get_database_size(self) -> int:
        """Get database file size in bytes"""
        try:
            return os.path.getsize(self.database_path)
        except Exception:
            return 0

    def get_table_counts(self) -> Dict[str, int]:
        """Get record counts for all tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            tables = [
                'market_data',
                'positions',
                'trades',
                'portfolio_history',
                'risk_metrics',
                'model_performance',
                'trading_signals',
                'bot_status',
            ]

            counts: Dict[str, int] = {}
            for table in tables:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    counts[table] = cursor.fetchone()[0]
                except Exception:
                    counts[table] = 0

            conn.close()
            return counts
        except Exception:
            return {}

    def check_database_health(self) -> HealthStatus:
        """Check database health"""
        if not self.check_connection():
            return HealthStatus(
                component='database',
                status='critical',
                message='Database connection failed',
                timestamp=datetime.now(),
                metrics={},
            )

        table_counts = self.get_table_counts()
        db_size = self.get_database_size()

        # Check for empty critical tables
        critical_tables = ['market_data', 'positions', 'trades']
        empty_tables = [table for table in critical_tables if table_counts.get(table, 0) == 0]

        if empty_tables:
            status = 'warning'
            message = f'Empty critical tables: {", ".join(empty_tables)}'
        else:
            status = 'healthy'
            message = 'Database is functioning normally'

        return HealthStatus(
            component='database',
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={'database_size': db_size, 'table_counts': table_counts},
        )

class TradingBotMonitor:
    """Trading bot specific monitoring"""

    def __init__(self, trading_bot):
        self.trading_bot = trading_bot

    def check_bot_status(self) -> HealthStatus:
        """Check trading bot status"""
        try:
            status = self.trading_bot.get_status()

            if not status['running']:
                return HealthStatus(
                    component='trading_bot',
                    status='critical',
                    message='Trading bot is not running',
                    timestamp=datetime.now(),
                    metrics=status,
                )

            # Check for recent activity
            positions_count = status.get('positions_count', 0)
            portfolio_value = status.get('portfolio_value', 0)

            if portfolio_value <= 0:
                status_level = 'warning'
                message = 'Portfolio value is zero or negative'
            elif positions_count > 20:
                status_level = 'warning'
                message = f'High number of positions: {positions_count}'
            else:
                status_level = 'healthy'
                message = 'Trading bot is running normally'

            return HealthStatus(
                component='trading_bot',
                status=status_level,
                message=message,
                timestamp=datetime.now(),
                metrics=status,
            )

        except Exception as e:
            return HealthStatus(
                component='trading_bot',
                status='critical',
                message=f'Trading bot check failed: {str(e)}',
                timestamp=datetime.now(),
                metrics={},
            )

class AlertManager:
    """Alert management system"""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_cooldown = 300  # 5 minutes
        self.last_alert_time: Dict[str, datetime] = {}

    def add_alert(self, level: str, component: str, message: str) -> None:
        """Add a new alert"""
        alert = Alert(level=level, component=component, message=message, timestamp=datetime.now())

        # Check cooldown
        alert_key = f"{component}:{level}"
        if alert_key in self.last_alert_time:
            time_since_last = (datetime.now() - self.last_alert_time[alert_key]).total_seconds()
            if time_since_last < self.alert_cooldown:
                return

        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = datetime.now()

        logger.warning(f"ALERT [{level.upper()}] {component}: {message}")

    def resolve_alert(self, component: str, message: str) -> None:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.component == component and alert.message == message and not alert.resolved:
                alert.resolved = True
                logger.info(f"ALERT RESOLVED {component}: {message}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]

    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts"""
        return [alert for alert in self.alerts if alert.level == 'critical' and not alert.resolved]

class PrometheusMetrics:
    """Prometheus metrics collection"""

    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()

        # Define metrics
        self.trades_total = Counter(
            'trading_bot_trades_total', 'Total number of trades', ['symbol', 'side']
        )
        self.trade_value = Histogram(
            'trading_bot_trade_value', 'Trade value distribution', ['symbol']
        )
        self.portfolio_value = Gauge('trading_bot_portfolio_value', 'Current portfolio value')
        self.positions_count = Gauge('trading_bot_positions_count', 'Number of open positions')
        self.api_requests = Counter(
            'trading_bot_api_requests_total', 'Total API requests', ['exchange', 'endpoint']
        )
        self.api_errors = Counter(
            'trading_bot_api_errors_total', 'Total API errors', ['exchange', 'error_type']
        )
        self.system_cpu = Gauge('trading_bot_system_cpu_percent', 'CPU usage percentage')
        self.system_memory = Gauge('trading_bot_system_memory_percent', 'Memory usage percentage')
        self.system_disk = Gauge('trading_bot_system_disk_percent', 'Disk usage percentage')

        # Start metrics server
        start_http_server(port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {port}")

    def update_trade_metrics(self, symbol: str, side: str, value: float):
        """Update trade-related metrics"""
        self.trades_total.labels(symbol=symbol, side=side).inc()
        self.trade_value.labels(symbol=symbol).observe(value)

    def update_portfolio_metrics(self, portfolio_value: float, positions_count: int):
        """Update portfolio metrics"""
        self.portfolio_value.set(portfolio_value)
        self.positions_count.set(positions_count)

    def update_api_metrics(
        self, exchange: str, endpoint: str, success: bool, error_type: Optional[str] = None
    ):
        """Update API metrics"""
        self.api_requests.labels(exchange=exchange, endpoint=endpoint).inc()
        if not success and error_type:
            self.api_errors.labels(exchange=exchange, error_type=error_type).inc()

    def update_system_metrics(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Update system metrics"""
        self.system_cpu.set(cpu_percent)
        self.system_memory.set(memory_percent)
        self.system_disk.set(disk_percent)

class MonitoringSystem:
    """Main monitoring system coordinator"""

    def __init__(self, trading_bot=None, database_path: str = "data/trading_bot.db"):
        self.trading_bot = trading_bot
        self.database_path = database_path

        # Initialize components
        self.system_monitor = SystemMonitor()
        self.database_monitor = DatabaseMonitor(database_path)
        self.trading_bot_monitor = TradingBotMonitor(trading_bot) if trading_bot else None
        self.alert_manager = AlertManager()

        # Initialize Prometheus metrics
        metrics_port = int(os.getenv('METRICS_PORT', 8000))
        self.prometheus_metrics = PrometheusMetrics(metrics_port)

        # Monitoring state
        self.running = False
        self.health_history: List[HealthStatus] = []

    async def start(self):
        """Start the monitoring system"""
        logger.info("🔍 Starting monitoring system...")
        self.running = True

        # Start monitoring loop
        await self._monitoring_loop()

    async def stop(self):
        """Stop the monitoring system"""
        logger.info("🛑 Stopping monitoring system...")
        self.running = False

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect health status from all components
                health_statuses: List[HealthStatus] = []

                # System health
                system_health = self.system_monitor.check_system_health()
                health_statuses.append(system_health)

                # Database health
                db_health = self.database_monitor.check_database_health()
                health_statuses.append(db_health)

                # Trading bot health
                if self.trading_bot_monitor:
                    bot_health = self.trading_bot_monitor.check_bot_status()
                    health_statuses.append(bot_health)

                # Process health statuses
                await self._process_health_statuses(health_statuses)

                # Update Prometheus metrics
                self._update_prometheus_metrics(health_statuses)

                # Store health history
                self.health_history.extend(health_statuses)

                # Keep only last 1000 health records
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _process_health_statuses(self, health_statuses: List[HealthStatus]):
        """Process health statuses and generate alerts"""
        for health in health_statuses:
            if health.status == 'critical':
                self.alert_manager.add_alert('critical', health.component, health.message)
            elif health.status == 'warning':
                self.alert_manager.add_alert('warning', health.component, health.message)
            else:
                # Resolve any existing alerts for this component
                self.alert_manager.resolve_alert(health.component, health.message)

    def _update_prometheus_metrics(self, health_statuses: List[HealthStatus]):
        """Update Prometheus metrics"""
        for health in health_statuses:
            if health.component == 'system':
                self.prometheus_metrics.update_system_metrics(
                    health.metrics.get('cpu_usage', 0),
                    health.metrics.get('memory_usage', 0),
                    health.metrics.get('disk_usage', 0),
                )
            elif health.component == 'trading_bot':
                self.prometheus_metrics.update_portfolio_metrics(
                    health.metrics.get('portfolio_value', 0),
                    health.metrics.get('positions_count', 0),
                )

    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = self.alert_manager.get_critical_alerts()

        return {
            'timestamp': datetime.now().isoformat(),
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'alerts': [
                {
                    'level': alert.level,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                }
                for alert in active_alerts
            ],
            'health_history_count': len(self.health_history),
            'monitoring_running': self.running,
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        system_health = self.system_monitor.check_system_health()
        db_health = self.database_monitor.check_database_health()

        return {
            'system': system_health.metrics,
            'database': db_health.metrics,
            'timestamp': datetime.now().isoformat(),
        }

# Health check endpoint for external monitoring
def create_health_check_endpoint(monitoring_system: MonitoringSystem):
    """Create a health check endpoint for external monitoring"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Trading Bot Health Check")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            summary = monitoring_system.get_health_summary()

            if summary['critical_alerts'] > 0:
                return JSONResponse(
                    status_code=503, content={"status": "unhealthy", "details": summary}
                )
            elif summary['active_alerts'] > 0:
                return JSONResponse(
                    status_code=200, content={"status": "degraded", "details": summary}
                )
            else:
                return JSONResponse(
                    status_code=200, content={"status": "healthy", "details": summary}
                )
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics"""
        try:
            metrics = monitoring_system.get_system_metrics()
            return JSONResponse(content=metrics)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

async def main():
    """Main function for testing monitoring system"""
    monitoring_system = MonitoringSystem()

    try:
        await monitoring_system.start()
    except KeyboardInterrupt:
        await monitoring_system.stop()

if __name__ == "__main__":
    asyncio.run(main())

