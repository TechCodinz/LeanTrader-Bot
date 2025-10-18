"""
Configuration Validator for Trading Bot
Validates all configuration settings for production readiness
"""

import os
import re

@dataclass
class ValidationResult:
    """Validation result data structure"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class ConfigValidator:
    """Comprehensive configuration validator for trading bot"""

    def __init__(self):
        self.required_env_vars = [
            'BOT_NAME',
            'VERSION',
            'INITIAL_CAPITAL',
            'MAX_POSITION_SIZE',
            'STOP_LOSS_PCT',
            'TAKE_PROFIT_PCT',
            'MIN_CONFIDENCE',
        ]

        self.optional_env_vars = [
            'BYBIT_API_KEY',
            'BYBIT_SECRET_KEY',
            'COINBASE_API_KEY',
            'COINBASE_SECRET_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'EMAIL_USER',
            'EMAIL_PASSWORD',
        ]

        self.security_critical_vars = [
            'BYBIT_API_KEY',
            'BYBIT_SECRET_KEY',
            'COINBASE_API_KEY',
            'COINBASE_SECRET_KEY',
            'TELEGRAM_BOT_TOKEN',
            'EMAIL_PASSWORD',
            'TWILIO_AUTH_TOKEN',
        ]

    def validate_all(self) -> ValidationResult:
        """Validate all configuration settings"""
        errors = []
        warnings = []
        recommendations = []

        # Validate environment variables
        env_result = self.validate_environment_variables()
        errors.extend(env_result.errors)
        warnings.extend(env_result.warnings)
        recommendations.extend(env_result.recommendations)

        # Validate trading configuration
        trading_result = self.validate_trading_config()
        errors.extend(trading_result.errors)
        warnings.extend(trading_result.warnings)
        recommendations.extend(trading_result.recommendations)

        # Validate risk management
        risk_result = self.validate_risk_config()
        errors.extend(risk_result.errors)
        warnings.extend(risk_result.warnings)
        recommendations.extend(risk_result.recommendations)

        # Validate security settings
        security_result = self.validate_security_config()
        errors.extend(security_result.errors)
        warnings.extend(security_result.warnings)
        recommendations.extend(security_result.recommendations)

        # Validate API configurations
        api_result = self.validate_api_config()
        errors.extend(api_result.errors)
        warnings.extend(api_result.warnings)
        recommendations.extend(api_result.recommendations)

        # Validate database configuration
        db_result = self.validate_database_config()
        errors.extend(db_result.errors)
        warnings.extend(db_result.warnings)
        recommendations.extend(db_result.recommendations)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def validate_environment_variables(self) -> ValidationResult:
        """Validate environment variables"""
        errors = []
        warnings = []
        recommendations = []

        # Check required variables
        for var in self.required_env_vars:
            if not os.getenv(var):
                errors.append(f"Required environment variable {var} is not set")

        # Check for hardcoded sensitive values
        for var in self.security_critical_vars:
            value = os.getenv(var, '')
            if value and self._is_hardcoded_value(value):
                errors.append(f"Environment variable {var} appears to be hardcoded: {value}")

        # Check for default/placeholder values
        placeholder_values = [
            'your_api_key_here',
            'your_secret_key_here',
            'your_telegram_bot_token_here',
            'your_email@gmail.com',
            'your_password_here',
        ]

        for var in self.optional_env_vars:
            value = os.getenv(var, '')
            if value in placeholder_values:
                warnings.append(f"Environment variable {var} contains placeholder value")
                recommendations.append(f"Update {var} with actual value")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def validate_trading_config(self) -> ValidationResult:
        """Validate trading configuration"""
        errors = []
        warnings = []
        recommendations = []

        # Validate initial capital
        try:
            initial_capital = float(os.getenv('INITIAL_CAPITAL', 0))
            if initial_capital <= 0:
                errors.append("INITIAL_CAPITAL must be greater than 0")
            elif initial_capital < 1000:
                warnings.append("INITIAL_CAPITAL is very low for production trading")
                recommendations.append("Consider using at least $1000 for production")
        except ValueError:
            errors.append("INITIAL_CAPITAL must be a valid number")

        # Validate position size
        try:
            position_size = float(os.getenv('MAX_POSITION_SIZE', 0))
            if not 0 < position_size <= 1:
                errors.append("MAX_POSITION_SIZE must be between 0 and 1")
            elif position_size > 0.5:
                warnings.append("MAX_POSITION_SIZE is very high")
                recommendations.append("Consider using a smaller position size for risk management")
        except ValueError:
            errors.append("MAX_POSITION_SIZE must be a valid number")

        # Validate stop loss
        try:
            stop_loss = float(os.getenv('STOP_LOSS_PCT', 0))
            if not 0 < stop_loss <= 1:
                errors.append("STOP_LOSS_PCT must be between 0 and 1")
            elif stop_loss > 0.2:
                warnings.append("STOP_LOSS_PCT is very high")
                recommendations.append("Consider using a smaller stop loss percentage")
        except ValueError:
            errors.append("STOP_LOSS_PCT must be a valid number")

        # Validate take profit
        try:
            take_profit = float(os.getenv('TAKE_PROFIT_PCT', 0))
            if not 0 < take_profit <= 1:
                errors.append("TAKE_PROFIT_PCT must be between 0 and 1")
        except ValueError:
            errors.append("TAKE_PROFIT_PCT must be a valid number")

        # Validate confidence threshold
        try:
            confidence = float(os.getenv('MIN_CONFIDENCE', 0))
            if not 0 <= confidence <= 1:
                errors.append("MIN_CONFIDENCE must be between 0 and 1")
            elif confidence < 0.5:
                warnings.append("MIN_CONFIDENCE is very low")
                recommendations.append("Consider using a higher confidence threshold")
        except ValueError:
            errors.append("MIN_CONFIDENCE must be a valid number")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def validate_risk_config(self) -> ValidationResult:
        """Validate risk management configuration"""
        errors = []
        warnings = []
        recommendations = []

        # Validate total exposure
        try:
            exposure = float(os.getenv('MAX_TOTAL_EXPOSURE', 0))
            if not 0 < exposure <= 1:
                errors.append("MAX_TOTAL_EXPOSURE must be between 0 and 1")
            elif exposure > 0.9:
                warnings.append("MAX_TOTAL_EXPOSURE is very high")
                recommendations.append("Consider using a lower exposure limit")
        except ValueError:
            errors.append("MAX_TOTAL_EXPOSURE must be a valid number")

        # Validate max drawdown
        try:
            drawdown = float(os.getenv('MAX_DRAWDOWN', 0))
            if not 0 < drawdown <= 1:
                errors.append("MAX_DRAWDOWN must be between 0 and 1")
            elif drawdown > 0.3:
                warnings.append("MAX_DRAWDOWN is very high")
                recommendations.append("Consider using a lower drawdown limit")
        except ValueError:
            errors.append("MAX_DRAWDOWN must be a valid number")

        # Validate VaR
        try:
            var = float(os.getenv('MAX_VAR', 0))
            if not 0 < var <= 1:
                errors.append("MAX_VAR must be between 0 and 1")
        except ValueError:
            errors.append("MAX_VAR must be a valid number")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def validate_security_config(self) -> ValidationResult:
        """Validate security configuration"""
        errors = []
        warnings = []
        recommendations = []

        # Check if trading is enabled in production
        trading_enabled = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
        environment = os.getenv('ENVIRONMENT', 'development')

        if trading_enabled and environment == 'production':
            # Check if API keys are configured
            bybit_key = os.getenv('BYBIT_API_KEY', '')
            coinbase_key = os.getenv('COINBASE_API_KEY', '')

            if not bybit_key and not coinbase_key:
                errors.append("No exchange API keys configured for production trading")

            # Check if sandbox mode is disabled
            bybit_sandbox = os.getenv('BYBIT_SANDBOX', 'true').lower() == 'true'
            coinbase_sandbox = os.getenv('COINBASE_SANDBOX', 'true').lower() == 'true'

            if bybit_sandbox and bybit_key:
                warnings.append("BYBIT_SANDBOX is enabled - trading will be in test mode")
            if coinbase_sandbox and coinbase_key:
                warnings.append("COINBASE_SANDBOX is enabled - trading will be in test mode")

        # Check for weak passwords
        email_password = os.getenv('EMAIL_PASSWORD', '')
        if email_password and len(email_password) < 8:
            warnings.append("EMAIL_PASSWORD is too short")
            recommendations.append("Use a password with at least 8 characters")

        # Check for SSL configuration
        enable_ssl = os.getenv('ENABLE_SSL', 'false').lower() == 'true'
        if not enable_ssl and environment == 'production':
            warnings.append("SSL is not enabled for production")
            recommendations.append("Enable SSL for production deployment")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def validate_api_config(self) -> ValidationResult:
        """Validate API configuration"""
        errors = []
        warnings = []
        recommendations = []

        # Validate Bybit configuration
        bybit_key = os.getenv('BYBIT_API_KEY', '')
        bybit_secret = os.getenv('BYBIT_SECRET_KEY', '')

        if bybit_key and not bybit_secret:
            errors.append("BYBIT_SECRET_KEY is required when BYBIT_API_KEY is set")
        if bybit_secret and not bybit_key:
            errors.append("BYBIT_API_KEY is required when BYBIT_SECRET_KEY is set")

        # Validate Coinbase configuration
        coinbase_key = os.getenv('COINBASE_API_KEY', '')
        coinbase_secret = os.getenv('COINBASE_SECRET_KEY', '')
        coinbase_passphrase = os.getenv('COINBASE_PASSPHRASE', '')

        if coinbase_key and (not coinbase_secret or not coinbase_passphrase):
            errors.append(
                "COINBASE_SECRET_KEY and COINBASE_PASSPHRASE are required when COINBASE_API_KEY is set"
            )

        # Validate Telegram configuration
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

        if telegram_token and not telegram_chat_id:
            warnings.append("TELEGRAM_CHAT_ID is recommended when TELEGRAM_BOT_TOKEN is set")
        if telegram_chat_id and not telegram_token:
            warnings.append("TELEGRAM_BOT_TOKEN is required when TELEGRAM_CHAT_ID is set")

        # Validate email configuration
        email_user = os.getenv('EMAIL_USER', '')
        email_password = os.getenv('EMAIL_PASSWORD', '')
        email_recipient = os.getenv('EMAIL_RECIPIENT', '')

        if email_user and (not email_password or not email_recipient):
            warnings.append(
                "EMAIL_PASSWORD and EMAIL_RECIPIENT are recommended when EMAIL_USER is set"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def validate_database_config(self) -> ValidationResult:
        """Validate database configuration"""
        errors = []
        warnings = []
        recommendations = []

        # Check database URL
        db_url = os.getenv('DATABASE_URL', '')
        if not db_url:
            warnings.append("DATABASE_URL is not set - using default SQLite")
        elif not db_url.startswith(('sqlite:///', 'postgresql://', 'mysql://')):
            errors.append("DATABASE_URL must be a valid database URL")

        # Check Redis URL
        redis_url = os.getenv('REDIS_URL', '')
        if not redis_url:
            warnings.append("REDIS_URL is not set - using default Redis configuration")
        elif not redis_url.startswith('redis://'):
            errors.append("REDIS_URL must be a valid Redis URL")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _is_hardcoded_value(self, value: str) -> bool:
        """Check if a value appears to be hardcoded"""
        # Check for common hardcoded patterns
        hardcoded_patterns = [
            r'^[a-f0-9]{32}$',  # MD5 hash
            r'^[a-f0-9]{40}$',  # SHA1 hash
            r'^[a-f0-9]{64}$',  # SHA256 hash
            r'^[A-Za-z0-9+/]{40}={0,2}$',  # Base64 encoded
        ]

        for pattern in hardcoded_patterns:
            if re.match(pattern, value):
                return True

        return False

    def generate_config_report(self) -> str:
        """Generate a comprehensive configuration report"""
        result = self.validate_all()

        report = []
        report.append("=" * 60)
        report.append("TRADING BOT CONFIGURATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
        report.append(f"Errors: {len(result.errors)}")
        report.append(f"Warnings: {len(result.warnings)}")
        report.append(f"Recommendations: {len(result.recommendations)}")
        report.append("")

        if result.errors:
            report.append("🚨 ERRORS:")
            for error in result.errors:
                report.append(f"  • {error}")
            report.append("")

        if result.warnings:
            report.append("⚠️ WARNINGS:")
            for warning in result.warnings:
                report.append(f"  • {warning}")
            report.append("")

        if result.recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for recommendation in result.recommendations:
                report.append(f"  • {recommendation}")
            report.append("")

        # Environment summary
        report.append("📋 ENVIRONMENT SUMMARY:")
        for var in self.required_env_vars + self.optional_env_vars:
            value = os.getenv(var, 'NOT SET')
            if var in self.security_critical_vars and value != 'NOT SET':
                value = '***HIDDEN***'
            report.append(f"  {var}: {value}")

        report.append("=" * 60)

        return "\n".join(report)

def main():
    """Main function for configuration validation"""
    validator = ConfigValidator()
    result = validator.validate_all()

    print(validator.generate_config_report())

    if not result.is_valid:
        logger.error("Configuration validation failed")
        exit(1)
    else:
        logger.info("Configuration validation passed")

if __name__ == "__main__":
    main()
