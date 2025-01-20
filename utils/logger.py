import logging
from datetime import datetime
from pathlib import Path
import json
import threading
from colorama import init, Fore, Style
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Optional, List
from collections import deque

init()

class MetricsCollector:
    """Classe para coletar e analisar métricas do sistema"""
    def __init__(self, max_entries: int = 1000):
        self.metrics = {
            'operations': deque(maxlen=max_entries),
            'errors': deque(maxlen=max_entries),
            'performance': deque(maxlen=max_entries),
            'system': deque(maxlen=max_entries)
        }
        self.lock = threading.Lock()

    def add_metric(self, category: str, data: Dict):
        """Adiciona uma nova métrica"""
        with self.lock:
            if category in self.metrics:
                data['timestamp'] = datetime.now()
                self.metrics[category].append(data)

class AlertManager:
    """Gerenciador de alertas do sistema"""
    def __init__(self, max_alerts: int = 100):
        self.alerts = deque(maxlen=max_alerts)
        self.critical_alerts = deque(maxlen=max_alerts)
        self.lock = threading.Lock()
        self.alert_levels = {
            'CRITICAL': 4,
            'ERROR': 3,
            'WARNING': 2,
            'INFO': 1,
            'DEBUG': 0
        }

    def add_alert(self, level: str, message: str, data: Optional[Dict] = None):
        """Adiciona um novo alerta"""
        with self.lock:
            alert = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'data': data
            }
            self.alerts.append(alert)
            
            if level in ['CRITICAL', 'ERROR']:
                self.critical_alerts.append(alert)

    def get_active_alerts(self, min_level: str = 'WARNING') -> List[Dict]:
        """Retorna alertas ativos acima do nível especificado"""
        with self.lock:
            min_level_value = self.alert_levels.get(min_level, 0)
            return [
                alert for alert in self.alerts
                if self.alert_levels.get(alert['level'], 0) >= min_level_value
            ]

class TradingLogger:
    def __init__(self, log_dir: str = 'data', max_files: int = 5):
        # Remove todos os handlers existentes para evitar duplicação
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True) 
        
        # Configuração básica do logger
        self.logger = logging.getLogger('OpitimusSafeTrade')
        self.logger.setLevel(logging.DEBUG)
        
        # Formato personalizado com mais informações
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - '
            '%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para arquivo com rotação por tamanho
        file_handler = RotatingFileHandler(
            self.log_dir / 'trading_bot.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=max_files
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para arquivo com rotação diária
        daily_handler = TimedRotatingFileHandler(
            self.log_dir / 'trading_bot_daily.log',
            when='midnight',
            interval=1,
            backupCount=30  # Mantém 30 dias
        )
        daily_handler.setFormatter(formatter)
        self.logger.addHandler(daily_handler)
        
        # Handler para console com cores
        console_handler = ColoredConsoleHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Gerenciadores especializados
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.lock = threading.Lock()

        # Performance monitoring
        self.performance_data = {
            'start_time': datetime.now(),
            'log_counts': {level: 0 for level in logging._nameToLevel},
            'last_error': None,
            'error_count': 0
        }

    def _log_with_context(self, level: str, message: str, data: Dict = None, 
                         context: Dict = None, alert: bool = False):
        """Registra log com contexto adicional"""
        with self.lock:
            try:
                # Prepara mensagem com contexto
                full_message = message
                if context:
                    full_message += f"\nContext: {json.dumps(context, indent=2)}"
                if data:
                    full_message += f"\nData: {json.dumps(data, indent=2)}"

                # Registra no logger
                log_func = getattr(self.logger, level.lower())
                log_func(full_message)

                # Atualiza métricas
                self.performance_data['log_counts'][level.upper()] += 1
                
                # Registra alerta se necessário
                if alert or level in ['CRITICAL', 'ERROR']:
                    self.alerts.add_alert(level, message, data)

                # Coleta métricas específicas
                if level in ['CRITICAL', 'ERROR']:
                    self.metrics.add_metric('errors', {
                        'level': level,
                        'message': message,
                        'data': data
                    })
                    self.performance_data['last_error'] = datetime.now()
                    self.performance_data['error_count'] += 1

            except Exception as e:
                self.error(f"Erro ao registrar log: {str(e)}")

    def critical(self, message: str, data: Dict = None, context: Dict = None):
        """Registra erro crítico com alta visibilidade"""
        self._log_with_context('CRITICAL', message, data, context, alert=True)

    def error(self, message: str, data: Dict = None, context: Dict = None):
        """Registra erro com contexto"""
        self._log_with_context('ERROR', message, data, context, alert=True)

    def warning(self, message: str, data: Dict = None, context: Dict = None):
        """Registra aviso com dados adicionais"""
        self._log_with_context('WARNING', message, data, context)

    def info(self, message: str, data: Dict = None, context: Dict = None):
        """Registra informação com contexto"""
        self._log_with_context('INFO', message, data, context)

    def debug(self, message: str, data: Dict = None, context: Dict = None):
        """Registra mensagem de debug com dados detalhados"""
        self._log_with_context('DEBUG', message, data, context)



class ColoredConsoleHandler(logging.StreamHandler):
    """Handler personalizado para console com cores"""
    def __init__(self):
        super().__init__()
        self.level_colors = {
            'DEBUG': Fore.WHITE,
            'INFO': Fore.CYAN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }

    def emit(self, record):
        try:
            color = self.level_colors.get(record.levelname, Fore.WHITE)
            message = self.format(record)
            print(f"{color}{message}{Style.RESET_ALL}")
        except Exception:
            self.handleError(record)