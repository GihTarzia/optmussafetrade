import yaml
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime, time
import json
from dataclasses import dataclass
import threading

@dataclass
class ConfigCache:
    timestamp: datetime
    data: Dict
    valido: bool = True

class Config:
    def __init__(self, logger, config_path: str = 'config/config.yaml'):
        self.config_path = Path(config_path)
        self.last_reload = datetime.now()
        self.lock = threading.Lock()
        self.cache = {}
        self.logger = logger
        
        # Configurações padrão essenciais
        self.DEFAULT_CONFIG = {
            'sistema': {
                'modo': 'producao',
                'debug': False,
                'auto_restart': True,
                'max_memoria': 1024
            },
            'trading': {
                'saldo_inicial': 1000,
                'risco_por_operacao': 0.01,
                'stop_diario': -0.05,
                'meta_diaria': 0.03
            }
        }
        
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
        
    def load_config(self) -> bool:
        """Carrega configurações do arquivo YAML com validação"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    
                if self._validar_configuracoes(yaml_config):
                    self._update_config(yaml_config)
                    return True
                else:
                    raise ValueError("Configurações inválidas detectadas")
            else:
                self._save_default_config()
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar configurações: {str(e)}")
            return False

    def _validar_configuracoes(self, config: Dict) -> bool:
        """Valida estrutura e valores das configurações"""
        try:
            # Verifica campos obrigatórios
            campos_obrigatorios = ['sistema', 'trading', 'analise', 'ativos']
            if not all(campo in config for campo in campos_obrigatorios):
                return False
            
            # Valida valores numéricos
            if not (0 < config['trading'].get('risco_por_operacao', 0) <= 0.05):
                return False
                
            if not (-0.2 < config['trading'].get('stop_diario', 0) < 0):
                return False
            
            # Valida horários
            #inicio = datetime.strptime(config['horarios']['inicio_operacoes'], '%H').time()
            #fim = datetime.strptime(config['horarios']['fim_operacoes'], '%H').time()
            #if inicio >= fim:
            #    return False
            
            return True
            
        except Exception:
            self.logger.error('Erro validar config')
            return False

    def _update_config(self, new_config: Dict):
        """Atualiza configurações mantendo consistência"""
        with self.lock:
            def update_dict(base: Dict, new: Dict):
                for key, value in new.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        update_dict(base[key], value)
                    else:
                        base[key] = value
            
            update_dict(self.config, new_config)
            self.last_reload = datetime.now()
            self.cache.clear()

    def get(self, path: str, default: Any = None) -> Any:
        """Obtém valor de configuração com cache"""
        try:
            # Verifica cache
            if path in self.cache:
                cache_entry = self.cache[path]
                if cache_entry.valido and (datetime.now() - cache_entry.timestamp).total_seconds() < 300:
                    return cache_entry.data
            
            # Busca valor
            value = self.config
            for key in path.split('.'):
                value = value[key]
            
            # Atualiza cache
            self.cache[path] = ConfigCache(
                timestamp=datetime.now(),
                data=value
            )
            
            return value
            
        except (KeyError, TypeError):
            self.logger.error(f'Erro get config: {path}')
            return default

    def set(self, path: str, value: Any) -> bool:
        """Define valor de configuração com validação"""
        try:
            with self.lock:
                keys = path.split('.')
                current = self.config
                
                # Navega até o penúltimo nível
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Valida novo valor
                if not self._validar_valor(keys[-1], value):
                    return False
                
                # Atualiza valor
                current[keys[-1]] = value
                
                # Invalida cache relacionado
                self._invalidar_cache_relacionado(path)
                
                # Salva alterações
                self.save_config()
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao definir configuração: {str(e)}")
            return False

    def _invalidar_cache_relacionado(self, path: str):
        """Invalida o cache relacionado a um caminho de configuração"""
        try:
            with self.lock:
                for key in self.cache.keys():
                    if key.startswith(path):
                        self.cache[key].valido = False
        except Exception as e:
            self.logger.error(f"Erro ao invalidar cache: {str(e)}")
            
    def save_config(self):
        """Salva as configurações atuais no arquivo YAML"""
        try:
            with self.lock:
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Erro ao salvar configurações: {str(e)}")


    def _validar_valor(self, key: str, value: Any) -> bool:
        """Valida valor específico baseado na chave"""
        try:
            if key.endswith('_porcentagem'):
                return 0 <= float(value) <= 1
            
            if key.endswith('_timeout'):
                return 0 < int(value) <= 3600
            
            if key.endswith('_intervalo'):
                return 0 < int(value) <= 86400
            
            return True
            
        except ValueError:
            self.logger.error('Erro validar valor config')

            return False

    def get_ativos_ativos(self) -> List[str]:
        """Retorna lista de ativos ativos com filtros"""
        try:
            ativos = self.config.get('ativos', {})
            if not ativos:
                return self._get_ativos_padrao()
            
            todos_ativos = []
            for categoria, lista in ativos.items():
                if isinstance(lista, list):
                    # Aplica filtros específicos da categoria
                    ativos_filtrados = self._filtrar_ativos(lista, categoria)
                    todos_ativos.extend(ativos_filtrados)
            
            return todos_ativos
            
        except Exception as e:
            self.logger.error(f"Erro ao obter lista de ativos: {str(e)}")
            # Retorna lista mínima em caso de erro
            return ["EURUSD=X", "GBPUSD=X"]
        
    def _filtrar_ativos(self, ativos: List[str], categoria: str) -> List[str]:
        """Filtra a lista de ativos baseado em critérios específicos"""
        try:
            ativos_filtrados = []
            for ativo in ativos:
                # Aplica filtros baseados na categoria
                if categoria == 'forex':
                    if any(par in ativo for par in ['USD', 'EUR', 'GBP', 'JPY']):
                        ativos_filtrados.append(ativo)

            return ativos_filtrados

        except Exception as e:
            self.logger.error(f"Erro ao filtrar ativos: {str(e)}")
            return []

    
    def _get_ativos_padrao(self) -> List[str]:
        """Retorna lista de ativos padrão para análise"""
        try:
            # Define uma lista padrão de ativos mais líquidos
            ativos_padrao = [
                'EURUSD=X',  # Euro
                'GBPUSD=X',  # Libra
                'USDJPY=X',  # Iene
                'AUDUSD=X'   # Dólar Australiano
            ]
            
            # Verifica se existem ativos configurados
            ativos_config = self.config.get('ativos.forex', [])
            
            return ativos_config if ativos_config else ativos_padrao
    
        except Exception as e:
            self.logger.error(f"Erro ao obter ativos padrão: {str(e)}")
            return ['EURUSD=X', 'GBPUSD=X']  # Retorna mínimo em caso de erro