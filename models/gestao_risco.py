from typing import Dict
import pandas as pd
from collections import deque

class GestaoRiscoAdaptativo:
    def __init__(self, saldo_inicial: float, logger, risco_inicial: float = 0.01):  # Reduzido para 1%
        self.saldo_inicial = saldo_inicial
        self.saldo_atual = saldo_inicial
        self.logger = logger
        self.risco_inicial = risco_inicial
        
        # Histórico de operações com limite
        self.operacoes = deque(maxlen=1000)
        
        # Parâmetros mais conservadores
        self.meta_diaria = 0.03  # Reduzido para 3%
        self.stop_diario = -0.05  # Reduzido para 5%
        
        # Métricas expandidas
        self.metricas = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'drawdown_atual': 0.0,
            'drawdown_maximo': 0.0,
            'resultado_dia': 0.0,
            'sequencia_atual': 0,
            'maior_sequencia_loss': 0,
            'valor_medio_op': 0.0,
            'volatilidade_media': 0.0,
            'tempo_medio_ops': 0.0
        }
        
        # Registro de horários
        self.analise_horarios = pd.DataFrame(columns=[
            'hora', 'win_rate', 'profit_factor', 'volume_ops', 'volatilidade_media'
        ])
            
    def get_estatisticas(self) -> Dict:
        """Retorna estatísticas detalhadas"""
        return {
            'saldo': {
                'inicial': self.saldo_inicial,
                'atual': self.saldo_atual,
                'variacao': ((self.saldo_atual / self.saldo_inicial) - 1) * 100
            },
            'operacoes': {
                'total': len(self.operacoes),
                #'hoje': self._contar_operacoes_hoje()
            },
            'metricas': self.metricas,
            'analise_horarios': self.analise_horarios.to_dict('records'),
            'limites': {
                'risco_maximo': self.saldo_atual * 0.03,
                'stop_diario_valor': self.saldo_atual * abs(self.stop_diario),
                'meta_diaria_valor': self.saldo_atual * self.meta_diaria
            }
        }