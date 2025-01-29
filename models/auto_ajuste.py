from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
import optuna
import ta
import pandas as pd

@dataclass
class ResultadoOtimizacao:
    parametros: Dict
    win_rate: float
    profit_factor: float
    drawdown: float
    score_final: float
    data_otimizacao: datetime
    volatilidade_media: float
    tempo_medio_operacao: int
    horarios_otimos: List[str]


class AutoAjuste:
    def __init__(self, config, db_manager, logger, metricas):
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.metricas = metricas

        # Histórico com limite de memória
        self.parametros_atuais = {}
        self.melhor_resultado = None
        
        # Configurações adaptativas melhoradas
        self.configuracoes = {
            'win_rate_minimo': 0.58,  # Aumentado
            'fator_kelly': 0.3,       # Mais conservador
            'drawdown_maximo': 0.10,  # 10% máximo
            'volatilidade_min': 0.0002,
            'volatilidade_max': 0.006,
            'tempo_min_entre_sinais': 5,  # 5 minutos
            'max_sinais_hora': 10,
            'min_operacoes_validacao': 30
        }
    
        
        # Controle de horários e períodos
        self.periodos_analise = {
            'manha': ['09:00', '12:00'],
            'tarde': ['13:00', '16:00'],
            'noite': ['17:00', '20:00']
        }
        
        # Inicializa estudos Optuna
        self.estudos = {}
        self._inicializar_otimizadores()
        
    async def ajustar_filtros(self, direcao: str):
        """Ajusta os filtros de entrada com base na direção"""
        try:
            if direcao == 'aumentar':
                # Aumenta o limite mínimo de score de entrada
                self.config.set('analise.min_score_entrada', 
                               self.config.get('analise.min_score_entrada') + 0.05)
            else:
                # Diminui o limite mínimo de score de entrada
                self.config.set('analise.min_score_entrada',
                               self.config.get('analise.min_score_entrada') - 0.05)
        except Exception as e:
            self.logger.error(f"Erro ao ajustar filtros: {str(e)}")
            
    def _inicializar_otimizadores(self):
        """Inicializa otimizadores por período"""
        for periodo in self.periodos_analise:
            self.estudos[periodo] = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner()
            )

    async def otimizar_parametros(self, dados: pd.DataFrame) -> Dict:
        """Otimiza os parâmetros de trading"""
        try:
            melhores_parametros = None
            melhor_score = -float('inf')

            # Dividir dados em treino e teste
            dados_treino, dados_teste = self._dividir_dados(dados)

            # Avaliar diferentes períodos
            for periodo in self.config['periodos']:
                parametros = self._avaliar_parametros_periodo(dados_treino, periodo)
                score = self._avaliar_parametros(dados_teste, parametros)

                if score > melhor_score:
                    melhor_score = score
                    melhores_parametros = parametros

            # Atualizar parâmetros com os melhores encontrados
            if melhores_parametros:
                self._atualizar_parametros_periodo(melhores_parametros, periodo)

            return melhores_parametros
        except Exception as e:
            self.logger.error(f"Erro ao otimizar parâmetros: {str(e)}")
            return {}

    def _avaliar_parametros_periodo(self, dados: pd.DataFrame, periodo: int) -> Dict:
        """Avalia os parâmetros para um período específico"""
        try:
            rsi = self._calcular_rsi(dados, periodo)
            medias_moveis = self._calcular_medias_moveis(dados, periodo)
            bollinger = self._calcular_bollinger(dados, periodo)
            momentum = self._calcular_momentum(dados, periodo)

            return {
                'rsi': rsi,
                'medias_moveis': medias_moveis,
                'bollinger': bollinger,
                'momentum': momentum
            }
        except Exception as e:
            self.logger.error(f"Erro ao avaliar parâmetros para o período {periodo}: {str(e)}")
            return {}
        
    def _dividir_dados(self, dados: pd.DataFrame, proporcao_treino: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide os dados em conjuntos de treino e teste"""
        try:
            tamanho_treino = int(len(dados) * proporcao_treino)
            dados_treino = dados.iloc[:tamanho_treino]
            dados_teste = dados.iloc[tamanho_treino:]
            return dados_treino, dados_teste
        except Exception as e:
            self.logger.error(f"Erro ao dividir dados: {str(e)}")
            return dados, pd.DataFrame()
    
    def _calcular_rsi(self, dados: pd.DataFrame, periodo: int) -> float:
        """Calcula o RSI para um período específico"""
        try:
            rsi = ta.momentum.RSIIndicator(dados['Close'], window=periodo).rsi()
            return rsi.iloc[-1]
        except Exception as e:
            self.logger.error(f"Erro ao calcular RSI: {str(e)}")
            return 0.0
        
    def _calcular_medias_moveis(self, dados: pd.DataFrame, periodo: int) -> float:
        """Calcula as médias móveis para um período específico"""
        try:
            media_movel = dados['Close'].rolling(window=periodo).mean()
            return media_movel.iloc[-1]
        except Exception as e:
            self.logger.error(f"Erro ao calcular médias móveis: {str(e)}")
            return 0.0
        
    def _calcular_bollinger(self, dados: pd.DataFrame, periodo: int) -> float:
        """Calcula as Bandas de Bollinger para um período específico"""
        try:
            bollinger = ta.volatility.BollingerBands(dados['Close'], window=periodo)
            banda_superior = bollinger.bollinger_hband().iloc[-1]
            banda_inferior = bollinger.bollinger_lband().iloc[-1]
            return (banda_superior + banda_inferior) / 2
        except Exception as e:
            self.logger.error(f"Erro ao calcular Bandas de Bollinger: {str(e)}")
            return 0.0
        
    def _calcular_momentum(self, dados: pd.DataFrame, periodo: int) -> float:
        """Calcula o Momentum para um período específico"""
        try:
            momentum = ta.momentum.ROCIndicator(dados['Close'], window=periodo).roc()
            return momentum.iloc[-1]
        except Exception as e:
            self.logger.error(f"Erro ao calcular Momentum: {str(e)}")
            return 0.0

    def _atualizar_parametros_periodo(self, parametros: Dict, periodo: int) -> None:
        """Atualiza os parâmetros para um período específico"""
        try:
            self.config['rsi']['periodo'] = periodo
            self.config['medias_moveis']['periodo'] = periodo
            self.config['bollinger']['periodo'] = periodo
            self.config['momentum']['periodo'] = periodo
            self.logger.info(f"Parâmetros atualizados para o período {periodo}")
        except Exception as e:
            self.logger.error(f"Erro ao atualizar parâmetros para o período {periodo}: {str(e)}")

    def _avaliar_parametros(self, dados: pd.DataFrame, parametros: Dict) -> float:
        """Avalia os parâmetros e retorna um score"""
        try:
            # Exemplo de avaliação: média dos scores dos indicadores
            score = (parametros['rsi'] + parametros['medias_moveis'] + parametros['bollinger'] + parametros['momentum']) / 4
            return score
        except Exception as e:
            self.logger.error(f"Erro ao avaliar parâmetros: {str(e)}")
            return 0.0
