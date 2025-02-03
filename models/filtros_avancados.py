from datetime import datetime
import pandas as pd
from typing import Dict
import ta

class FiltrosAvancados:
    def __init__(self, logger):
        self.logger = logger
        
        # Períodos do dia e suas características
        self.periodos = {
            'abertura': {
                'inicio': 8,
                'fim': 11,
                'vol_min': 0.00001,
                'vol_max': 0.004,
                'volume_min_ratio': 0.9,  # Espera mais volume na abertura
                'score_min': 0.65  # Score mínimo mais alto na abertura
            },
            'meio_dia': {
                'inicio': 11,
                'fim': 14,
                'vol_min': 0.00005,
                'vol_max': 0.003,
                'volume_min_ratio': 0.8,
                'score_min': 0.65
            },
            'fechamento': {
                'inicio': 14,
                'fim': 18,
                'vol_min': 0.0001,
                'vol_max': 0.0035,
                'volume_min_ratio': 0.9,
                'score_min': 0.65
            }
        }
        
        # Configurações por par de moedas
        self.config_pares = {
            'JPY': {
                'vol_multiplicador': 1.2,  # JPY tende a ter mais volatilidade
                'volume_min_ratio': 0.9
            },
            'GBP': {
                'vol_multiplicador': 1.1,
                'volume_min_ratio': 0.9
            }
        }

    def identificar_periodo(self, hora: int) -> dict:
        """Identifica o período do dia e suas configurações"""
        for periodo, config in self.periodos.items():
            if config['inicio'] <= hora < config['fim']:
                return {'periodo': periodo, **config}
        return None

    def ajustar_parametros_par(self, ativo: str, parametros: dict) -> dict:
        """Ajusta parâmetros baseado no par de moedas"""
        for moeda, config in self.config_pares.items():
            if moeda in ativo:
                parametros['vol_min'] *= config['vol_multiplicador']
                parametros['vol_max'] *= config['vol_multiplicador']
                parametros['volume_min_ratio'] *= config['volume_min_ratio']
        return parametros

    def analisar_filtros(self, dados: pd.DataFrame, ativo: str) -> Dict:
        """Analisa todos os filtros e retorna resultado detalhado"""
        try:
            hora_atual = datetime.now().hour
            periodo_info = self.identificar_periodo(hora_atual)
            
            if not periodo_info:
                return {
                    'valido': False,
                    'mensagem': 'Fora do horário de operação',
                    'detalhes': {'periodo': 'fora_horario'}
                }

            # Ajusta parâmetros para o par específico
            parametros = self.ajustar_parametros_par(ativo, periodo_info.copy())

            resultados = {
                'valido': True,
                'mensagem': [],
                'detalhes': {
                    'periodo': periodo_info['periodo'],
                    'parametros': parametros
                }
            }

            # Análise de Volatilidade
            volatilidade = self._calcular_volatilidade(dados)
            resultados['detalhes']['volatilidade'] = volatilidade

            if not (parametros['vol_min'] <= volatilidade <= parametros['vol_max']):
                resultados['valido'] = False
                resultados['mensagem'].append(
                    f"Volatilidade {volatilidade:.6f} fora do range para {periodo_info['periodo']}"
                )

            # Análise de Volume
            volume_ratio = self._calcular_volume_ratio(dados)
            resultados['detalhes']['volume_ratio'] = volume_ratio

            if volume_ratio < parametros['volume_min_ratio']:
                resultados['valido'] = False
                resultados['mensagem'].append(
                    f"Volume ratio {volume_ratio:.2f} abaixo do mínimo para {periodo_info['periodo']}"
                )

            # Análise de Tendência por Período
            tendencia = self._analisar_tendencia_periodo(dados)
            resultados['detalhes']['tendencia'] = tendencia

            # Log detalhado
            self.logger.info(f"""
            Análise de Filtros para {ativo}:
            Período: {periodo_info['periodo']}
            Volatilidade: {volatilidade:.6f} (min: {parametros['vol_min']:.6f}, max: {parametros['vol_max']:.6f})
            Volume Ratio: {volume_ratio:.2f} (min: {parametros['volume_min_ratio']:.2f})
            Tendência: {tendencia}
            """)

            return resultados

        except Exception as e:
            self.logger.error(f"Erro na análise de filtros: {str(e)}")
            return {'valido': False, 'mensagem': ['Erro na análise'], 'detalhes': {}}

    def _calcular_volatilidade(self, dados: pd.DataFrame) -> float:
        """Calcula volatilidade com diferentes períodos"""
        try:
            # Volatilidade de curto prazo (5 períodos)
            vol_curta = dados['close'].pct_change().rolling(5).std()
            
            # Volatilidade de médio prazo (15 períodos)
            vol_media = dados['close'].pct_change().rolling(15).std()
            
            # Média ponderada (mais peso para volatilidade recente)
            volatilidade = (vol_curta.iloc[-1] * 0.7) + (vol_media.iloc[-1] * 0.3)
            
            return volatilidade
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular volatilidade: {str(e)}")
            return 0.0

    def _calcular_volume_ratio(self, dados: pd.DataFrame) -> float:
        """Calcula ratio de volume com diferentes períodos"""
        try:
            volume_atual = dados['volume'].iloc[-1]
            volume_medio_curto = dados['volume'].rolling(5).mean().iloc[-1]
            volume_medio_longo = dados['volume'].rolling(20).mean().iloc[-1]
            
            # Média ponderada dos ratios
            ratio_curto = volume_atual / volume_medio_curto if volume_medio_curto > 0 else 0
            ratio_longo = volume_atual / volume_medio_longo if volume_medio_longo > 0 else 0
            
            return (ratio_curto * 0.6) + (ratio_longo * 0.4)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular volume ratio: {str(e)}")
            return 0.0

    def _analisar_tendencia_periodo(self, dados: pd.DataFrame) -> str:
        """Analisa tendência considerando o período do dia"""
        try:
            # EMAs de diferentes períodos
            ema_curta = ta.trend.EMAIndicator(dados['close'], 9).ema_indicator()
            ema_media = ta.trend.EMAIndicator(dados['close'], 21).ema_indicator()
            
            # Verifica direção das EMAs
            if ema_curta.iloc[-1] > ema_media.iloc[-1]:
                return "ALTA"
            elif ema_curta.iloc[-1] < ema_media.iloc[-1]:
                return "BAIXA"
            return "LATERAL"
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar tendência: {str(e)}")
            return "INDEFINIDA"