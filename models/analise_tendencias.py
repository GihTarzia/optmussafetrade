import pandas as pd
import ta
from typing import Dict

class AnaliseTendencias:
    def __init__(self, logger):
        self.logger = logger
        
        # Configurações para análise
        self.config = {
            'periodos_curto': 9,
            'periodos_medio': 21,
            'periodos_longo': 50,
            'min_forca_tendencia': 25,  # ADX mínimo
            'min_inclinacao': 0.001,    # Inclinação mínima
            'volume_confirmacao': 1.2    # Volume 20% acima da média
        }

    async def analisar(self, dados: pd.DataFrame) -> Dict:
        """Realiza análise completa de tendências"""
        try:
            resultado = {
                'tendencia': None,
                'forca': 0,
                'confianca': 0,
                'suporte': 0,
                'resistencia': 0,
                'detalhes': {}
            }
            
            # 1. Análise de Médias Móveis
            medias = self._analisar_medias_moveis(dados)
            
            # 2. Análise de Força (ADX/DMI)
            forca = self._analisar_forca_tendencia(dados)
            
            # 3. Análise de Estrutura de Preços
            estrutura = self._analisar_estrutura_precos(dados)
            
            # 4. Análise de Volume
            volume = self._analisar_volume(dados)
            
            # 5. Análise de Momentum
            momentum = self._analisar_momentum(dados)
            
            # Combina resultados
            resultado['tendencia'] = self._determinar_tendencia(medias, forca, estrutura)
            resultado['forca'] = self._calcular_forca_geral(medias, forca, volume, momentum)
            resultado['confianca'] = self._calcular_confianca(medias, forca, estrutura, volume)
            resultado['suporte'] = estrutura['suporte']
            resultado['resistencia'] = estrutura['resistencia']
            
            # Adiciona detalhes
            resultado['detalhes'] = {
                'medias': medias,
                'forca': forca,
                'estrutura': estrutura,
                'volume': volume,
                'momentum': momentum
            }
            
            self._log_resultado(resultado)
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Erro na análise de tendências: {str(e)}")
            return None

    def _analisar_medias_moveis(self, dados: pd.DataFrame) -> Dict:
        """Análise detalhada de médias móveis"""
        try:
            # EMAs de diferentes períodos
            ema_curta = ta.trend.EMAIndicator(dados['close'], 
                                            self.config['periodos_curto']).ema_indicator()
            ema_media = ta.trend.EMAIndicator(dados['close'], 
                                            self.config['periodos_medio']).ema_indicator()
            ema_longa = ta.trend.EMAIndicator(dados['close'], 
                                            self.config['periodos_longo']).ema_indicator()
            
            # Calcula inclinações
            inclinacao_curta = (ema_curta.iloc[-1] - ema_curta.iloc[-2]) / ema_curta.iloc[-2]
            inclinacao_media = (ema_media.iloc[-1] - ema_media.iloc[-2]) / ema_media.iloc[-2]
            
            # Alinhamento das médias
            alinhamento_alta = (ema_curta.iloc[-1] > ema_media.iloc[-1] > ema_longa.iloc[-1])
            alinhamento_baixa = (ema_curta.iloc[-1] < ema_media.iloc[-1] < ema_longa.iloc[-1])
            
            return {
                'direcao': 'ALTA' if alinhamento_alta else 'BAIXA' if alinhamento_baixa else 'LATERAL',
                'inclinacao_curta': inclinacao_curta,
                'inclinacao_media': inclinacao_media,
                'alinhamento': 'ALTA' if alinhamento_alta else 'BAIXA' if alinhamento_baixa else 'NEUTRO',
                'ema_curta': ema_curta.iloc[-1],
                'ema_media': ema_media.iloc[-1],
                'ema_longa': ema_longa.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de médias móveis: {str(e)}")
            return {}

    def _analisar_forca_tendencia(self, dados: pd.DataFrame) -> Dict:
        """Análise de força usando ADX/DMI"""
        try:
            # ADX e DMI
            adx_indicator = ta.trend.ADXIndicator(dados['high'], dados['low'], dados['close'])
            adx = adx_indicator.adx()
            di_pos = adx_indicator.adx_pos()
            di_neg = adx_indicator.adx_neg()
            
            # Valores atuais
            adx_atual = adx.iloc[-1]
            di_pos_atual = di_pos.iloc[-1]
            di_neg_atual = di_neg.iloc[-1]
            
            # Determina força e direção
            forca_tendencia = adx_atual
            direcao = 'ALTA' if di_pos_atual > di_neg_atual else 'BAIXA'
            
            return {
                'adx': adx_atual,
                'di_pos': di_pos_atual,
                'di_neg': di_neg_atual,
                'forca': forca_tendencia,
                'direcao': direcao,
                'tendencia_forte': adx_atual > self.config['min_forca_tendencia']
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de força de tendência: {str(e)}")
            return {}

    def _analisar_estrutura_precos(self, dados: pd.DataFrame) -> Dict:
        """Análise da estrutura de preços"""
        try:
            # Identifica topos e fundos
            maximas = dados['high'].rolling(5, center=True).max()
            minimas = dados['low'].rolling(5, center=True).min()
            
            # Calcula suporte e resistência
            suporte = minimas.iloc[-10:].min()
            resistencia = maximas.iloc[-10:].max()
            
            # Identifica padrão de topos e fundos
            topos_ascendentes = maximas.iloc[-20:] > maximas.iloc[-21:-1].values
            fundos_ascendentes = minimas.iloc[-20:] > minimas.iloc[-21:-1].values
            
            return {
                'suporte': suporte,
                'resistencia': resistencia,
                'topos_ascendentes': topos_ascendentes.sum() > len(topos_ascendentes) * 0.6,
                'fundos_ascendentes': fundos_ascendentes.sum() > len(fundos_ascendentes) * 0.6,
                'range': resistencia - suporte,
                'preco_atual_rel': (dados['close'].iloc[-1] - suporte) / (resistencia - suporte)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de estrutura de preços: {str(e)}")
            return {}

    def _analisar_volume(self, dados: pd.DataFrame) -> Dict:
        """Análise detalhada do volume"""
        try:
            volume = dados['volume']
            volume_medio = volume.rolling(20).mean()
            
            # Volume relativo
            volume_rel = volume.iloc[-1] / volume_medio.iloc[-1]
            
            # Tendência do volume
            tendencia_volume = volume.iloc[-5:].mean() > volume_medio.iloc[-5:].mean()
            
            return {
                'volume_relativo': volume_rel,
                'tendencia': 'ALTA' if tendencia_volume else 'BAIXA',
                'confirmacao': volume_rel > self.config['volume_confirmacao'],
                'volume_atual': volume.iloc[-1],
                'volume_medio': volume_medio.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de volume: {str(e)}")
            return {}

    def _analisar_momentum(self, dados: pd.DataFrame) -> Dict:
        """Análise de momentum"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(dados['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(dados['high'], dados['low'], dados['close'])
            k = stoch.stoch().iloc[-1]
            d = stoch.stoch_signal().iloc[-1]
            
            return {
                'rsi': rsi,
                'macd_diff': macd_line - signal_line,
                'stoch_k': k,
                'stoch_d': d,
                'momentum_positivo': rsi > 50 and macd_line > signal_line and k > d
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de momentum: {str(e)}")
            return {}

    def _determinar_tendencia(self, medias: Dict, forca: Dict, estrutura: Dict) -> str:
        """Determina a tendência geral"""
        try:
            # Pesos para cada componente
            pesos = {
                'medias': 0.4,
                'forca': 0.3,
                'estrutura': 0.3
            }
            
            # Pontuação para alta
            score_alta = 0
            if medias['direcao'] == 'ALTA': score_alta += pesos['medias']
            if forca['direcao'] == 'ALTA': score_alta += pesos['forca']
            if estrutura['topos_ascendentes'] and estrutura['fundos_ascendentes']:
                score_alta += pesos['estrutura']
            
            # Define tendência
            if score_alta > 0.7:
                return 'ALTA'
            elif score_alta < 0.3:
                return 'BAIXA'
            return 'LATERAL'
            
        except Exception as e:
            self.logger.error(f"Erro ao determinar tendência: {str(e)}")
            return 'INDEFINIDA'

    def _calcular_forca_geral(self, medias: Dict, forca: Dict, 
                             volume: Dict, momentum: Dict) -> float:
        """Calcula força geral da tendência"""
        try:
            # Pesos dos componentes
            pesos = {
                'adx': 0.3,
                'medias': 0.3,
                'volume': 0.2,
                'momentum': 0.2
            }
            
            score = 0
            
            # Componente ADX
            score += (forca['adx'] / 100) * pesos['adx']
            
            # Componente Médias
            if abs(medias['inclinacao_curta']) > self.config['min_inclinacao']:
                score += pesos['medias']
            
            # Componente Volume
            if volume['confirmacao']:
                score += pesos['volume']
            
            # Componente Momentum
            if momentum['momentum_positivo']:
                score += pesos['momentum']
            
            return min(100, score * 100)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular força geral: {str(e)}")
            return 0

    def _calcular_confianca(self, medias: Dict, forca: Dict, 
                           estrutura: Dict, volume: Dict) -> float:
        """Calcula nível de confiança da análise"""
        try:
            pontos = 0
            total_pontos = 5
            
            # Alinhamento de médias
            if medias['alinhamento'] != 'NEUTRO':
                pontos += 1
            
            # Força da tendência
            if forca['tendencia_forte']:
                pontos += 1
            
            # Estrutura de preços
            if estrutura['topos_ascendentes'] == estrutura['fundos_ascendentes']:
                pontos += 1
            
            # Volume
            if volume['confirmacao']:
                pontos += 1
            
            # Consistência geral
            if all(comp['direcao'] == medias['direcao'] for comp in [forca, volume]):
                pontos += 1
            
            return (pontos / total_pontos) * 100
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular confiança: {str(e)}")
            return 0

    def _log_resultado(self, resultado: Dict):
        """Gera log detalhado do resultado"""
        self.logger.info(f"""
        Análise de Tendências:
        Tendência: {resultado['tendencia']}
        Força: {resultado['forca']:.2f}
        Confiança: {resultado['confianca']:.2f}%
        Suporte: {resultado['suporte']:.5f}
        Resistência: {resultado['resistencia']:.5f}
        
        Detalhes:
        - Médias Móveis: {resultado['detalhes']['medias']}
        - Força (ADX): {resultado['detalhes']['forca']}
        - Volume: {resultado['detalhes']['volume']}
        - Momentum: {resultado['detalhes']['momentum']}
        """)