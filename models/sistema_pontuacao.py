import pandas as pd
import ta

class SistemaPontuacao:
    def __init__(self, logger):
        self.logger = logger
        
        # Pesos para diferentes componentes
        self.pesos = {
            'tendencia': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'volatilidade': 0.15,
            'indicadores_tecnicos': 0.25
        }
        
        # Limites para pontuação
        self.limites = {
            'score_minimo': 60.0,  # Pontuação mínima para validar sinal
            'score_otimo': 80.0,   # Pontuação considerada ótima
            'rsi_min': 30,
            'rsi_max': 70,
            'volume_min_ratio': 0.8,
            'volatilidade_min': 0.00007,
            'volatilidade_max': 0.004
        }

    def ajustar_limites(self, classificacao: str, direcao: str):
        """
        Ajusta os limites de classificação
        :param classificacao: Classificação que precisa ajuste
        :param direcao: 'aumentar' ou 'diminuir'
        """
        try:
            # Valor de ajuste (5%)
            ajuste = 5.0
            
            if direcao == 'aumentar':
                # Aumenta os limites para tornar mais restritivo
                if classificacao == 'EXCELENTE':
                    self.limites['excelente'] = min(95, self.limites['excelente'] + ajuste)
                elif classificacao == 'MUITO_BOM':
                    self.limites['muito_bom'] = min(85, self.limites['muito_bom'] + ajuste)
                elif classificacao == 'BOM':
                    self.limites['bom'] = min(75, self.limites['bom'] + ajuste)
                elif classificacao == 'REGULAR':
                    self.limites['regular'] = min(65, self.limites['regular'] + ajuste)
                
            else:  # diminuir
                # Diminui os limites para tornar menos restritivo
                if classificacao == 'EXCELENTE':
                    self.limites['excelente'] = max(80, self.limites['excelente'] - ajuste)
                elif classificacao == 'MUITO_BOM':
                    self.limites['muito_bom'] = max(70, self.limites['muito_bom'] - ajuste)
                elif classificacao == 'BOM':
                    self.limites['bom'] = max(60, self.limites['bom'] - ajuste)
                elif classificacao == 'REGULAR':
                    self.limites['regular'] = max(50, self.limites['regular'] - ajuste)
            
            self.logger.info(f"""
            Limites ajustados para {classificacao}:
            - Excelente: {self.limites['excelente']}
            - Muito Bom: {self.limites['muito_bom']}
            - Bom: {self.limites['bom']}
            - Regular: {self.limites['regular']}
            """)
            
        except Exception as e:
            self.logger.error(f"Erro ao ajustar limites: {str(e)}")

    def calcular_score(self, dados: pd.DataFrame, predicao: dict) -> dict:
        """Calcula score completo do sinal"""
        try:
            scores = {}
            
            # 1. Score de Tendência (25%)
            scores['tendencia'] = self._avaliar_tendencia(dados)
            
            # 2. Score de Momentum (20%)
            scores['momentum'] = self._avaliar_momentum(dados)
            
            # 3. Score de Volume (15%)
            scores['volume'] = self._avaliar_volume(dados)
            
            # 4. Score de Volatilidade (15%)
            scores['volatilidade'] = self._avaliar_volatilidade(dados)
            
            # 5. Score de Indicadores Técnicos (25%)
            scores['indicadores_tecnicos'] = self._avaliar_indicadores(dados)
            
            # Calcula score final ponderado
            score_final = sum(
                scores[componente] * self.pesos[componente] 
                for componente in scores
            )
            
            # Avalia qualidade do sinal
            qualidade = self._avaliar_qualidade(score_final)
            
            resultado = {
                'score_final': score_final,
                'scores_detalhados': scores,
                'qualidade': qualidade,
                'valido': score_final >= self.limites['score_minimo'],
                'detalhes': self._gerar_detalhes(scores, score_final)
            }
            
            # Log detalhado
            self._log_resultado(resultado)
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de score: {str(e)}")
            return {'valido': False, 'score_final': 0, 'qualidade': 'ERRO'}

    def _avaliar_tendencia(self, dados: pd.DataFrame) -> float:
        """Avalia a força e qualidade da tendência"""
        try:
            # EMAs de diferentes períodos
            ema9 = ta.trend.EMAIndicator(dados['close'], 9).ema_indicator()
            ema21 = ta.trend.EMAIndicator(dados['close'], 21).ema_indicator()
            ema50 = ta.trend.EMAIndicator(dados['close'], 50).ema_indicator()
            
            # Calcula alinhamento das médias
            tendencia_curta = ema9.iloc[-1] > ema21.iloc[-1]
            tendencia_media = ema21.iloc[-1] > ema50.iloc[-1]
            
            # ADX para força da tendência
            adx = ta.trend.ADXIndicator(dados['high'], dados['low'], dados['close'])
            forca_tendencia = adx.adx().iloc[-1] / 100.0  # Normaliza para 0-1
            
            # Calcula score
            score = 0.0
            if tendencia_curta == tendencia_media:  # Tendências alinhadas
                score += 50.0
            score += forca_tendencia * 50.0  # Adiciona força da tendência
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar tendência: {str(e)}")
            return 0.0

    def _avaliar_momentum(self, dados: pd.DataFrame) -> float:
        """Avalia o momentum do movimento"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi().iloc[-1]
            
            # ROC (Rate of Change)
            roc = ta.momentum.ROCIndicator(dados['close']).roc().iloc[-1]
            
            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(dados['close'])
            k = stoch_rsi.stochrsi_k().iloc[-1]
            
            # Calcula scores individuais
            score_rsi = 100 - abs(rsi - 50) * 2  # Centraliza em 50
            score_roc = min(100, max(0, abs(roc) * 100))  # Normaliza ROC
            score_stoch = k * 100  # Já está em 0-100
            
            # Média ponderada
            score = (score_rsi * 0.4) + (score_roc * 0.3) + (score_stoch * 0.3)
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar momentum: {str(e)}")
            return 0.0

    def _avaliar_volume(self, dados: pd.DataFrame) -> float:
        """Avalia o volume e sua tendência"""
        try:
            volume = dados['volume']
            volume_medio = volume.rolling(20).mean()
            
            # Volume ratio atual
            ratio_atual = volume.iloc[-1] / volume_medio.iloc[-1]
            
            # Tendência do volume (últimas 5 barras)
            tendencia_volume = volume.iloc[-5:].mean() / volume_medio.iloc[-1]
            
            # Consistência do volume
            consistencia = 1 - (volume.iloc[-5:].std() / volume.iloc[-5:].mean())
            
            # Calcula score
            score = 0.0
            score += min(50.0, ratio_atual * 30.0)  # Até 50 pontos pelo ratio atual
            score += min(30.0, tendencia_volume * 20.0)  # Até 30 pontos pela tendência
            score += max(0.0, min(20.0, consistencia * 20.0))  # Até 20 pontos pela consistência
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar volume: {str(e)}")
            return 0.0

    def _avaliar_volatilidade(self, dados: pd.DataFrame) -> float:
        """Avalia a volatilidade do ativo"""
        try:
            # Volatilidade atual
            volatilidade = dados['close'].pct_change().std()
            
            # Volatilidade normalizada
            vol_norm = (volatilidade - self.limites['volatilidade_min']) / (
                self.limites['volatilidade_max'] - self.limites['volatilidade_min']
            )
            
            # Penaliza volatilidade muito alta ou muito baixa
            if volatilidade < self.limites['volatilidade_min']:
                return max(20.0, vol_norm * 100.0)
            elif volatilidade > self.limites['volatilidade_max']:
                return max(20.0, (2 - vol_norm) * 100.0)
            
            return min(100.0, vol_norm * 100.0)
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar volatilidade: {str(e)}")
            return 0.0

    def _avaliar_indicadores(self, dados: pd.DataFrame) -> float:
        """Avalia conjunto de indicadores técnicos"""
        try:
            scores_ind = {}
            
            # MACD
            macd = ta.trend.MACD(dados['close'])
            macd_diff = macd.macd_diff().iloc[-1]
            scores_ind['macd'] = min(100.0, abs(macd_diff) * 1000)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(dados['close'])
            bb_pos = (dados['close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (
                bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]
            )
            scores_ind['bb'] = 100.0 - abs(bb_pos - 0.5) * 200
            
            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(dados['high'], dados['low'])
            tenkan = ichimoku.ichimoku_conversion_line().iloc[-1]
            kijun = ichimoku.ichimoku_base_line().iloc[-1]
            scores_ind['ichimoku'] = 100.0 if tenkan > kijun else 0.0
            
            # Média dos scores
            return sum(scores_ind.values()) / len(scores_ind)
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar indicadores: {str(e)}")
            return 0.0

    def _avaliar_qualidade(self, score: float) -> str:
        """Avalia qualidade do sinal baseado no score"""
        if score >= self.limites['score_otimo']:
            return 'EXCELENTE'
        elif score >= self.limites['score_minimo']:
            return 'BOM'
        return 'RUIM'

    def _gerar_detalhes(self, scores: dict, score_final: float) -> dict:
        """Gera detalhes do score para logging"""
        return {
            'componentes': scores,
            'score_final': score_final,
            'pesos_utilizados': self.pesos
        }

    def _log_resultado(self, resultado: dict):
        """Gera log detalhado do resultado"""
        self.logger.info(f"""
        Análise de Score:
        Score Final: {resultado['score_final']:.2f}
        Qualidade: {resultado['qualidade']}
        Scores Detalhados:
        - Tendência: {resultado['scores_detalhados']['tendencia']:.2f}
        - Momentum: {resultado['scores_detalhados']['momentum']:.2f}
        - Volume: {resultado['scores_detalhados']['volume']:.2f}
        - Volatilidade: {resultado['scores_detalhados']['volatilidade']:.2f}
        - Indicadores Técnicos: {resultado['scores_detalhados']['indicadores_tecnicos']:.2f}
        """)