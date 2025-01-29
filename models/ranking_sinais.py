import pandas as pd
import numpy as np
import ta   

class RankingSinais:
    def __init__(self, logger):
        self.logger = logger
        
        # Configurações de pesos para diferentes aspectos
        self.pesos = {
            'ml_score': 0.25,          # Peso da previsão ML
            'tendencia': 0.20,         # Peso da análise de tendência
            'padroes': 0.20,           # Peso dos padrões técnicos
            'momentum': 0.15,          # Peso do momentum
            'volume': 0.10,            # Peso do volume
            'volatilidade': 0.10       # Peso da volatilidade
        }
        
        # Limites para classificação
        self.limites = {
            'excelente': 85,
            'muito_bom': 75,
            'bom': 65,
            'regular': 55,
            'fraco': 45
        }
        
        # Histórico de performance
        self.historico_ranking = {}

    async def calcular_ranking(self, sinal: dict, dados: pd.DataFrame) -> dict:
        """Calcula ranking completo do sinal"""
        try:
            ranking = {
                'score_final': 0,
                'scores_componentes': {},
                'classificacao': '',
                'confianca': 0,
                'detalhes': {},
                'recomendacao': ''
            }
            
            # 1. Score ML
            score_ml = self._calcular_score_ml(sinal)
            ranking['scores_componentes']['ml'] = score_ml
            
            # 2. Score Tendência
            score_tendencia = self._calcular_score_tendencia(sinal)
            ranking['scores_componentes']['tendencia'] = score_tendencia
            
            # 3. Score Padrões
            score_padroes = self._calcular_score_padroes(sinal)
            ranking['scores_componentes']['padroes'] = score_padroes
            
            # 4. Score Momentum
            score_momentum = self._calcular_score_momentum(dados)
            ranking['scores_componentes']['momentum'] = score_momentum
            
            # 5. Score Volume
            score_volume = self._calcular_score_volume(dados)
            ranking['scores_componentes']['volume'] = score_volume
            
            # 6. Score Volatilidade
            score_volatilidade = self._calcular_score_volatilidade(dados)
            ranking['scores_componentes']['volatilidade'] = score_volatilidade
            
            # Calcula score final ponderado
            ranking['score_final'] = self._calcular_score_final(ranking['scores_componentes'])
            
            # Define classificação
            ranking['classificacao'] = self._classificar_sinal(ranking['score_final'])
            
            # Calcula confiança
            ranking['confianca'] = self._calcular_confianca(ranking['scores_componentes'])
            
            # Gera recomendação
            ranking['recomendacao'] = self._gerar_recomendacao(ranking)
            
            # Adiciona detalhes
            ranking['detalhes'] = self._gerar_detalhes(ranking)
            
            # Log do ranking
            self._log_ranking(ranking, sinal)
            
            return ranking
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular ranking: {str(e)}")
            return None

    def _calcular_score_ml(self, sinal: dict) -> float:
        """Calcula score baseado na previsão ML"""
        try:
            # Combina probabilidade e assertividade
            prob = sinal.get('probabilidade', 0)
            assert_ml = sinal.get('assertividade', 0)
            
            # Normaliza para 0-100
            score = (prob * 0.7 + assert_ml * 0.3) * 100
            
            return min(100, max(0, score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score ML: {str(e)}")
            return 0

    def _calcular_score_tendencia(self, sinal: dict) -> float:
        """Calcula score baseado na análise de tendência"""
        try:
            forca = sinal.get('forca_tendencia', 0)
            confianca = sinal.get('confianca_tendencia', 0)
            
            # Normaliza para 0-100
            score = (forca * 0.6 + confianca * 0.4)
            
            return min(100, max(0, score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score tendência: {str(e)}")
            return 0

    def _calcular_score_padroes(self, sinal: dict) -> float:
        """Calcula score baseado nos padrões técnicos"""
        try:
            padroes = sinal.get('padroes', [])
            if not padroes:
                return 0
                
            # Média ponderada da força dos padrões
            score = sum(p.forca * p.confiabilidade for p in padroes) / len(padroes) * 100
            
            return min(100, max(0, score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score padrões: {str(e)}")
            return 0

    def _calcular_score_momentum(self, dados: pd.DataFrame) -> float:
        """Calcula score baseado no momentum"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(dados['close'])
            macd_diff = macd.macd_diff().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(dados['high'], dados['low'], dados['close'])
            k = stoch.stoch().iloc[-1]
            d = stoch.stoch_signal().iloc[-1]
            
            # Calcula scores individuais
            score_rsi = 100 - abs(rsi - 50) * 2
            score_macd = min(100, abs(macd_diff) * 1000)
            score_stoch = 100 - abs(k - d)
            
            # Média ponderada
            score = (score_rsi * 0.4 + score_macd * 0.3 + score_stoch * 0.3)
            
            return min(100, max(0, score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score momentum: {str(e)}")
            return 0

    def _calcular_score_volume(self, dados: pd.DataFrame) -> float:
        """Calcula score baseado no volume"""
        try:
            volume = dados['volume']
            volume_medio = volume.rolling(20).mean()
            
            # Volume relativo
            volume_ratio = volume.iloc[-1] / volume_medio.iloc[-1]
            
            # Tendência do volume
            volume_tendencia = volume.iloc[-5:].mean() / volume_medio.iloc[-5:].mean()
            
            # Calcula score
            score = min(100, (volume_ratio * 50 + volume_tendencia * 50))
            
            return max(0, score)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score volume: {str(e)}")
            return 0

    def _calcular_score_volatilidade(self, dados: pd.DataFrame) -> float:
        """Calcula score baseado na volatilidade"""
        try:
            volatilidade = dados['close'].pct_change().std()
            
            # Define faixas ideais
            vol_min = 0.0001
            vol_max = 0.003
            vol_ideal_min = 0.0003
            vol_ideal_max = 0.002
            
            # Calcula score
            if vol_min <= volatilidade <= vol_max:
                if vol_ideal_min <= volatilidade <= vol_ideal_max:
                    score = 100  # Volatilidade ideal
                else:
                    score = 75   # Volatilidade aceitável
            else:
                score = 0  # Volatilidade fora do range
            
            return score
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score volatilidade: {str(e)}")
            return 0

    def _calcular_score_final(self, scores: dict) -> float:
        """Calcula score final ponderado"""
        try:
            score_final = sum(
                scores[componente] * self.pesos[componente] 
                for componente in scores
            )
            
            return min(100, max(0, score_final))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score final: {str(e)}")
            return 0

    def _classificar_sinal(self, score: float) -> str:
        """Classifica o sinal baseado no score"""
        if score >= self.limites['excelente']:
            return 'EXCELENTE'
        elif score >= self.limites['muito_bom']:
            return 'MUITO BOM'
        elif score >= self.limites['bom']:
            return 'BOM'
        elif score >= self.limites['regular']:
            return 'REGULAR'
        elif score >= self.limites['fraco']:
            return 'FRACO'
        return 'RUIM'

    def _calcular_confianca(self, scores: dict) -> float:
        """Calcula nível de confiança baseado na consistência dos scores"""
        try:
            # Calcula desvio padrão dos scores
            valores = list(scores.values())
            std = np.std(valores)
            
            # Quanto menor o desvio, maior a confiança
            confianca = 100 - (std / 2)
            
            return min(100, max(0, confianca))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular confiança: {str(e)}")
            return 0

    def _gerar_recomendacao(self, ranking: dict) -> str:
        """Gera recomendação baseada no ranking"""
        try:
            score = ranking['score_final']
            confianca = ranking['confianca']
            
            if score >= self.limites['excelente'] and confianca >= 80:
                return "ENTRADA FORTE"
            elif score >= self.limites['muito_bom'] and confianca >= 70:
                return "ENTRADA MODERADA"
            elif score >= self.limites['bom'] and confianca >= 60:
                return "ENTRADA POSSÍVEL"
            else:
                return "AGUARDAR"
                
        except Exception as e:
            self.logger.error(f"Erro ao gerar recomendação: {str(e)}")
            return "ERRO"

    def _gerar_detalhes(self, ranking: dict) -> dict:
        """Gera detalhes do ranking para análise"""
        return {
            'scores': ranking['scores_componentes'],
            'pesos_utilizados': self.pesos,
            'limites_classificacao': self.limites
        }

    def _log_ranking(self, ranking: dict, sinal: dict):
        """Gera log detalhado do ranking"""
        self.logger.info(f"""
        Ranking do Sinal para {sinal.get('ativo')}:
        Score Final: {ranking['score_final']:.2f}
        Classificação: {ranking['classificacao']}
        Confiança: {ranking['confianca']:.2f}%
        Recomendação: {ranking['recomendacao']}
        
        Scores Componentes:
        - ML: {ranking['scores_componentes']['ml']:.2f}
        - Tendência: {ranking['scores_componentes']['tendencia']:.2f}
        - Padrões: {ranking['scores_componentes']['padroes']:.2f}
        - Momentum: {ranking['scores_componentes']['momentum']:.2f}
        - Volume: {ranking['scores_componentes']['volume']:.2f}
        - Volatilidade: {ranking['scores_componentes']['volatilidade']:.2f}
        """)