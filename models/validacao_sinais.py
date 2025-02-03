import pandas as pd
import ta
from typing import Dict

class ValidacaoSinais:
    def __init__(self, logger):
        self.logger = logger
        
        # Configurações de validação
        self.config = {
            'rsi': {
                'sobrecomprado': 65,
                'sobrevendido': 35,
                'neutro_min': 40,
                'neutro_max': 60
            },
            'volume': {
                'min_ratio': 0.8,
                'tendencia_min': 1.0
            },
            'volatilidade': {
                'min': 0.0001,
                'max': 0.003,
                'ideal_min': 0.0002,
                'ideal_max': 0.002
            },
            'tendencia': {
                'min_forca': 25,  # ADX mínimo
                'min_inclinacao': 0.0001
            },
            'confirmacoes': {
                'minimas': 2,  # Mínimo de confirmações necessárias
                'peso_minimo': 0.5  # Peso mínimo total das confirmações
            }
        }

    async def validar_sinal(self, dados: pd.DataFrame, sinal: dict, tendencia: dict) -> dict:
        """Validação completa do sinal"""
        try:
            resultado = {
                'valido': True,
                'mensagem': [],
                'confirmacoes': [],
                'rejeicoes': [],
                'peso_total': 0,
                'detalhes': {}
            }

            # 1. Validação de Preço
            validacao_preco = self._validar_preco(dados, sinal, tendencia)
            if not validacao_preco['valido']:
                resultado['rejeicoes'].extend(validacao_preco['mensagens'])
            else:
                resultado['confirmacoes'].extend(validacao_preco['mensagens'])
                resultado['peso_total'] += validacao_preco['peso']

            # 2. Validação de Momentum
            validacao_momentum = self._validar_momentum(dados, sinal)
            if not validacao_momentum['valido']:
                resultado['rejeicoes'].extend(validacao_momentum['mensagens'])
            else:
                resultado['confirmacoes'].extend(validacao_momentum['mensagens'])
                resultado['peso_total'] += validacao_momentum['peso']

            # 3. Validação de Volume
            validacao_volume = self._validar_volume(dados, sinal)
            if not validacao_volume['valido']:
                resultado['rejeicoes'].extend(validacao_volume['mensagens'])
            else:
                resultado['confirmacoes'].extend(validacao_volume['mensagens'])
                resultado['peso_total'] += validacao_volume['peso']

            # 4. Validação de Volatilidade
            validacao_volatilidade = self._validar_volatilidade(dados, sinal)
            if not validacao_volatilidade['valido']:
                resultado['rejeicoes'].extend(validacao_volatilidade['mensagens'])
            else:
                resultado['confirmacoes'].extend(validacao_volatilidade['mensagens'])
                resultado['peso_total'] += validacao_volatilidade['peso']

            # 5. Validação de Tendência
            validacao_tendencia = self._validar_tendencia(dados, sinal, tendencia)
            if not validacao_tendencia['valido']:
                resultado['rejeicoes'].extend(validacao_tendencia['mensagens'])
            else:
                resultado['confirmacoes'].extend(validacao_tendencia['mensagens'])
                resultado['peso_total'] += validacao_tendencia['peso']

            # Avaliação final
            resultado['valido'] = (
                len(resultado['confirmacoes']) >= self.config['confirmacoes']['minimas'] and
                resultado['peso_total'] >= self.config['confirmacoes']['peso_minimo'] and
                len(resultado['rejeicoes']) == 0
            )

            # Log do resultado
            self._log_resultado(resultado, sinal)

            return resultado

        except Exception as e:
            self.logger.error(f"Erro na validação do sinal: {str(e)}")
            return {'valido': False, 'mensagem': ['Erro na validação'], 'peso_total': 0}

    def _validar_preco(self, dados: pd.DataFrame, sinal: dict, tendencia: dict) -> dict:
        """Validação detalhada do preço"""
        try:
            resultado = {
                'valido': True,
                'mensagens': [],
                'peso': 0
            }

            preco_atual = dados['close'].iloc[-1]
            
            # Verifica distância de suporte/resistência
            if sinal['direcao'] == 'CALL':
                dist_resistencia = (tendencia['resistencia'] - preco_atual) / preco_atual
                if dist_resistencia < 0.001:  # Muito próximo da resistência
                    resultado['valido'] = False
                    resultado['mensagens'].append("Preço muito próximo da resistência")
                else:
                    resultado['peso'] += 0.2
                    resultado['mensagens'].append("Distância adequada da resistência")
            else:  # PUT
                dist_suporte = (preco_atual - tendencia['suporte']) / preco_atual
                if dist_suporte < 0.001:  # Muito próximo do suporte
                    resultado['valido'] = False
                    resultado['mensagens'].append("Preço muito próximo do suporte")
                else:
                    resultado['peso'] += 0.2
                    resultado['mensagens'].append("Distância adequada do suporte")

            return resultado

        except Exception as e:
            self.logger.error(f"Erro na validação de preço: {str(e)}")
            return {'valido': False, 'mensagens': ['Erro na validação de preço'], 'peso': 0}

    def _validar_momentum(self, dados: pd.DataFrame, sinal: dict) -> dict:
        """Validação detalhada do momentum"""
        try:
            resultado = {
                'valido': True,
                'mensagens': [],
                'peso': 0
            }

            # RSI
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(dados['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            macd_hist = macd_line - signal_line

            # Stochastic
            stoch = ta.momentum.StochasticOscillator(dados['high'], dados['low'], dados['close'])
            k = stoch.stoch().iloc[-1]
            d = stoch.stoch_signal().iloc[-1]

            if sinal['direcao'] == 'CALL':
                # Validações para CALL
                if rsi > self.config['rsi']['sobrecomprado']:
                    resultado['valido'] = False
                    resultado['mensagens'].append(f"RSI sobrecomprado: {rsi:.2f}")
                elif rsi < self.config['rsi']['neutro_min']:
                    resultado['peso'] += 0.15
                    resultado['mensagens'].append(f"RSI favorável para CALL: {rsi:.2f}")

                if macd_hist > 0:
                    resultado['peso'] += 0.15
                    resultado['mensagens'].append("MACD positivo")
                else:
                    resultado['valido'] = False
                    resultado['mensagens'].append("MACD negativo")

                if k > d:
                    resultado['peso'] += 0.1
                    resultado['mensagens'].append("Stochastic favorável")

            else:  # PUT
                # Validações para PUT
                if rsi < self.config['rsi']['sobrevendido']:
                    resultado['valido'] = False
                    resultado['mensagens'].append(f"RSI sobrevendido: {rsi:.2f}")
                elif rsi > self.config['rsi']['neutro_max']:
                    resultado['peso'] += 0.15
                    resultado['mensagens'].append(f"RSI favorável para PUT: {rsi:.2f}")

                if macd_hist < 0:
                    resultado['peso'] += 0.15
                    resultado['mensagens'].append("MACD negativo")
                else:
                    resultado['valido'] = False
                    resultado['mensagens'].append("MACD positivo")

                if k < d:
                    resultado['peso'] += 0.1
                    resultado['mensagens'].append("Stochastic favorável")

            return resultado

        except Exception as e:
            self.logger.error(f"Erro na validação de momentum: {str(e)}")
            return {'valido': False, 'mensagens': ['Erro na validação de momentum'], 'peso': 0}

    def _validar_volume(self, dados: pd.DataFrame, sinal: dict) -> dict:
        """Validação detalhada do volume"""
        try:
            resultado = {
                'valido': True,
                'mensagens': [],
                'peso': 0
            }

            volume = dados['volume']
            volume_medio = volume.rolling(20).mean()
            volume_ratio = volume.iloc[-1] / volume_medio.iloc[-1]

            # Volume mínimo
            if volume_ratio < self.config['volume']['min_ratio']:
                resultado['valido'] = False
                resultado['mensagens'].append(f"Volume baixo: {volume_ratio:.2f}x média")
                return resultado

            # Tendência do volume
            volume_tendencia = volume.iloc[-5:].mean() / volume_medio.iloc[-5:]
            if volume_tendencia.mean() > self.config['volume']['tendencia_min']:
                resultado['peso'] += 0.2
                resultado['mensagens'].append("Tendência de volume positiva")
            else:
                resultado['mensagens'].append("Volume sem tendência clara")

            return resultado

        except Exception as e:
            self.logger.error(f"Erro na validação de volume: {str(e)}")
            return {'valido': False, 'mensagens': ['Erro na validação de volume'], 'peso': 0}

    def _validar_volatilidade(self, dados: pd.DataFrame, sinal: dict) -> dict:
        """Validação detalhada da volatilidade"""
        try:
            resultado = {
                'valido': True,
                'mensagens': [],
                'peso': 0
            }

            # Calcula volatilidade
            volatilidade = dados['close'].pct_change().std()

            # Verifica limites
            if volatilidade < self.config['volatilidade']['min']:
                resultado['valido'] = False
                resultado['mensagens'].append(f"Volatilidade muito baixa: {volatilidade:.6f}")
            elif volatilidade > self.config['volatilidade']['max']:
                resultado['valido'] = False
                resultado['mensagens'].append(f"Volatilidade muito alta: {volatilidade:.6f}")
            elif (self.config['volatilidade']['ideal_min'] <= volatilidade <= 
                  self.config['volatilidade']['ideal_max']):
                resultado['peso'] += 0.2
                resultado['mensagens'].append("Volatilidade ideal")
            else:
                resultado['peso'] += 0.1
                resultado['mensagens'].append("Volatilidade aceitável")

            return resultado

        except Exception as e:
            self.logger.error(f"Erro na validação de volatilidade: {str(e)}")
            return {'valido': False, 'mensagens': ['Erro na validação de volatilidade'], 'peso': 0}

    def _validar_tendencia(self, dados: pd.DataFrame, sinal: dict, tendencia: dict) -> dict:
        """Validação detalhada da tendência"""
        try:
            resultado = {
                'valido': True,
                'mensagens': [],
                'peso': 0
            }

            # Valida força da tendência
            if tendencia['forca'] < self.config['tendencia']['min_forca']:
                resultado['valido'] = False
                resultado['mensagens'].append(f"Força de tendência insuficiente: {tendencia['forca']:.2f}")
                return resultado

            # Valida alinhamento
            if sinal['direcao'] != tendencia['tendencia']:
                resultado['valido'] = False
                resultado['mensagens'].append(f"Direção não alinhada com tendência: {tendencia['tendencia']}")
                return resultado

            # Adiciona peso pela força da tendência
            if tendencia['forca'] >= 40:
                resultado['peso'] += 0.3
                resultado['mensagens'].append("Tendência forte")
            else:
                resultado['peso'] += 0.15
                resultado['mensagens'].append("Tendência moderada")

            return resultado

        except Exception as e:
            self.logger.error(f"Erro na validação de tendência: {str(e)}")
            return {'valido': False, 'mensagens': ['Erro na validação de tendência'], 'peso': 0}

    def _log_resultado(self, resultado: dict, sinal: dict):
        """Gera log detalhado do resultado da validação"""
        self.logger.info(f"""
        Validação de Sinal para {sinal['ativo']}:
        Válido: {resultado['valido']}
        Peso Total: {resultado['peso_total']:.2f}
        
        Confirmações ({len(resultado['confirmacoes'])}):
        {chr(10).join(f"- {conf}" for conf in resultado['confirmacoes'])}
        
        Rejeições ({len(resultado['rejeicoes'])}):
        {chr(10).join(f"- {rej}" for rej in resultado['rejeicoes'])}
        """)