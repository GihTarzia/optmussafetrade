import pandas as pd
import ta
from typing import Dict, List
from dataclasses import dataclass
import traceback
import numpy as np
@dataclass
class Padrao:
    nome: str
    forca: float  # 0 a 1
    direcao: str  # 'CALL' ou 'PUT'
    confiabilidade: float  # Histórico de acertos
    tipo: str  # Categoria do padrão (candlestick, tendência, etc)
    tempo_expiracao: int  # Tempo sugerido para expiração em minutos
    confirmacoes: int  # Novo: número de confirmações técnicas

class AnalisePadroesComplexos:
    def __init__(self, config, logger):
        self.config= config
        self.logger = logger
        self.max_sinais_hora= self.config.get('trading.controles.max_sinais_hora')      # máximo de sinais por hora (todos os ativos)
        self.min_confirmacoes= self.config.get('trading.controles.min_confirmacoes')     # reduzido para facilitar sinais iniciais

        # Novas configurações
        self.padroes_config = {
            'candles_analise': 30,  # Número de candles para análise
            'min_tamanho_padrao': 3,  # Mínimo de candles para formar padrão
            'min_forca_padrao': 0.7,  # Força mínima do padrão (0-1)
            'confirmacao_volume': 1.2,  # Volume mínimo para confirmação
            'tempo_exp_padrao': {
                'fraco': 2,    # Padrões fracos
                'medio': 3,    # Padrões médios
                'forte': 5     # Padrões fortes
            }
        }

        # Pesos dos diferentes tipos de padrões
        self.pesos_padroes = {
            'reversao': 1.0,      # Padrões de reversão
            'continuacao': 0.8,   # Padrões de continuação
            'harmonica': 1.2,     # Padrões harmônicos
            'candlestick': 0.9    # Padrões de candlestick
        }


    def _adicionar_padroes_candlestick(self, padroes: List[Padrao], dados: pd.DataFrame, config: Dict) -> List[Padrao]:
        """Análise otimizada de padrões de candlestick"""
        try:
            if len(dados) < 5:
                return padroes

            ultimo = dados.iloc[-1]
            penultimo = dados.iloc[-2]

            # Volume médio
            volume_medio = dados['volume'].rolling(20).mean().iloc[-1]
            volume_atual = dados['volume'].iloc[-1]
            volume_confirmacao = volume_atual > volume_medio * 1.2

            # Cálculos de tamanhos
            def get_candle_metrics(candle):
                corpo = abs(candle['open'] - candle['close'])
                sombra_sup = candle['high'] - max(candle['open'], candle['close'])
                sombra_inf = min(candle['open'], candle['close']) - candle['low']
                range_total = candle['high'] - candle['low']
                return corpo, sombra_sup, sombra_inf, range_total

            # Tendência de curto prazo
            ema9 = ta.trend.EMAIndicator(dados['close'], window=9).ema_indicator()
            tendencia = 'ALTA' if dados['close'].iloc[-1] > ema9.iloc[-1] else 'BAIXA'

            # RSI para confirmação
            rsi = ta.momentum.RSIIndicator(dados['close'], window=14).rsi().iloc[-1]

            # MACD para confirmação  
            macd = ta.trend.MACD(dados['close'])
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]

            # Padrões de Reversão

            # 1. Doji
            corpo_ultimo, sombra_sup_ultimo, sombra_inf_ultimo, range_ultimo = get_candle_metrics(ultimo)
            if corpo_ultimo <= range_ultimo * 0.1:
                confirmacoes = 0
                if volume_confirmacao: confirmacoes += 1
                if rsi < 30 or rsi > 70: confirmacoes += 1  
                if abs(macd_line - signal_line) < 0.0001: confirmacoes += 1

                if confirmacoes >= 2:
                    padroes.append(Padrao(
                        nome="Doji",
                        forca=0.7 * (confirmacoes/3),
                        direcao="CALL" if tendencia == "BAIXA" else "PUT",
                        confiabilidade=0.75,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 2. Hammer/Shooting Star
            if corpo_ultimo < range_ultimo * 0.3:
                if sombra_inf_ultimo > corpo_ultimo * 2 and sombra_sup_ultimo < corpo_ultimo:  # Hammer
                    confirmacoes = 0
                    if tendencia == "BAIXA": confirmacoes += 1
                    if rsi < 30: confirmacoes += 1
                    if volume_confirmacao: confirmacoes += 1
                    if macd_line > signal_line: confirmacoes += 1

                    if confirmacoes >= 2:
                        padroes.append(Padrao(
                            nome="Hammer",
                            forca=0.85 * (confirmacoes/4),
                            direcao="CALL",
                            confiabilidade=0.8,
                            tipo="candlestick",
                            tempo_expiracao=2,
                            confirmacoes=confirmacoes
                        ))

                elif sombra_sup_ultimo > corpo_ultimo * 2 and sombra_inf_ultimo < corpo_ultimo:  # Shooting Star
                    confirmacoes = 0
                    if tendencia == "ALTA": confirmacoes += 1 
                    if rsi > 70: confirmacoes += 1
                    if volume_confirmacao: confirmacoes += 1
                    if macd_line < signal_line: confirmacoes += 1

                    if confirmacoes >= 2:
                        padroes.append(Padrao(
                            nome="Shooting Star",
                            forca=0.85 * (confirmacoes/4),
                            direcao="PUT", 
                            confiabilidade=0.8,
                            tipo="candlestick",
                            tempo_expiracao=2,
                            confirmacoes=confirmacoes
                        ))

            # 3. Engulfing Patterns
            corpo_pen, sombra_sup_pen, sombra_inf_pen, range_pen = get_candle_metrics(penultimo)

            if (ultimo['open'] > penultimo['close'] and ultimo['close'] < penultimo['open'] and
                corpo_ultimo > corpo_pen):  # Bearish Engulfing
                confirmacoes = 0
                if tendencia == "ALTA": confirmacoes += 1
                if rsi > 70: confirmacoes += 1  
                if volume_confirmacao: confirmacoes += 1
                if macd_line < signal_line: confirmacoes += 1

                if confirmacoes >= 2:
                    padroes.append(Padrao(
                        nome="Bearish Engulfing",
                        forca=0.9 * (confirmacoes/4),
                        direcao="PUT",
                        confiabilidade=0.85,
                        tipo="candlestick", 
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            elif (ultimo['open'] < penultimo['close'] and ultimo['close'] > penultimo['open'] and
                  corpo_ultimo > corpo_pen):  # Bullish Engulfing
                confirmacoes = 0
                if tendencia == "BAIXA": confirmacoes += 1
                if rsi < 30: confirmacoes += 1
                if volume_confirmacao: confirmacoes += 1
                if macd_line > signal_line: confirmacoes += 1

                if confirmacoes >= 2:
                    padroes.append(Padrao(
                        nome="Bullish Engulfing",
                        forca=0.9 * (confirmacoes/4),
                        direcao="CALL",
                        confiabilidade=0.85,
                        tipo="candlestick",
                        tempo_expiracao=2,
                        confirmacoes=confirmacoes
                    ))

            # 4. Three White Soldiers / Black Crows
            if len(dados) >= 3:
                if (all(dados.iloc[-3:]['close'] > dados.iloc[-3:]['open']) and  # Three White Soldiers
                    all(dados.iloc[-3:]['close'].diff() > 0)):
                    confirmacoes = 0
                    if tendencia == "BAIXA": confirmacoes += 1
                    if volume_confirmacao: confirmacoes += 1
                    if rsi < 50: confirmacoes += 1
                    if macd_line > signal_line: confirmacoes += 1

                    if confirmacoes >= 2:
                        padroes.append(Padrao(
                            nome="Three White Soldiers",
                            forca=0.95 * (confirmacoes/4),
                            direcao="CALL",
                            confiabilidade=0.9,
                            tipo="candlestick",
                            tempo_expiracao=3,
                            confirmacoes=confirmacoes
                        ))

                elif (all(dados.iloc[-3:]['close'] < dados.iloc[-3:]['open']) and  # Three Black Crows
                      all(dados.iloc[-3:]['close'].diff() < 0)):
                    confirmacoes = 0
                    if tendencia == "ALTA": confirmacoes += 1
                    if volume_confirmacao: confirmacoes += 1
                    if rsi > 50: confirmacoes += 1
                    if macd_line < signal_line: confirmacoes += 1

                    if confirmacoes >= 2:
                        padroes.append(Padrao(
                            nome="Three Black Crows",
                            forca=0.95 * (confirmacoes/4),
                            direcao="PUT",
                            confiabilidade=0.9,
                            tipo="candlestick",
                            tempo_expiracao=3,
                            confirmacoes=confirmacoes
                        ))

            # 5. Morning/Evening Star
            if len(dados) >= 3:
                corpo1 = abs(dados.iloc[-3]['close'] - dados.iloc[-3]['open'])
                corpo2 = abs(dados.iloc[-2]['close'] - dados.iloc[-2]['open'])
                corpo3 = abs(dados.iloc[-1]['close'] - dados.iloc[-1]['open'])

                # Morning Star
                if (corpo1 > corpo2 and corpo3 > corpo2 and
                    dados.iloc[-3]['close'] < dados.iloc[-3]['open'] and
                    dados.iloc[-1]['close'] > dados.iloc[-1]['open']):
                    confirmacoes = 0
                    if tendencia == "BAIXA": confirmacoes += 1
                    if volume_confirmacao: confirmacoes += 1
                    if rsi < 30: confirmacoes += 1
                    if macd_line > signal_line: confirmacoes += 1

                    if confirmacoes >= 2:
                        padroes.append(Padrao(
                            nome="Morning Star",
                            forca=0.9 * (confirmacoes/4),
                            direcao="CALL",
                            confiabilidade=0.85,
                            tipo="candlestick",
                            tempo_expiracao=3,
                            confirmacoes=confirmacoes
                        ))

                # Evening Star
                elif (corpo1 > corpo2 and corpo3 > corpo2 and
                      dados.iloc[-3]['close'] > dados.iloc[-3]['open'] and
                      dados.iloc[-1]['close'] < dados.iloc[-1]['open']):
                    confirmacoes = 0
                    if tendencia == "ALTA": confirmacoes += 1
                    if volume_confirmacao: confirmacoes += 1
                    if rsi > 70: confirmacoes += 1
                    if macd_line < signal_line: confirmacoes += 1

                    if confirmacoes >= 2:
                        padroes.append(Padrao(
                            nome="Evening Star",
                            forca=0.9 * (confirmacoes/4),
                            direcao="PUT",
                            confiabilidade=0.85,
                            tipo="candlestick",
                            tempo_expiracao=3,
                            confirmacoes=confirmacoes
                        ))

            return padroes

        except Exception as e:
            self.logger.error(f"Erro na análise de padrões candlestick: {str(e)}")
            return padroes

    async def analisar(self, dados: pd.DataFrame) -> List[Padrao]:
        """Analisa padrões nos dados com validações melhoradas"""
        try:
            self.logger.info(f"Iniciando análise de padrões com {len(dados)} registros")

            if len(dados) < self.padroes_config['candles_analise']:
                self.logger.warning(f"Dados insuficientes: {len(dados)} candles")
                return []

            if not self._validar_volume(dados):
                self.logger.warning("Volume insuficiente para análise")
                return []

            self.logger.info("Validação de volume: OK")
            padroes = []

            # 1. Padrões de Candlestick
            self.logger.info("Analisando padrões de candlestick...")
            padroes_candlestick = self._adicionar_padroes_candlestick(padroes, dados, self.config)
            if padroes_candlestick:
                padroes.extend(padroes_candlestick)

            # 2. Padrões Harmônicos
            self.logger.info("Analisando padrões harmônicos...")
            padroes_harmonicos = self._identificar_padroes_harmonicos(dados)
            if padroes_harmonicos:
                padroes.extend(padroes_harmonicos)

            # 3. Padrões de Continuação
            self.logger.info("Analisando padrões de continuação...")
            padroes_continuacao = self._identificar_padroes_continuacao(dados)
            if padroes_continuacao:
                padroes.extend(padroes_continuacao)

            # Valida confluência entre padrões
            if padroes:
                confluencia = self._validar_confluencia_padroes(padroes)
                if not confluencia['valido']:
                    self.logger.info("Padrões sem confluência clara")
                    return []

                # Filtra apenas padrões na direção dominante
                padroes = [p for p in padroes if p.direcao == confluencia['direcao']]

                # Log detalhado dos padrões encontrados
                self._log_padroes(padroes, confluencia)

            return padroes

        except Exception as e:
            self.logger.error(f"Erro na análise de padrões: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return []

    def _log_padroes(self, padroes: List[Padrao], confluencia: Dict):
        """Gera log detalhado dos padrões encontrados"""
        self.logger.info(f"""
        Análise de Padrões Concluída:
        - Total de Padrões: {len(padroes)}
        - Direção Dominante: {confluencia['direcao']}
        - Peso Total: {confluencia['peso']:.2f}

        Padrões Encontrados:
        {chr(10).join(f'- {p.nome} ({p.tipo}): Força={p.forca:.2f}, Conf={p.confiabilidade:.2f}' for p in padroes)}
        """)

    def _validar_volume(self, dados: pd.DataFrame) -> bool:
        """Valida se o volume está adequado para operação"""
        try:
            # Padroniza o nome da coluna para minúsculo
            dados.columns = [col.lower() for col in dados.columns]
            
            volume = dados['volume']
            volume_medio = volume.rolling(20).mean()
            
            # Reduzir exigência de volume
            return volume.iloc[-1] > volume_medio.iloc[-1] * 0.5
            
        except Exception as e:
            self.logger.error(f"Erro ao validar volume: {str(e)}")
            return False
        
    def _identificar_padroes_harmonicos(self, dados: pd.DataFrame) -> List[Padrao]:
        """Identifica padrões harmônicos (Gartley, Butterfly, etc)"""
        try:
            padroes = []
            high = dados['high'].values
            low = dados['low'].values

            # Encontra pontos de swing (pivôs)
            swing_highs = self._encontrar_swing_points(high, 'high')
            swing_lows = self._encontrar_swing_points(low, 'low')

            # Analisa últimos 5 pontos de swing para padrões
            if len(swing_highs) >= 5 and len(swing_lows) >= 5:
                # Gartley Pattern
                if self._validar_gartley(swing_highs[-5:], swing_lows[-5:]):
                    padroes.append(Padrao(
                        nome="Gartley",
                        forca=0.85,
                        direcao=self._determinar_direcao_harmonica(swing_highs[-1], swing_lows[-1]),
                        confiabilidade=0.8,
                        tipo="harmonica",
                        tempo_expiracao=self.padroes_config['tempo_exp_padrao']['forte'],
                        confirmacoes=3
                    ))

                # Butterfly Pattern
                if self._validar_butterfly(swing_highs[-5:], swing_lows[-5:]):
                    padroes.append(Padrao(
                        nome="Butterfly",
                        forca=0.9,
                        direcao=self._determinar_direcao_harmonica(swing_highs[-1], swing_lows[-1]),
                        confiabilidade=0.85,
                        tipo="harmonica",
                        tempo_expiracao=self.padroes_config['tempo_exp_padrao']['forte'],
                        confirmacoes=3
                    ))

            return padroes

        except Exception as e:
            self.logger.error(f"Erro ao identificar padrões harmônicos: {str(e)}")
            return []

    def _identificar_padroes_continuacao(self, dados: pd.DataFrame) -> List[Padrao]:
        """Identifica padrões de continuação"""
        try:
            padroes = []

            # Triangulos
            if self._validar_triangulo(dados):
                direcao = self._determinar_direcao_triangulo(dados)
                padroes.append(Padrao(
                    nome="Triângulo",
                    forca=0.75,
                    direcao=direcao,
                    confiabilidade=0.7,
                    tipo="continuacao",
                    tempo_expiracao=self.padroes_config['tempo_exp_padrao']['medio'],
                    confirmacoes=2
                ))

            # Bandeiras
            if self._validar_bandeira(dados):
                direcao = self._determinar_direcao_bandeira(dados)
                padroes.append(Padrao(
                    nome="Bandeira",
                    forca=0.8,
                    direcao=direcao,
                    confiabilidade=0.75,
                    tipo="continuacao",
                    tempo_expiracao=self.padroes_config['tempo_exp_padrao']['medio'],
                    confirmacoes=2
                ))

            return padroes

        except Exception as e:
            self.logger.error(f"Erro ao identificar padrões de continuação: {str(e)}")
            return []

    def _validar_confluencia_padroes(self, padroes: List[Padrao]) -> Dict:
        """Valida confluência entre diferentes padrões encontrados"""
        try:
            if not padroes:
                return {'valido': False, 'peso': 0, 'direcao': None}

            # Conta direções
            direcoes = {'CALL': 0, 'PUT': 0}
            peso_total = 0

            for padrao in padroes:
                direcoes[padrao.direcao] += 1
                peso_total += (padrao.forca * self.pesos_padroes[padrao.tipo])

            # Determina direção dominante
            direcao_final = 'CALL' if direcoes['CALL'] > direcoes['PUT'] else 'PUT'

            # Calcula força da confluência
            total_padroes = len(padroes)
            concordancia = max(direcoes['CALL'], direcoes['PUT']) / total_padroes

            return {
                'valido': concordancia >= 0.7,  # 70% dos padrões concordam
                'peso': peso_total / total_padroes,
                'direcao': direcao_final if concordancia >= 0.7 else None
            }

        except Exception as e:
            self.logger.error(f"Erro ao validar confluência: {str(e)}")
            return {'valido': False, 'peso': 0, 'direcao': None}

    def _encontrar_swing_points(self, dados: np.array, tipo: str) -> List[float]:
        """Encontra pontos de swing (pivôs) nos dados"""
        try:
            swing_points = []
            for i in range(2, len(dados) - 2):
                if tipo == 'high':
                    if (dados[i] > dados[i-1] and 
                        dados[i] > dados[i-2] and 
                        dados[i] > dados[i+1] and 
                        dados[i] > dados[i+2]):
                        swing_points.append(dados[i])
                else:  # low
                    if (dados[i] < dados[i-1] and 
                        dados[i] < dados[i-2] and 
                        dados[i] < dados[i+1] and 
                        dados[i] < dados[i+2]):
                        swing_points.append(dados[i])

            return swing_points

        except Exception as e:
            self.logger.error(f"Erro ao encontrar swing points: {str(e)}")
            return []