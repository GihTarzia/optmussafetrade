import pandas as pd
import ta
from typing import Dict, List
from dataclasses import dataclass
import traceback

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
        """Analisa padrões nos dados"""
        try:
            self.logger.info(f"Iniciando análise de padrões com {len(dados)} registros")
            
            if not self._validar_volume(dados):
                self.logger.warning("Volume insuficiente para análise")
                return []
                
            self.logger.info("Validação de volume: OK")
            padroes = []
            
            # Análise de padrões de candlestick
            self.logger.info("Analisando padrões de candlestick...")
            padroes_candlestick = self._adicionar_padroes_candlestick(padroes, dados, self.config)  # Adicionado self.config
            if padroes_candlestick:
                padroes.extend(padroes_candlestick)
                
            return padroes
            
        except Exception as e:
            self.logger.error(f"Erro na análise de padrões: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return []   

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