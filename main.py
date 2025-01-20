import sys
import traceback
import pandas as pd
import numpy as np
import ta
import asyncio
import yfinance as yf
from pathlib import Path
from datetime import datetime, time

# Adiciona o diretório raiz ao PATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from tqdm import tqdm
from datetime import datetime, timedelta
from dataclasses import dataclass
from colorama import init, Fore, Style
from utils.notificador import Notificador
from typing import Dict, List, Optional
from models.ml_predictor import MLPredictor
from models.analise_padroes import AnalisePadroesComplexos
from models.gestao_risco import GestaoRiscoAdaptativo
from models.auto_ajuste import AutoAjuste
from utils.logger import TradingLogger
from utils.database import DatabaseManager
from config.parametros import Config

class Metricas:
    def __init__(self):
        self.metricas = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'drawdown': 0.0,
            'volume_operacoes': 0,
            'assertividade_media': 0.0,
            'tempo_medio_operacao': 0
        }
        self.historico_operacoes = []
        
    def atualizar(self, operacao: Dict):
        self.historico_operacoes.append(operacao)
        self._recalcular_metricas()
        
    def _recalcular_metricas(self):
        # Implementa cálculo das métricas
        pass
    
class TradingSystem:
    def __init__(self):
        self.logger = TradingLogger()
        self.logger.info(f"Iniciando Trading Bot...")
        self.db = DatabaseManager(self.logger)
        self.config = Config(self.logger)
        self.min_tempo_entre_analises= 5
        # Novos parâmetros de controle
        self.ultima_analise = {}  # Registro do momento da última análise por ativo

        # Estatísticas e histórico
        self.melhores_horarios = {}
        
        # Inicializa atributos que serão preenchidos posteriormente
        self.notificador = None
        self.ml_predictor = None
        self.analise_padroes = None
        self.gestao_risco = None
        self.auto_ajuste = None

    async def inicializar(self):
        """Inicializa componentes de forma assíncrona"""
        try:
            self.logger.debug("Iniciando inicialização dos componentes...")
            # Configura notificador
            token = self.config.get('notificacoes.telegram.token')
            chat_id = self.config.get('notificacoes.telegram.chat_id')
            self.notificador = Notificador(token, chat_id)
            self.logger.info("Notificador configurado com sucesso")

            # Inicializa componentes principais
            self.ml_predictor = MLPredictor(
                self.config,
                self.logger
            )
            self.analise_padroes = AnalisePadroesComplexos(self.config, self.logger)
            self.gestao_risco = GestaoRiscoAdaptativo(self.config.get('trading.saldo_inicial', 1000), self.logger)

            # Inicializa otimizadores
            self.logger.debug(f"\nConfigurando otimizadores...")
            self.auto_ajuste = AutoAjuste(self.config, self.db, self.logger, Metricas)
            self.logger.info("Componentes principais inicializados")

        except Exception as e:
            self.logger.critical(f"Erro na inicialização: {str(e)}")
            raise


    async def executar_backtest(self, dias: int = 30) -> Dict:
        """Executa backtesting com processamento otimizado"""
        self.logger.debug("Iniciando processo de backtesting...")
        timeout = 1800  # 30 minutos de timeout máximo

        try:
            # Carrega dados históricos
            dados = await self.db.get_dados_historicos(dias=dias)
            if dados.empty:
                self.logger.error("Sem dados suficientes para backtest")
                return {}
            
            # Adicionar verificação de dados mínimos
            if len(dados) < 20:  # Mínimo de 20 candles
                self.logger.error(f"Dados insuficientes para backtest: {len(dados)} candles")
                return {}
            
            # Agrupa dados por ativo
            dados_por_ativo = dados.groupby('ativo')
            resultados_por_ativo = {}
            # Cria tasks para processar cada ativo em paralelo
            tasks = []
            
            # Cria tasks para processar cada ativo em paralelo
            tasks = [
                asyncio.create_task(self._executar_backtest_ativo(ativo, dados_ativo))
                for ativo, dados_ativo in dados_por_ativo
            ]

            # Aguarda todas as tasks concluírem e obtém os resultados
            resultados_por_ativo = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout
            )
            # Consolida resultados
            resultados_consolidados = self._consolidar_resultados_backtest(resultados_por_ativo)

            # Exibe e salva resultados
            if len(resultados_por_ativo) > 0:
                await self._salvar_resultados_backtest(resultados_consolidados)
                self._exibir_resultados_backtest(resultados_consolidados)
                return resultados_consolidados
            else:
                raise Exception("Nenhum resultado válido obtido no backtest")

            return resultados_consolidados

        except Exception as e:
            self.logger.error(f"Erro crítico durante backtesting: {str(e)}")
            return {}

    async def monitorar_desempenho(self):
        """Monitora desempenho e ajusta parâmetros"""
        while True:
            try:
                metricas = self.gestao_risco.get_estatisticas()
                
                # Limpa dados antigos a cada 24 horas
                await self.db.limpar_dados_antigos(dias_retencao=90)
            
                # Verifica drawdown
                if metricas['metricas']['drawdown_atual'] > self.config.get('trading.max_drawdown'):
                    await self.pausar_operacoes()
                    await self.auto_ajuste.otimizar_parametros()
                
                # Verifica win rate
                #if metricas['metricas']['win_rate'] < self.config.get('trading.win_rate_minimo'):
                #    await self.auto_ajuste.ajustar_filtros('aumentar')
                
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {str(e)}")
                await asyncio.sleep(60)

    async def pausar_operacoes(self):
        """Pausa operações temporariamente"""
        self.operacoes_ativas = False
        await self.notificador.enviar_mensagem(
            "⚠️ Operações pausadas por atingir drawdown máximo"
        )
        
    def _consolidar_resultados_backtest(self, resultados_por_ativo: List[Dict]) -> Dict:
        resultados_consolidados = {
            'metricas_gerais': {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'drawdown_maximo': 0.0,
                'retorno_total': 0.0
            },
            'resultados_por_ativo': {}
        }

        for resultado in resultados_por_ativo:
            if resultado is None or 'ativo' not in resultado:
                continue
            
            ativo = resultado['ativo']
            resultados_consolidados['resultados_por_ativo'][ativo] = resultado

            resultados_consolidados['metricas_gerais']['total_trades'] += resultado['total_trades']
            resultados_consolidados['metricas_gerais']['wins'] += resultado['wins']
            resultados_consolidados['metricas_gerais']['losses'] += resultado['losses']
            resultados_consolidados['metricas_gerais']['drawdown_maximo'] = max(
                resultados_consolidados['metricas_gerais']['drawdown_maximo'],
                resultado['drawdown_maximo']
            )
            resultados_consolidados['metricas_gerais']['retorno_total'] += resultado['retorno_total']

        if resultados_consolidados['metricas_gerais']['total_trades'] > 0:
            resultados_consolidados['metricas_gerais']['win_rate'] = (
                resultados_consolidados['metricas_gerais']['wins'] / 
                resultados_consolidados['metricas_gerais']['total_trades']
            )
            resultados_consolidados['metricas_gerais']['profit_factor'] = (
                resultados_consolidados['metricas_gerais']['retorno_total'] / 
                abs(resultados_consolidados['metricas_gerais']['drawdown_maximo']) 
                if resultados_consolidados['metricas_gerais']['drawdown_maximo'] != 0 else float('inf')
            )

        return resultados_consolidados

    async def _executar_backtest_ativo(self, ativo: str, dados: pd.DataFrame) -> Dict:
        """Executa backtest para um ativo específico"""
        resultados = {
            'trades': [],
            'total_trades': 0,
            'wins': 0,
            'losses': 0
        }

        try:
            self.logger.debug(f"Iniciando backtest para {ativo}")
            
            for i in range(len(dados) - 1):
                dados_ate_momento = dados.iloc[:i+1]
                dados_futuros = dados.iloc[i+1:i+13]

                # Executa análises
                analise = await self._analisar_periodo(
                    ativo,
                    dados_ate_momento,
                    dados_futuros
                )

                if analise and analise.get('trade'):
                    trade = analise['trade']
                    resultados['trades'].append(trade)
                    resultados['total_trades'] += 1
                    
                    if trade.resultado == 'WIN':
                        resultados['wins'] += 1
                    else:
                        resultados['losses'] += 1

        except Exception as e:
            self.logger.error(f"Erro no backtest de {ativo}: {str(e)}")
            return resultados

        return resultados

    async def _analisar_periodo(self, ativo: str, dados_historicos: pd.DataFrame, 
                              dados_futuros: pd.DataFrame) -> Optional[Dict]:
        """Análise unificada de período para backtest"""
        try:
            # Análise ML
            sinal_ml = await self.ml_predictor.prever(dados_historicos, ativo)
            if not sinal_ml:
                return None

            # Análise padrões
            analise_tecnica = self.analise_padroes.analisar(
                dados=dados_historicos,
                ativo=ativo
            )
            if not analise_tecnica:
                return None

            # Validação do sinal
            if not self._validar_sinal(sinal_ml, analise_tecnica):
                return None

            # Simulação do trade
            trade = self._simular_trade(
                timestamp=dados_historicos.index[-1],
                dados_futuros=dados_futuros,
                sinal_ml=sinal_ml,
                analise_tecnica=analise_tecnica
            )

            return {'trade': trade} if trade else None

        except Exception as e:
            self.logger.error(f"Erro na análise do período: {str(e)}")
            return None


    def _exibir_resultados_backtest(self, resultados: Dict):
        """Exibe resultados do backtest"""
        self.logger.info(f"\n=== Resultados do Backtest ===")
        self.logger.info(f"Total de trades: {resultados['metricas_gerais']['total_trades']}")
        self.logger.info(f"Win Rate: {resultados['metricas_gerais']['win_rate']:.2%}")
        self.logger.info(f"Profit Factor: {resultados['metricas_gerais']['profit_factor']:.2f}")
        self.logger.info(f"Drawdown Máximo: {resultados['metricas_gerais']['drawdown_maximo']:.2f}%")
        self.logger.info(f"Retorno Total: {resultados['metricas_gerais']['retorno_total']:.2f}%")
        
        self.logger.info(f"\nMelhores Horários:")
        for hora, stats in resultados['melhores_horarios'].items():
            self.logger.info(f"• {hora}:00 - Win Rate: {stats['win_rate']:.2f}% ({stats['total_trades']} trades)")
    
    async def _salvar_resultados_backtest(self, resultados: Dict):
        """Salva resultados do backtest no banco de dados"""
        try:
            await self.db.salvar_resultados_backtest({
                'timestamp': datetime.now(),
                'metricas': resultados['metricas_gerais'],
                'melhores_horarios': resultados['melhores_horarios'],
                'evolucao_capital': resultados['evolucao_capital']
            })
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados do backtest: {str(e)}")
    
    def _validar_sinal(self, sinal_ml: Dict, analise_tecnica: Dict) -> bool:
        """Validação rigorosa de sinais para 1min"""
        try:
            
            
            
            
                    # Análise Price Action
            price_action = analise_tecnica.get('price_action', {})
            if price_action:
                preco_atual = float(sinal_ml.get('preco_atual', 0))
                if sinal_ml['direcao'] == 'CALL':
                    if preco_atual > price_action['resistencia']:
                        return False
                else:  # PUT
                    if preco_atual < price_action['suporte']:
                        return False
                
                
                
                    # Verifica indicadores técnicos
            indicadores = analise_tecnica.get('indicadores', {})
            confirmacoes = 0

            if indicadores.get('ema_cross') and sinal_ml['direcao'] == 'CALL':
                confirmacoes += 1
        
            if abs(indicadores.get('cci', 0)) > 100:
                if (indicadores['cci'] < -100 and sinal_ml['direcao'] == 'CALL') or \
                   (indicadores['cci'] > 100 and sinal_ml['direcao'] == 'PUT'):
                    confirmacoes += 1

            if indicadores.get('force_index', 0) != 0:
                if (indicadores['force_index'] > 0 and sinal_ml['direcao'] == 'CALL') or \
                   (indicadores['force_index'] < 0 and sinal_ml['direcao'] == 'PUT'):
                    confirmacoes += 1
                
                
            # Precisa de pelo menos 2 confirmações
            if confirmacoes < 2:
                return False

            # Validação de probabilidade ML mais rigorosa
            if sinal_ml['probabilidade'] < 0.75:  # Aumentado threshold
                return False
            
            
            # Adicionar validação de score mínimo
            if sinal_ml.get('score', 0) < 0.4:
                return False
            
            # Nova validação de volume
            volume_ratio = float(analise_tecnica.get('volume_ratio', 0))
            if volume_ratio < 1.2:
                return False
            
            # Nova validação de tendência
            tendencia = analise_tecnica.get('tendencia', 'NEUTRO')
            forca_tendencia = float(analise_tecnica.get('forca_tendencia', 0))
            if tendencia != 'NEUTRO' and forca_tendencia < 0.6:
                return False

            # Nova validação de momentum
            momentum_score = float(analise_tecnica.get('momentum_score', 0))
            if momentum_score < 0.55:
                return False

            # Direções devem concordar
            if sinal_ml['direcao'] != analise_tecnica['direcao']:
                return False

            # Força mínima dos padrões aumentada
            if analise_tecnica.get('forca_sinal', 0) < 0.8:
                return False

            # Validação de volatilidade mais restrita
            volatilidade = float(sinal_ml.get('volatilidade', 0))
            if not (0.001 <= volatilidade <= 0.003):  # Range mais restrito
                return False

            # Verificação de tendência
            tendencia = analise_tecnica.get('tendencia', 'NEUTRO')
            if tendencia != 'NEUTRO' and tendencia != sinal_ml['direcao']:
                return False

            # 1. Número mínimo de confirmações por tipo
            confirmacoes_por_tipo = {}
            for padrao in analise_tecnica.get('padroes', []):
                tipo = padrao.get('tipo', '')
                confirmacoes_por_tipo[tipo] = confirmacoes_por_tipo.get(tipo, 0) + 1

            # Requer pelo menos 2 tipos diferentes de confirmação
            if len(confirmacoes_por_tipo) < 2:
                return False
            
            # 2. Verifica momentum
            momentum_score = float(analise_tecnica.get('momentum_score', 0))
            if momentum_score < 0.6:  # Requer momentum mínimo
                return False

            # 3. Volume mínimo
            volume_ratio = float(analise_tecnica.get('volume_ratio', 0))
            if volume_ratio < 1.2:  # Volume pelo menos 10% acima da média
                return False

            # 4. Padrões conflitantes
            direcoes_padroes = [p.get('direcao') for p in analise_tecnica.get('padroes', [])]
            if 'CALL' in direcoes_padroes and 'PUT' in direcoes_padroes:
                return False


            return True

        except Exception as e:
            self.logger.error(f"Erro na validação do sinal: {str(e)}")
            return False

    def _simular_trade(self, timestamp: datetime, dados_futuros: pd.DataFrame, 
                      sinal_ml: Dict, analise_tecnica: Dict) -> Optional[Dict]:
        """Simula uma operação completa"""
        try:
            if dados_futuros.empty:
                return None
                
            preco_entrada = dados_futuros.iloc[0]['Open']
            tempo_exp = analise_tecnica.get('tempo_expiracao', 5)
            
            # Encontra candle de expiração
            idx_exp = min(int(tempo_exp * 12), len(dados_futuros) - 1)  # 12 candles = 1 hora
            if idx_exp < 1:
                return None
                
            preco_saida = dados_futuros.iloc[idx_exp]['Close']
            
            # Determina resultado
            if sinal_ml['direcao'] == 'CALL':
                resultado = 'WIN' if preco_saida > preco_entrada else 'LOSS'
            else:  # PUT
                resultado = 'WIN' if preco_saida < preco_entrada else 'LOSS'
                
            # Calcula lucro
            variacao = abs(preco_saida - preco_entrada) / preco_entrada
            lucro = variacao * 100 if resultado == 'WIN' else -variacao * 100
            
            return BacktestTrade(
                entrada_timestamp=timestamp,
                saida_timestamp=dados_futuros.index[idx_exp],
                ativo=sinal_ml['ativo'],
                direcao=sinal_ml['direcao'],
                preco_entrada=preco_entrada,
                preco_saida=preco_saida,
                resultado=resultado,
                lucro=lucro,
                score_entrada=sinal_ml['probabilidade'],
                assertividade_prevista=sinal_ml.get('score', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Erro na simulação: {str(e)}")
            return None

    async def _notificar_resultado(self, operacao: Dict):
        """Envia notificação de resultado"""
        mensagem = self.notificador.formatar_resultado(operacao)
        await self.notificador.enviar_mensagem(mensagem)

    def calcular_timing_entrada(self, ativo: str, sinal: Dict) -> Dict:
        """Timing otimizado para 1min"""
        try:
            agora = datetime.now()

            # Análise de horários mais restrita
            horarios_sucesso = self.db.get_horarios_sucesso(ativo)
            if not horarios_sucesso:
                return {
                    'momento_ideal': agora + timedelta(seconds=30),
                    'tempo_espera': timedelta(seconds=30),
                    'taxa_sucesso_horario': 0.5
                }

            # Encontra melhor horário no próximo minuto
            hora_atual = agora.time()
            proximos_minutos = []

            for horario, taxa in horarios_sucesso.items():
                try:
                    horario_dt = datetime.strptime(horario, "%H:%M").time()
                    if taxa >= 0.65:  # Aumentado threshold mínimo
                        proximos_minutos.append((horario_dt, taxa))
                except ValueError:
                    continue

            if proximos_minutos:
                # Ordena por taxa de sucesso
                proximos_minutos.sort(key=lambda x: x[1], reverse=True)
                melhor_horario = proximos_minutos[0][0]
                taxa_sucesso = proximos_minutos[0][1]

                # Calcula tempo até próxima entrada
                if hora_atual < melhor_horario:
                    tempo_espera = datetime.combine(agora.date(), melhor_horario) - datetime.combine(agora.date(), hora_atual)
                else:
                    tempo_espera = timedelta(seconds=30)  # Espera mínima
            else:
                tempo_espera = timedelta(seconds=30)
                taxa_sucesso = 0.5

            return {
                'momento_ideal': agora + tempo_espera,
                'tempo_espera': tempo_espera,
                'taxa_sucesso_horario': taxa_sucesso
            }

        except Exception as e:
            self.logger.error(f"Erro ao calcular timing: {str(e)}")
            return {
                'momento_ideal': agora + timedelta(seconds=30),
                'tempo_espera': timedelta(seconds=30),
                'taxa_sucesso_horario': 0.5
            }
   
    def calcular_assertividade(self, ativo: str, sinal: Dict) -> float:
        """Cálculo de assertividade otimizado para 1min"""
        try:
            # Componentes da assertividade
            prob_ml = float(sinal.get('ml_prob', 0))
            forca_padroes = float(sinal.get('score', 0))

            if 'indicadores' in sinal:
                prob_ml = float(sinal['indicadores'].get('ml_prob', prob_ml))
                forca_padroes = float(sinal['indicadores'].get('padroes_forca', forca_padroes))

            # Normalização
            prob_ml = min(1.0, max(0.0, prob_ml))
            forca_padroes = min(1.0, max(0.0, forca_padroes))

            # Histórico específico
            tempo_exp = sinal.get('tempo_expiracao', 1)  # Default 1min
            historico = float(self.db.get_assertividade_recente(
                ativo, 
                sinal['direcao'],
                tempo_expiracao=tempo_exp
            ) or 50) / 100

            # Taxa de sucesso do horário
            hora_atual = datetime.now().hour
            horarios_sucesso = self.db.get_horarios_sucesso(ativo)
            taxa_horario = horarios_sucesso.get(f"{hora_atual:02d}:00", 0.5)

            # Análise de volatilidade (mantida)
            volatilidade = float(sinal.get('volatilidade', 0))
            volatilidade_score = 1.0
            if volatilidade > 0:
                if 0.0005 <= volatilidade <= 0.002:  # Range ideal para 1min
                    volatilidade_score = 1.0
                elif 0.002 < volatilidade <= 0.003:
                    volatilidade_score = 0.8
                elif 0.003 < volatilidade <= 0.004:
                    volatilidade_score = 0.6
                else:
                    volatilidade_score = 0.4

            # Verifica tendência (mantida)
            tendencia_match = sinal.get('tendencia') == sinal.get('direcao', '')
            tendencia_score = 1.2 if tendencia_match else 0.8

            # Novos componentes
            momento_score = self._calcular_score_momento(sinal)
            tech_score = self._calcular_score_tecnico(sinal)
            historico_ponderado = self._get_historico_ponderado(
                ativo, 
                sinal['direcao'],
                tempo_exp
            )

            # Cálculo ponderado atualizado
            base_score = (
                prob_ml * 0.25 +               # ML
                forca_padroes * 0.20 +         # Padrões técnicos
                historico * 0.10 +             # Histórico simples
                volatilidade_score * 0.15 +    # Volatilidade
                historico_ponderado * 0.10 +   # Histórico ponderado (novo)
                taxa_horario * 0.10 +          # Horário
                momento_score * 0.10 +         # Momento (novo)
                tech_score * 0.10              # Score técnico (novo)
            )

            # Aplica multiplicadores
            assertividade = base_score * tendencia_score

            # Limita entre 0 e 100
            return min(100, max(0, assertividade * 100))

        except Exception as e:
            self.logger.error(f"Erro ao calcular assertividade: {str(e)}")
            return 0



    def _calcular_score_momento(self, sinal: Dict) -> float:
        """Calcula score baseado no momento atual"""
        try:
            agora = datetime.now()

            # Fatores de momento
            hora_score = self._get_hora_score(agora.hour)
            tendencia_score = self._get_tendencia_score(sinal)
            volatilidade_score = self._get_volatilidade_score(sinal)

            return (hora_score + tendencia_score + volatilidade_score) / 3

        except Exception:
            return 0.5

    def _get_historico_ponderado(self, ativo: str, direcao: str, tempo_exp: int) -> float:
        """Retorna histórico com peso maior para operações mais recentes"""
        try:
            historico = self.db.get_historico_operacoes(
                ativo, direcao, limite=20
            )

            if not historico:
                return 0.5

            pesos = [1.0 * (0.95 ** i) for i in range(len(historico))]
            soma_pesos = sum(pesos)

            win_rate_ponderado = sum(
                peso * (1.0 if op['resultado'] == 'WIN' else 0.0)
                for peso, op in zip(pesos, historico)
            ) / soma_pesos

            return win_rate_ponderado

        except Exception:
            return 0.5


    async def analisar_mercado(self):
        """Análise contínua do mercado"""
        ativos_falha = set()  # Conjunto para controlar ativos com problemas

        while True:
            try:
                # Obtém lista de ativos ativos
                ativos = self.config.get_ativos_ativos()
                
                # Remove ativos que falharam recentemente
                ativos_analise = [a for a in ativos if a not in ativos_falha]
                if not ativos_analise:
                    self.logger.warning("Nenhum ativo disponível para análise")
                    await asyncio.sleep(self.min_tempo_entre_analises * 2)
                    ativos_falha.clear()  # Limpa lista de falhas após espera
                    continue
            
                # Cria tasks apenas uma vez para cada ativo
                tasks = [
                    asyncio.create_task(self._analisar_ativo(ativo))
                    for ativo in ativos_analise
                ]

                # Executa análises em paralelo
                resultados = await asyncio.gather(*tasks, return_exceptions=True)

                # Processa resultados válidos
                for i, resultado in enumerate(resultados):
                    if isinstance(resultado, Exception):
                        self.logger.error(f"Erro ao analisar {ativos_analise[i]}: {str(resultado)}")
                        ativos_falha.add(ativos_analise[i])
                        self.logger.error(f"Stack trace completo: {traceback.format_exc()}")
                    elif resultado:
                        self.logger.info(f"Sinal gerado para {ativos_analise[i]}: {resultado}")
                        await self._processar_sinal(resultado)
                    else:
                        self.logger.warning(f"Nenhum sinal gerado para {ativos_analise[i]}")


                # Limpa ativos com falha periodicamente
                if len(ativos_falha) > 0 and len(resultados) % 10 == 0:
                    ativos_falha.clear()

                await asyncio.sleep(self.min_tempo_entre_analises)

            except Exception as e:
                self.logger.error(f"Erro no ciclo de análise: {str(e)}")
        
    async def _processar_sinal(self, sinal: Dict):
        """Processa sinal identificado"""
        try:
            
            # Valida horário novamente antes de processar o sinal
            if not self._validar_horario_operacao(datetime.now()):
                self.logger.warning("Sinal ignorado devido ao horário inadequado")
                return
        
            # Calcula melhor horário
            timing = self.calcular_timing_entrada(sinal['ativo'], sinal)

            # Calcula assertividade
            assertividade = self.calcular_assertividade(
                sinal['ativo'], 
                sinal
            )
            
            # Adiciona assertividade ao sinal
            sinal['assertividade'] = assertividade
        
            dados_mercado = await self.db.get_dados_mercado(sinal['ativo'])
            if dados_mercado.empty:
                self.logger.error(f"Não foi possível obter dados para {sinal['ativo']}")
                return None

            preco_entrada = dados_mercado['Close'].iloc[-1]
            volatilidade = dados_mercado['Close'].pct_change().std() * np.sqrt(252)

            dadosSinal = {
                'ativo': sinal['ativo'],
                'direcao': sinal['direcao'],
                'momento_entrada': timing['momento_ideal'],
                'tempo_expiracao': sinal.get('tempo_expiracao', 5),
                'score': sinal['score'],
                'assertividade': assertividade,
                'ml_prob': float(sinal['indicadores'].get('ml_prob', 0)),
                'padroes_forca': float(sinal['indicadores'].get('padroes_forca', 0)),
                'indicadores': sinal.get('indicadores', {}),
                'processado': False,
                'preco_entrada':preco_entrada,
                'volatilidade':volatilidade
                }

            sinalRepetido = await self.db.valida_sinal_repetido(dadosSinal);

            if sinalRepetido:
                return None

            # Registra sinal no banco de dados
            sinal_id = await self.db.registrar_sinal(dadosSinal)

            # NOVO: Salva análise detalhada
            if sinal_id:
                await self.db.salvar_analise_completa({
                    'id': sinal_id,
                    'ativo': sinal['ativo'],
                    'ml_prob': float(sinal['indicadores'].get('ml_prob', 0)),
                    'padroes_forca': float(sinal['indicadores'].get('padroes_forca', 0)),
                    'tendencia': sinal['indicadores'].get('tendencia', 'NEUTRO'),
                    'volatilidade': sinal['volatilidade'],
                    'momento_score': float(sinal['indicadores'].get('momento_score', 0.5)),
                    'tech_score': float(self._calcular_score_tecnico(sinal))
                })

            # Formata mensagem completa
            sinal_formatado = {
                'id': sinal_id,
                'ativo': sinal['ativo'],
                'direcao': sinal['direcao'],
                'momento_entrada': timing['momento_ideal'].strftime('%H:%M:%S'),
                'tempo_expiracao': sinal['tempo_expiracao'],
                'score': sinal['score'],
                'assertividade': assertividade,
                'indicadores': {
                    'ml_prob': float(sinal['indicadores'].get('ml_prob', 0)),
                    'padroes_forca': float(sinal['indicadores'].get('padroes_forca', 0)),
                    'tendencia': sinal['indicadores'].get('tendencia', 'NEUTRO'),
                    'volume_ratio': float(sinal['indicadores'].get('volume_ratio', 1.0)),
                    'momento_score': float(sinal['indicadores'].get('momento_score', 0.5)),
                    'tech_score': float(self._calcular_score_tecnico(sinal))  # NOVO
                },
                'padroes': sinal.get('padroes', []),  # Inclui lista completa de padrões
                'preco_entrada': preco_entrada,
                'volatilidade': volatilidade,
            }

            # Notifica via telegram
            mensagem = self.notificador.formatar_sinal(sinal_formatado)
            await self.notificador.enviar_mensagem(mensagem)

        except Exception as e:
            self.logger.error(f"Erro ao processar sinal: {str(e)}")

    async def _analisar_ativo(self, ativo: str) -> Optional[Dict]:
        """Analisa um único ativo de forma assíncrona"""
        try:
            self.logger.debug(f"Iniciando análise de {ativo}")
            
            # Valida horário atual
            agora = datetime.now()
            if not self._validar_horario_operacao(agora):
                self.logger.warning(f"Horário não apropriado para análise de {ativo}")
                return None
            
            
            # Obtém dados do mercado de forma assíncrona
            dados_mercado = await self.db.get_dados_mercado(ativo)
            if dados_mercado is None or dados_mercado.empty:
                self.logger.warning(f"Sem dados para {ativo}")
                return None 

            # Análises em paralelo
            analises = await asyncio.gather(
                self.ml_predictor.prever(dados_mercado, ativo),
                self.analise_padroes.analisar(dados_mercado, ativo=ativo)
            )
            
            sinal_ml, analise_tecnica = analises
            
            if not sinal_ml or not analise_tecnica:
                return None
                
            # Combina análises
            sinal_combinado = self._combinar_analises(
                ativo, sinal_ml, analise_tecnica, dados_mercado
            )
            if not sinal_combinado:
                return None
            
            if sinal_combinado:
                self.ultima_analise[ativo] = datetime.now()
                
            # Retorna o sinal combinado no formato esperado por _processar_sinal
            return {
                'ativo': ativo,
                'direcao': sinal_combinado['direcao'],
                'score': sinal_combinado['score'],
                'tempo_expiracao': sinal_combinado['tempo_expiracao'],
                'indicadores': {
                    'ml_prob': sinal_combinado['ml_prob'],
                    'padroes_forca': sinal_combinado['padroes_forca'],
                    'tendencia': sinal_combinado['tendencia'],
                    'volatilidade': sinal_combinado['volatilidade'],
                    'score': sinal_combinado['score']
                },
                'volatilidade': sinal_combinado['volatilidade']
            }
            
        except Exception as e:
            self.logger.error(f"1Erro na análise de {ativo}: {str(e)}")
            return None

    def _combinar_analises(self, ativo: str, sinal_ml: Dict, analise_padroes: Dict, dados_mercado: pd.DataFrame) -> Dict:
        """Combina análises ML e técnica"""
        try:
            self.logger.debug(f"\nCombinando análises para {ativo}...")
            self.logger.info(f"Sinal ML: {sinal_ml}")
            self.logger.info(f"Análise Técnica: {analise_padroes}")            
            # Verifica dados de entrada
            if not all([sinal_ml, analise_padroes]):
                self.logger.warning(f"Dados insuficientes para análise completa")
                return None

            if dados_mercado is None or dados_mercado.empty:
                self.logger.warning(f"Sem dados de mercado disponíveis")
                return None

            # Verifica se temos as colunas necessárias
            if 'Close' not in dados_mercado.columns:
                self.logger.warning(f"Dados de mercado inválidos. Colunas disponíveis: {dados_mercado.columns.tolist()}")
                return None
            
            # Direção predominante
            direcao_ml = sinal_ml.get('direcao')
            direcao_padroes = analise_padroes.get('direcao')
            
            if not all([direcao_ml, direcao_padroes]):
                self.logger.warning(f"Direções não definidas")
                return None
            
            # Análise de tendência
            tendencia = self._analisar_tendencia(dados_mercado)
            
            # Normaliza scores individuais
            score_ml = float(min(sinal_ml.get('probabilidade', 0), 0.85))  # Convertido para float
            score_padroes = float(min(analise_padroes.get('forca_sinal', 0), 0.80))  # Convertido para float
            score_tendencia = float(min(tendencia.get('forca', 0), 0.75))  # Convertido para float  
               
            # Calcula score de volume
            volume_score = self._calcular_score_volume(dados_mercado)   
               
            # NOVO: Bloqueio imediato se direção e tendência divergirem
            if tendencia['direcao'] != direcao_ml and tendencia['direcao'] != 'NEUTRO':
                self.logger.warning(f"Sinal descartado: divergência direção ({direcao_ml}) vs tendência ({tendencia['direcao']})")
                return None      
            
            # Calcula volatilidade no início (MOVIDO PARA CÁ)
            volatilidade = dados_mercado['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            volatilidade = float(volatilidade.iloc[-1]) if not volatilidade.empty else 0
      
            # Score base ponderado
            score_final = (
                score_ml * 0.35 +           # ML (reduzido)
                score_padroes * 0.25 +      # Padrões técnicos
                score_tendencia * 0.25 +    # Tendência (aumentado)
                volume_score * 0.15         # Volume (novo)
            )


            # Multiplicadores de confiança
            multiplicadores = {
                'concordancia_direcao': 1.2 if direcao_ml == direcao_padroes else 0.8,
                'tendencia': 1.15 if tendencia['direcao'] == direcao_ml else 0.85,
                'volatilidade': self._get_volatilidade_multiplicador(volatilidade),
                'momento_dia': self._get_momento_multiplicador(datetime.now())
            }

            # Aplica multiplicadores
            for mult in multiplicadores.values():
                score_final *= mult

            # Se chegou até aqui e a tendência for neutra, penaliza o score
            if tendencia['direcao'] == 'NEUTRO':
                score_final *= 0.8  # Penalização de 20% para tendência neutra


            # Bônus mais agressivos para concordância
            if direcao_ml == direcao_padroes:
                score_final *= 1.25  # +25% (aumentado)
            if tendencia['direcao'] == direcao_ml:
                score_final *= 0.6   # Penaliza em % quando há divergência

            # Limita score final
            score_final = min(0.95, max(0.1, score_final))

            # Calcula volatilidade corretamente
            volatilidade = dados_mercado['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            volatilidade = float(volatilidade.iloc[-1]) if not volatilidade.empty else 0

            # Determina tempo de expiração baseado na volatilidade
            tempo_expiracao = self._calcular_tempo_expiracao(volatilidade)

            resultado = {
                'ativo': ativo,
                'direcao': direcao_ml,
                'score': float(score_final),  # Garantindo que é float
                'ml_prob': float(score_ml),   # Garantindo que é float
                'padroes_forca': float(score_padroes),  # Garantindo que é float
                'tendencia': tendencia['direcao'],
                'sinais': analise_padroes.get('padroes', []),
                'tempo_expiracao': tempo_expiracao,
                'volatilidade': float(volatilidade),  # Garantindo que é float
                'indicadores': {
                    'ml_prob': float(score_ml),
                    'padroes_forca': float(score_padroes),
                    'tendencia': tendencia['direcao']
                }
            }
            self.logger.info(f"Resultado combinado: {resultado}")
            return resultado
            
        except Exception as e:
            self.logger.error(f"\nErro ao combinar análises: {str(e)}")
            return None
    
    def _validar_horario_operacao(self, timestamp: datetime) -> bool:
        try:
            hora = timestamp.hour
            minuto = timestamp.minute
            horario_atual = timestamp.time()
            
            hora_inicio = self.config.get('horarios.inicio_operacoes', 8)
            hora_fim = self.config.get('horarios.fim_operacoes', 18)
            horario_atual = timestamp.time()

            if not (time(hora_inicio, 0) <= horario_atual <= time(hora_fim, 0)):
                self.logger.warning(f"Fora do horário de operação: {horario_atual}")
                return False

            # Verifica período do dia
            is_fora_mercado = hora <= self.config.get('horarios.inicio_operacoes') and hora > self.config.get('horarios.fim_operacoes')

            # Ajusta requisitos baseado no período
            min_taxa_sucesso = self.config.get('horarios.analise_horarios.win_rate_minimo_horario', 0.60)

            if is_fora_mercado:
                return False
                #min_taxa_sucesso *= 0.9  # Reduz requisito em 10% para horários alternativos
                #self.logger.info(f"Operando em horário alternativo: {horario_atual} - Min taxa: {min_taxa_sucesso:.1%}")

            # Verifica taxa de sucesso
            taxa_sucesso = self.db.get_taxa_sucesso_horario(hora)

            if taxa_sucesso < min_taxa_sucesso:
                self.logger.warning(
                    f"Taxa de sucesso insuficiente para horário {hora}h: {taxa_sucesso:.1%} "
                    f"(mínimo: {min_taxa_sucesso:.1%})"
                )
                return False

            # Evita horários de alta volatilidade apenas em horário comercial
            if not is_fora_mercado:
                horarios_volateis = [
                    (8, 30, 9, 30),   # Abertura NY
                    (14, 30, 15, 30), # Fechamento Europa
                    (15, 45, 16, 15)  # Alta volatilidade NY
                ]

                for inicio_h, inicio_m, fim_h, fim_m in horarios_volateis:
                    inicio = time(inicio_h, inicio_m)
                    fim = time(fim_h, fim_m)
                    if inicio <= horario_atual <= fim:
                        self.logger.warning(f"Horário volátil detectado: {horario_atual}")
                        return False

            # Evita últimos 5 minutos de cada hora
            if minuto >= 55:
                self.logger.warning("Últimos 5 minutos da hora")
                return False

            self.logger.info(
                f"Horário validado: {horario_atual} "
                f"(Taxa sucesso: {taxa_sucesso:.1%}, "
                f"Min requerido: {min_taxa_sucesso:.1%})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Erro ao validar horário: {str(e)}")
            return False

    def _analisar_tendencia(self, dados: pd.DataFrame) -> Dict:
        """Analisa a tendência atual do ativo"""
        try:
            if dados is None or dados.empty:
                return {'direcao': 'NEUTRO', 'forca': 0}

            # Padroniza nome da coluna
            close_col = 'Close' if 'Close' in dados.columns else 'close'
            close = dados[close_col]

            # Médias adaptativas 
            ema3 = ta.trend.EMAIndicator(close, window=3).ema_indicator()
            ema8 = ta.trend.EMAIndicator(close, window=8).ema_indicator()
            ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()

            # ADX para força da tendência
            adx = ta.trend.ADXIndicator(
                dados['High'], 
                dados['Low'], 
                dados['Close'],
                window=14
            )

            # MACD e RSI originais
            macd = ta.trend.MACD(close).macd()
            signal_line = ta.trend.MACD(close).macd_signal()
            rsi = ta.momentum.RSIIndicator(close).rsi()

            # Analisa inclinações das médias
            inclinacao_3 = (ema3.iloc[-1] - ema3.iloc[-3]) / ema3.iloc[-3]
            inclinacao_8 = (ema8.iloc[-1] - ema8.iloc[-3]) / ema8.iloc[-3]
            inclinacao_21 = (ema21.iloc[-1] - ema21.iloc[-3]) / ema21.iloc[-3]

            # Sistema de pontos aprimorado
            pontos = 0

            # 1. Análise de inclinações (mantida do código original)
            limiar_inclinacao = 0.0005
            if inclinacao_3 > limiar_inclinacao: pontos += 2
            if inclinacao_8 > limiar_inclinacao: pontos += 1
            if inclinacao_21 > limiar_inclinacao: pontos += 1
            if inclinacao_3 < -limiar_inclinacao: pontos -= 2
            if inclinacao_8 < -limiar_inclinacao: pontos -= 1
            if inclinacao_21 < -limiar_inclinacao: pontos -= 1

            # 2. MACD (mantido do código original)
            macd_positivo = macd.iloc[-1] > 0
            macd_crescente = macd.iloc[-1] > macd.iloc[-2]
            if macd_positivo: pontos += 1
            if macd_crescente: pontos += 1
            if not macd_positivo: pontos -= 1
            if not macd_crescente: pontos -= 1

            # 3. RSI (mantido do código original)
            rsi_ultimo = rsi.iloc[-1]
            rsi_crescente = rsi.iloc[-1] > rsi.iloc[-2]
            if rsi_ultimo > 50 and rsi_crescente: pontos += 1
            if rsi_ultimo < 50 and not rsi_crescente: pontos -= 1

            # 4. Alinhamento de médias (novo)
            if ema3.iloc[-1] > ema8.iloc[-1] > ema21.iloc[-1]:
                pontos += 2
            elif ema3.iloc[-1] < ema8.iloc[-1] < ema21.iloc[-1]:
                pontos -= 2

            # 5. ADX (novo)
            adx_value = adx.adx().iloc[-1]
            if adx_value > 25:
                if adx.adx_pos().iloc[-1] > adx.adx_neg().iloc[-1]:
                    pontos += 1
                else:
                    pontos -= 1

            # 6. Volume confirma tendência (novo)
            volume = dados['Volume'] if 'Volume' in dados.columns else None
            if volume is not None:
                if volume.iloc[-1] > volume.rolling(5).mean().iloc[-1]:
                    if close.pct_change().iloc[-1] > 0:
                        pontos += 1
                    else:
                        pontos -= 1

            # Calcula força da tendência (normalizada entre 0 e 1)
            max_pontos = 12  # Ajustado para o total máximo possível
            forca = abs(pontos) / max_pontos
            forca = min(1.0, max(0.0, forca))

            # Determina direção
            if pontos >= 3:  # Threshold ajustado
                return {'direcao': 'CALL', 'forca': forca}
            elif pontos <= -3:
                return {'direcao': 'PUT', 'forca': forca}
            else:
                return {'direcao': 'NEUTRO', 'forca': forca}

        except Exception as e:
            self.logger.error(f"Erro ao analisar tendência: {str(e)}")
            self.logger.error(f"Colunas disponíveis: {dados.columns.tolist() if dados is not None else 'None'}")
            return {'direcao': 'NEUTRO', 'forca': 0}    

    def _calcular_score_volume(self, dados: pd.DataFrame) -> float:
        try:
            volume = dados['Volume']
            volume_ma = volume.rolling(5).mean()
            volume_std = volume.rolling(5).std()

            # Volume atual vs média
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1]

            # Crescimento do volume
            volume_growth = volume.pct_change().iloc[-1]

            # Score baseado no volume
            score = 0.0
            if volume_ratio > 1.2: score += 0.3
            if volume_ratio > 1.5: score += 0.2
            if volume_growth > 0: score += 0.3
            if volume.iloc[-1] > volume.iloc[-2]: score += 0.2

            return min(1.0, score)

        except Exception:
            return 0.5

    def _get_volatilidade_multiplicador(self, volatilidade: float) -> float:
       """Retorna multiplicador baseado na volatilidade"""
       if volatilidade < 0.001:
           return 0.8  # Volatilidade muito baixa
       elif 0.001 <= volatilidade <= 0.003:
           return 1.2  # Volatilidade ideal
       elif 0.003 < volatilidade <= 0.005:
           return 1.0  # Volatilidade aceitável
       else:
           return 0.7  # Volatilidade muito alta

    def _get_momento_multiplicador(self, momento: datetime) -> float:
        """Retorna multiplicador baseado no momento do dia"""
        hora = momento.hour
        minuto = momento.minute

        # Horários mais favoráveis
        if 8 <= hora <= 11 or 14 <= hora <= 16:
            return 1.2
        # Horários menos favoráveis
        elif hora < 7 or hora > 20:
            return 0.8
        # Evitar últimos minutos da hora
        elif minuto >= 55:
            return 0.85
        else:
            return 1.0
        
        
        
    def _calcular_score_tecnico(self, sinal: Dict) -> float:
        """Calcula score técnico baseado nos indicadores"""
        try:
            indicadores = sinal.get('indicadores', {})

            # Pesos para diferentes componentes
            pesos = {
                'tendencia': 0.4,
                'momentum': 0.3,
                'volume': 0.3
            }

            scores = {
                'tendencia': 0.0,
                'momentum': 0.0,
                'volume': 0.0
            }

            # Score de tendência
            if indicadores.get('tendencia') == sinal.get('direcao'):
                scores['tendencia'] = 1.0
            elif indicadores.get('tendencia') == 'NEUTRO':
                scores['tendencia'] = 0.5

            # Score de momentum
            rsi = float(indicadores.get('rsi', 50))
            if sinal['direcao'] == 'CALL' and rsi < 30:
                scores['momentum'] = 1.0
            elif sinal['direcao'] == 'PUT' and rsi > 70:
                scores['momentum'] = 1.0
            else:
                scores['momentum'] = 0.5

            # Score de volume
            volume_ratio = float(indicadores.get('volume_ratio', 1.0))
            if volume_ratio > 1.2:
                scores['volume'] = 1.0
            elif volume_ratio > 1.0:
                scores['volume'] = 0.7
            else:
                scores['volume'] = 0.5

            # Calcula score final ponderado
            score_final = sum(scores[k] * pesos[k] for k in pesos)

            return score_final

        except Exception:
            return 0.5




    async def verificar_resultados(self):
        """Verifica resultados das operações de forma otimizada"""
        try:
            while True:
                sinais_pendentes = await self.db.get_sinais_sem_resultado()
                self.logger.debug(f"Verificando {len(sinais_pendentes)} sinais pendentes")


                for sinal in sinais_pendentes:
                    try:
                        self.logger.info(f"\nProcessando sinal ID {sinal.get('id')} - {sinal.get('ativo')}")

                        # Verifica se timestamp já é datetime ou precisa converter
                        if isinstance(sinal['timestamp'], datetime):
                            momento_entrada = sinal['timestamp']
                        else:
                            momento_entrada = datetime.strptime(sinal['timestamp'], '%Y-%m-%d %H:%M:%S')
                  
                        tempo_expiracao = sinal['tempo_expiracao']
                        momento_expiracao = momento_entrada + timedelta(minutes=tempo_expiracao)

                        if datetime.now() > momento_expiracao:
                            self.logger.info(f"Sinal {sinal['id']} expirado, calculando resultado...")

                            # Busca preços
                            preco_entrada = sinal['preco_entrada']
                            preco_saida = await self.db.get_preco(sinal['ativo'], momento_expiracao)

                            if preco_entrada and preco_saida:
                                #self.logger.info(f"Preços obtidos - Entrada: {preco_entrada}, Saída: {preco_saida}")

                                # Calcula resultado
                                if sinal['direcao'] == 'CALL':
                                    resultado = 'WIN' if preco_saida > preco_entrada else 'LOSS'
                                else:  # PUT
                                    resultado = 'WIN' if preco_saida < preco_entrada else 'LOSS'

                                # Calcula lucro fixo baseado na configuração
                                payout = self.config.get('trading.payout', 0.85)  # 85% padrão
                                valor_entrada = self.config.get('trading.valor_entrada', 100)
                                
                                lucro = valor_entrada * payout if resultado == 'WIN' else -valor_entrada

                                #self.logger.info(f"Resultado calculado: {resultado} (lucro: {lucro})")

                                # Atualiza sinal no banco de dados
                                await self.db.atualizar_resultado_sinal(
                                    sinal['id'],
                                    resultado=resultado,
                                    lucro=lucro,
                                    preco_saida=preco_saida,
                                    data_processamento=datetime.now()
                                )

                                # Notifica resultado
                                await self._notificar_resultado({
                                    'ativo': sinal['ativo'],
                                    'direcao': sinal['direcao'],
                                    'resultado': resultado,
                                    #'lucro': lucro,
                                    'preco_entrada': preco_entrada,
                                    'preco_saida': preco_saida,
                                    'id': sinal['id'],
                                })

                    except Exception as e:
                        self.logger.error(f"Erro ao processar sinal {sinal['id']}: {str(e)}")

                await asyncio.sleep(10)  # Espera 10 segundos entre verificações

        except Exception as e:
            self.logger.error(f"Erro no verificador de resultados: {str(e)}")      
            
    def _calcular_tempo_expiracao(self, volatilidade: float) -> int:
        """Define tempo de expiração para opções binárias"""
        try:
            if volatilidade < 0.001:  # Volatilidade baixa
                return 5  # Mais tempo para o movimento se desenvolver
            elif volatilidade > 0.008:  # Volatilidade muito alta
                return 1   # Tempo curto para evitar reversões
            elif volatilidade > 0.006:  # Volatilidade alta
                return 2   # Tempo moderado-curto
            else:  # Volatilidade ideal
                return 3   # Tempo padrão
                
        except Exception as e:
            self.logger.error(f"Erro ao definir tempo de expiração: {str(e)}")
            return 3
   
    # TradingSystem - Correção da função baixar_dados_historicos
    async def baixar_dados_historicos(self):
        """Baixa dados históricos iniciais para todos os ativos"""
        self.logger.debug(f"Iniciando download de dados históricos...")

        ativos = self.config.get_ativos_ativos()
        dados_salvos = False
        hoje = datetime.now()
        dfs = []

        for ativo in tqdm(ativos, desc="Baixando dados"):
            try:
                # Download de 30 dias de dados em intervalos de 1 minuto
                # Divide em 4 períodos de 7 dias para obter dados de 1 minuto
                for i in range(4):
                    end = hoje - timedelta(days=i*7)
                    start = end - timedelta(days=7)

                    df = yf.download(
                        ativo,
                        start=start,
                        end=end,
                        interval="1m",
                        progress=False
                    )
                    if not df.empty:
                        dfs.append(df)

                if len(dfs) > 0:  # Verifica se temos dados
                    dados_combinados = pd.concat(dfs).sort_index()
                    dados_combinados = dados_combinados[~dados_combinados.index.duplicated(keep='first')]

                    if not dados_combinados.empty:
                        dados_combinados.columns = [col if col == 'Volume' else col.title() for col in dados_combinados.columns]
                        dados_combinados = dados_combinados[['Open', 'High', 'Low', 'Close', 'Volume']]

                        sucesso = await self.db.salvar_precos(ativo, dados_combinados)
                        if sucesso:
                            dados_salvos = True
                            self.logger.info(f"Dados salvos com sucesso para {ativo}: {len(dados_combinados)} registros")
                    else:
                        self.logger.error(f"Erro ao salvar dados para {ativo}")
                else:
                    self.logger.warning(f"Nenhum dado disponível para {ativo}")

            except Exception as e:
                self.logger.error(f"Erro ao baixar dados para {ativo}: {str(e)}")

        if dados_salvos:
            self.logger.info(f"Download de dados históricos concluído com sucesso")
            return True
        else:
            self.logger.warning(f"Nenhum dado foi salvo durante o processo")
            return False

    async def executar(self):
        """Loop principal do sistema"""
        try:
            self.logger.info(f"\nIniciando sequência de inicialização...")

            # Inicializa componentes básicos
            await self.inicializar()

            # Fase 1: Baixa dados históricos
            self.logger.info(f"\nFase 1: Baixando dados históricos...")
            sucesso_download = await self.baixar_dados_historicos()
            if not sucesso_download:
                raise Exception("Falha ao baixar dados históricos")

            # Fase 2: Inicializa ML com dados baixados
            self.logger.info(f"\nFase 2: Inicializando modelos ML...")
            dados_historicos = await self.db.get_dados_historicos(dias=30)
            if dados_historicos.empty:
                self.logger.error(f"Sem dados históricos para treinar modelos")
                raise Exception("Sem dados históricos para treinar modelos")

            sucesso_ml = await self.ml_predictor.inicializar(dados_historicos)
            if not sucesso_ml:
                self.logger.error(f"Falha ao inicializar modelos ML")
                raise Exception("Falha ao inicializar modelos ML")

            # Fase 3: Executa backtesting
            #self.logger.info(f"\nFase 3: Executando backtesting...")
            #resultados_backtest = await self.executar_backtest(dias=30)

            # Fase 4: Inicia monitoramento
            self.logger.info(f"\nSistema pronto! Iniciando monitoramento contínuo...")

            # Cria tasks para monitoramento e verificação
            tasks = [
                asyncio.create_task(self.analisar_mercado()),
                asyncio.create_task(self.verificar_resultados()),
                asyncio.create_task(self.monitorar_desempenho())  # Adiciona monitoramento
            ]

            # Executa tasks
            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"Erro crítico: {str(e)}")
            raise
        except KeyboardInterrupt:
            self.logger.info("Sistema encerrado pelo usuário")
      
if __name__ == "__main__":
    init()  # Inicializa colorama
    
    # Limpa qualquer event loop residual
    if asyncio._get_running_loop() is not None:
        asyncio._set_running_loop(None)
    
    sistema = TradingSystem()
    
    try:
        asyncio.run(sistema.executar())
    except KeyboardInterrupt:
        print("Sistema encerrado pelo usuário")
    
    
@dataclass
class BacktestTrade:
    entrada_timestamp: datetime
    saida_timestamp: datetime
    ativo: str
    direcao: str  # 'CALL' ou 'PUT'
    preco_entrada: float
    preco_saida: float
    resultado: str  # 'WIN' ou 'LOSS'
    lucro: float
    score_entrada: float
    assertividade_prevista: float