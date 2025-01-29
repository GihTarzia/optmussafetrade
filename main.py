import sys
import traceback
import pandas as pd
import numpy as np
import ta
import asyncio
import yfinance as yf
from pathlib import Path
import pytz
import json

# Adiciona o diretório raiz ao PATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from datetime import datetime, timedelta
from dataclasses import dataclass
from colorama import init
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
        self.min_tempo_entre_analises = 5
        self.ultima_analise = {} 
        self.melhores_horarios = {}
        
        # Configura notificador
        token = self.config.get('notificacoes.telegram.token')
        chat_id = self.config.get('notificacoes.telegram.chat_id')
        self.notificador = Notificador(token, chat_id)
        
        self.cache = {}
        self.cache_timeout = 300  # 5 minutos
        self.limpar_cache_periodicamente()

        self.loop_intervals = {
            'verificar_sinais': 30,      # segundos
            'verificar_resultados': 60,   # segundos
            'atualizar_dados': 60        # segundos
        }

        # Inicializa componentes principais
        self.ml_predictor = MLPredictor(self.config, self.logger)
        self.analise_padroes = AnalisePadroesComplexos(self.config, self.logger)
        self.gestao_risco = GestaoRiscoAdaptativo(self.config, self.logger)
        self.dados_buffer = {}
        self.min_registros_necessarios = 30
        
    async def executar(self):
        """Executa o sistema de trading"""
        try:
            self.logger.info("=== Iniciando Sistema de Trading ===")
            
            # Inicializa componentes que precisam de await
            await self.inicializar()
            
            # Cria tasks para execução paralela
            tasks = [
                asyncio.create_task(self.loop_verificar_sinais()),
                asyncio.create_task(self.loop_verificar_resultados()),
                asyncio.create_task(self.loop_atualizar_dados())
            ]
            
            # Aguarda todas as tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Erro na execução: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def inicializar(self):
        """Inicializa componentes que precisam de await"""
        try:
            self.logger.info("Inicializando componentes...")
            
            # Inicializa notificador
            await self.notificador.start()
            
            # Inicializa ML predictor
            await self.ml_predictor.inicializar_modelos()
            
            self.logger.info("Componentes inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    async def monitorar_desempenho(self):
        """Monitora desempenho e ajusta parâmetros"""
        while True:
            try:
                metricas = self.gestao_risco.get_estatisticas()
                
                # Limpa dados antigos a cada 24 horas
                await self.db.limpar_dados_antigos(1)
            
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
                            
            return BacktestTrade(
                entrada_timestamp=timestamp,
                saida_timestamp=dados_futuros.index[idx_exp],
                ativo=sinal_ml['ativo'],
                direcao=sinal_ml['direcao'],
                preco_entrada=preco_entrada,
                preco_saida=preco_saida,
                resultado=resultado,
                score_entrada=sinal_ml['probabilidade'],
                assertividade_prevista=sinal_ml.get('score', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Erro na simulação: {str(e)}")
            return None

    async def analisar_mercado(self):
        """Analisa mercado e gera sinais com validações melhoradas"""
        while True:
            try:
                ativos = self.config.get_ativos_ativos()
                hora_atual = datetime.now().hour
                
                for ativo in ativos:
                    # Verifica tempo desde última análise
                    ultima = self.ultima_analise.get(ativo, datetime.min)
                    if (datetime.now() - ultima).total_seconds() < self.min_tempo_entre_analises * 60:
                        continue
                        
                    # Obtém dados recentes
                    dados = await self.db.get_dados_recentes(ativo)
                    if dados.empty:
                        continue
                        
                    # Análise técnica e ML
                    padroes = self.analise_padroes.analisar(dados)
                    previsao_ml = await self.ml_predictor.prever(dados, ativo)
                    
                    if not padroes or not previsao_ml:
                        continue
                        
                    # Calcula métricas de mercado
                    volatilidade = self._calcular_volatilidade(dados)
                    momentum_score = self._calcular_momentum_score(dados)
                    
                    # Determina tempo de expiração dinâmico
                    tempo_expiracao = self._calcular_tempo_expiracao(
                        volatilidade=volatilidade,
                        momentum_score=momentum_score
                    )
                    
                    # Valida win rate do ativo no horário
                    win_rate = await self.db.get_taxa_sucesso_horario_ativo(hora_atual, ativo)
                    if win_rate < self.config.get('trading.win_rate_minimo', 58):
                        self.logger.info(f"Win rate insuficiente para {ativo} às {hora_atual}h: {win_rate:.1f}%")
                        continue
                    
                    # Gera sinal
                    sinal = {
                        'ativo': ativo,
                        'direcao': previsao_ml['direcao'],
                        'timestamp': datetime.now(pytz.UTC),
                        'tempo_expiracao': tempo_expiracao,
                        'preco_entrada': dados['close'].iloc[-1],
                        'score': previsao_ml['score'],
                        'volatilidade': volatilidade,
                        'momentum_score': momentum_score,
                        'probabilidade': previsao_ml['probabilidade'],
                        'assertividade': previsao_ml['assertividade']
                    }
                    
                    # Valida sinal repetido
                    if await self.db.valida_sinal_repetido(sinal):
                        self.logger.info(f"Sinal repetido ignorado para {ativo}")
                        continue
                    
                    # Salva e notifica
                    if await self.db.salvar_sinal(sinal):
                        await self.notificador.enviar_sinal(sinal)
                        self.ultima_analise[ativo] = datetime.now()
                    
                await asyncio.sleep(30)  # Aguarda 30 segundos antes da próxima análise
                
            except Exception as e:
                self.logger.error(f"Erro na análise de mercado: {str(e)}")
                await asyncio.sleep(60)  # Aguarda 1 minuto em caso de erro

    def _calcular_volatilidade(self, dados: pd.DataFrame) -> float:
        """Calcula volatilidade normalizada"""
        try:
            retornos = dados['close'].pct_change()
            volatilidade = retornos.std() * np.sqrt(252)
            return volatilidade
        except Exception as e:
            self.logger.error(f"Erro ao calcular volatilidade: {str(e)}")
            return 0.001

    def _calcular_momentum_score(self, dados: pd.DataFrame) -> float:
        """Calcula score de momentum entre 0 e 1"""
        try:
            # RSI
            rsi = ta.momentum.RSIIndicator(dados['close']).rsi().iloc[-1]
            rsi_norm = (rsi - 30) / 40  # Normaliza entre 0 e 1
            
            # ROC (Rate of Change)
            roc = ta.momentum.ROCIndicator(dados['close']).roc().iloc[-1]
            roc_norm = (roc + 2) / 4  # Normaliza assumindo range típico de -2% a 2%
            
            # Média ponderada
            momentum_score = (rsi_norm * 0.6 + roc_norm * 0.4)
            return max(0, min(1, momentum_score))  # Garante entre 0 e 1
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular momentum score: {str(e)}")
            return 0.5

    def _calcular_tempo_expiracao(self, volatilidade: float, momentum_score: float) -> int:
        """Calcula tempo de expiração dinâmico baseado em análise de mercado"""
        try:
            # Base inicial de 3 minutos
            tempo_base = 3
            
            # Ajusta baseado na volatilidade (normalizada)
            if volatilidade > 0.002:  # Alta volatilidade
                tempo_base += 2
            elif volatilidade < 0.0005:  # Baixa volatilidade
                tempo_base -= 1
            
            # Ajusta baseado no momentum
            if momentum_score > 0.7:  # Momentum forte
                tempo_base -= 1
            elif momentum_score < 0.3:  # Momentum fraco
                tempo_base += 1
            
            # Garante mínimo de 2 e máximo de 5 minutos
            return max(2, min(5, tempo_base))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular tempo expiração: {str(e)}")
            return 3  # Valor padrão em caso de erro

    async def verificar_resultados(self):
        """Verifica resultados dos sinais pendentes"""
        try:
            self.logger.info("=== Iniciando verificação de resultados ===")
            
            # Busca sinais pendentes
            sinais_pendentes = await self.db.get_sinais_sem_resultado()
            
            if not sinais_pendentes:
                return
            
            self.logger.info(f"Encontrados {len(sinais_pendentes)} sinais pendentes")
            
            for sinal in sinais_pendentes:
                try:
                    self.logger.info(f"""
                        Processando sinal ID {sinal['id']}:
                        - Ativo: {sinal['ativo']}
                        - Direção: {sinal['direcao']}
                        - Timestamp: {sinal['timestamp']}
                        - Tempo expiração: {sinal['tempo_expiracao']} min
                        - Preço entrada: {sinal['preco_entrada']}
                    """)
                    
                    # Garante que timestamp está em UTC
                    if isinstance(sinal['timestamp'], str):
                        sinal['timestamp'] = datetime.fromisoformat(sinal['timestamp'].replace('Z', '+00:00'))
                    if sinal['timestamp'].tzinfo is None:
                        sinal['timestamp'] = pytz.UTC.localize(sinal['timestamp'])
                    
                    # Calcula momento de saída
                    momento_saida = sinal['timestamp'] + timedelta(minutes=sinal['tempo_expiracao'])
                    
                    # Verifica se já passou o tempo de expiração
                    agora = datetime.now(pytz.UTC)
                    if agora < momento_saida:
                        self.logger.info(f"Sinal ainda não expirou. Aguardando... ({momento_saida - agora})")
                        continue
                    
                    # Busca dados do período
                    dados = await self.db.get_dados_periodo(
                        ativo=sinal['ativo'],
                        inicio=sinal['timestamp'],
                        fim=momento_saida
                    )
                    
                    if dados.empty:
                        self.logger.error(f"Sem dados para verificar resultado do sinal {sinal['id']}")
                        continue
                    
                    # Obtém preço de saída
                    preco_saida = float(dados['close'].iloc[-1])
                    
                    # Determina resultado
                    if sinal['direcao'] == 'CALL':
                        win = preco_saida > sinal['preco_entrada']
                    else:  # PUT
                        win = preco_saida < sinal['preco_entrada']
                    
                    # Calcula lucro/prejuízo
                    variacao = abs(preco_saida - sinal['preco_entrada']) / sinal['preco_entrada']
                    lucro = variacao * 100 if win else -variacao * 100
                    
                    # Monta resultado
                    resultado = {
                        'id': sinal['id'],
                        'resultado': 'WIN' if win else 'LOSS',
                        'preco_saida': preco_saida,
                        'timestamp_saida': momento_saida,
                        'lucro': lucro
                    }
                    
                    # Salva resultado
                    if await self.db.salvar_resultado(sinal['id'], resultado):
                        self.logger.info(f"""
                            Resultado processado:
                            - ID: {sinal['id']}
                            - Resultado: {resultado['resultado']}
                            - Lucro: {resultado['lucro']:.2f}%
                            - Preço Saída: {resultado['preco_saida']}
                        """)
                        
                        # Envia notificação
                        operacao = {**sinal, **resultado}
                        await self.notificador.enviar_resultado(operacao)
                    else:
                        self.logger.error(f"Erro ao salvar resultado do sinal {sinal['id']}")
                
                except Exception as e:
                    self.logger.error(f"Erro ao processar resultado do sinal {sinal['id']}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    continue
                
        except Exception as e:
            self.logger.error(f"Erro na verificação de resultados: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def loop_verificar_sinais(self):
        """Loop contínuo para verificação de sinais"""
        while True:
            try:
                self.logger.info("=== Iniciando verificação de sinais ===")
                await self.verificar_sinais()
            except Exception as e:
                self.logger.error(f"Erro no loop de sinais: {str(e)}")
            finally:
                await asyncio.sleep(self.loop_intervals['verificar_sinais'])

    async def loop_verificar_resultados(self):
        """Loop contínuo para verificação de resultados"""
        while True:
            try:
                self.logger.info("=== Iniciando verificação de resultados ===")
                await self.verificar_resultados()
            except Exception as e:
                self.logger.error(f"Erro no loop de resultados: {str(e)}")
            finally:
                await asyncio.sleep(self.loop_intervals['verificar_resultados'])

    async def loop_atualizar_dados(self):
        """Loop contínuo para atualização de dados"""
        while True:
            try:
                self.logger.info("=== Iniciando atualização de dados ===")
                await self.atualizar_dados_mercado()
            except Exception as e:
                self.logger.error(f"Erro no loop de dados: {str(e)}")
            finally:
                await asyncio.sleep(self.loop_intervals['atualizar_dados'])

    async def limpar_cache_periodicamente(self):
        """Limpa cache periodicamente para evitar vazamento de memória"""
        while True:
            try:
                agora = datetime.now()
                for key in list(self.cache.keys()):
                    if (agora - self.cache[key]['timestamp']).total_seconds() > self.cache_timeout:
                        del self.cache[key]
                await asyncio.sleep(300)  # Verifica a cada 5 minutos
            except Exception as e:
                self.logger.error(f"Erro ao limpar cache: {str(e)}")

    async def verificar_sinais(self):
        """Verifica sinais para cada ativo"""
        try:
            self.logger.info("=== Iniciando verificação de sinais ===")
            
            for ativo in self.config.get_ativos_ativos():
                # Log detalhado por ativo
                self.logger.info(f"\nAnalisando {ativo}...")
                
                dados = await self.db.get_dados_recentes(ativo)
                if dados is None or dados.empty:
                    self.logger.warning(f"Sem dados para {ativo}")
                    continue
                    
                            # Gerencia buffer de dados
                if ativo not in self.dados_buffer:
                    self.dados_buffer[ativo] = dados
                else:
                    # Concatena novos dados mantendo histórico
                    self.dados_buffer[ativo] = pd.concat([
                        self.dados_buffer[ativo],
                        dados
                    ]).drop_duplicates()
                            # Usa últimos N registros para análise
                dados_analise = self.dados_buffer[ativo].tail(50)  # Mantém margem extra
                
                if len(dados_analise) < self.min_registros_necessarios:
                    self.logger.warning(f"Dados insuficientes para {ativo}: {len(dados_analise)}")
                    continue
            
                # Log dos preços atuais
                self.logger.info(f"""
                Preços {ativo}:
                - Atual: {dados['close'].iloc[-1]:.5f}
                - Máximo: {dados['high'].iloc[-1]:.5f}
                - Mínimo: {dados['low'].iloc[-1]:.5f}
                - Volume: {dados['volume'].iloc[-1]:.2f}
                """)
                
                # Análise ML
                predicao = await self.ml_predictor.prever(dados, ativo)
                if predicao:
                    # Análise de Padrões
                    padroes = await self.analise_padroes.analisar(dados)
                    padroes_str = json.dumps([{
                        'nome': p.nome,
                        'forca': p.forca,
                        'confiabilidade': p.confiabilidade
                    } for p in padroes]) if padroes else '[]'
                    
                    # Indicadores técnicos
                    indicadores = predicao.get('indicadores', {})
                    indicadores_str = json.dumps(indicadores) if indicadores else None
                    
                    # Calcula assertividade baseada na probabilidade
                    prob = predicao['probabilidade']
                    assertividade = prob * 100 if prob > 0.5 else (1 - prob) * 100
                    
                    # Calcula volatilidade
                    volatilidade = dados['close'].pct_change().std() * 100  # em porcentagem
                    
                    # Prepara indicadores
                    indicadores = {
                        'rsi': float(predicao['indicadores']['rsi']),
                        'macd_diff': float(predicao['indicadores']['macd_diff']),
                        'bb_spread': float(predicao['indicadores']['bb_spread']),
                        'momentum': float(predicao['indicadores']['momentum'])
                    }
                    
                    # Monta sinal completo com todos os campos necessários
                    sinal = {
                        'ativo': ativo,
                        'direcao': predicao['direcao'],
                        'timestamp': datetime.now(pytz.UTC),
                        'tempo_expiracao': 5,
                        'preco_entrada': float(dados['close'].iloc[-1]),
                        'score': float(predicao['score']),
                        'probabilidade': float(predicao['probabilidade'] * 100),  # Convertido para porcentagem
                        'assertividade': float(predicao['assertividade']),
                        'padroes': padroes_str,
                        'indicadores': indicadores_str,
                        'ml_prob': float(predicao['probabilidade'] * 100),  # Também em porcentagem
                        'volatilidade': float(volatilidade)
                    }
                    
                    self.logger.info(f"""
                    Sinal gerado:
                    Ativo: {sinal['ativo']}
                    Direção: {sinal['direcao']}
                    Timestamp: {sinal['timestamp']}
                    Probabilidade: {sinal['probabilidade']:.4f}
                    Score: {sinal['score']:.4f}
                    """)
                    
                    # Salva e notifica
                    if await self.db.salvar_sinal(sinal):
                        await self.notificador.enviar_sinal(sinal)
                        self.logger.info(f"Sinal salvo e enviado com sucesso para {ativo}")
                    else:
                        self.logger.error(f"Erro ao salvar sinal para {ativo}")
                
                # Análise de Padrões
                padroes = await self.analise_padroes.analisar(dados)
                if padroes:
                    self.logger.info(f"Padrões encontrados para {ativo}: {padroes}")
                
                # Validação de Volume
                volume_ok = self.analise_padroes._validar_volume(dados)
                self.logger.info(f"Volume válido para {ativo}: {volume_ok}")
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar sinais: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def atualizar_dados_mercado(self):
        """Atualiza dados de mercado a cada 60 segundos"""
        while True:
            try:
                self.logger.info("Iniciando atualização de dados...")
                
                for ativo in self.config.get_ativos_ativos():
                    try:
                        # Baixa dados mais recentes
                        dados = yf.download(
                            ativo,
                            period="1d",  # Último dia
                            interval="1m",  # Intervalo de 1 minuto
                            progress=False
                        )
                        
                        if not dados.empty:
                            # Garante que o índice está em UTC
                            if dados.index.tz is None:
                                dados.index = dados.index.tz_localize('UTC')
                            elif dados.index.tz.zone != 'UTC':
                                dados.index = dados.index.tz_convert('UTC')
                                
                            # Padroniza nomes das colunas
                            dados.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                            dados = dados[['Open', 'High', 'Low', 'Close', 'Volume']]
                            
                            # Salva novos dados
                            await self.db.salvar_precos_novos(ativo, dados)
                        else:
                            self.logger.warning(f"Nenhum dado obtido para {ativo}")
                            
                    except Exception as e:
                        self.logger.error(f"Erro ao atualizar {ativo}: {str(e)}")
                        continue
                        
                await asyncio.sleep(60)  # Aguarda 60 segundos
                
            except Exception as e:
                self.logger.error(f"Erro na atualização de dados: {str(e)}")
                await asyncio.sleep(60)  # Aguarda antes de tentar novamente

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
    score_entrada: float
    assertividade_prevista: float