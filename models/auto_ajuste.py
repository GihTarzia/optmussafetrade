import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import asyncio
import optuna
import ta
import numpy as np
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
        self.historico_otimizacoes = deque(maxlen=100)
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

    async def _ajustar_filtros(self, direcao: str):
        """Ajusta filtros de entrada"""
        ajustes = {
            'analise.min_score_entrada': 0.05,
            'analise.min_confirmacoes': 1,
            'analise.min_volatilidade': 0.0001
        }

        for param, valor in ajustes.items():
            atual = self.config.get(param)
            novo = atual * (1 + valor) if direcao == 'aumentar' else atual * (1 - valor)
            self.config.set(param, novo)

    async def _ajustar_gestao_risco(self, direcao: str):
        """Ajusta parâmetros de gestão de risco"""
        ajustes = {
            'risco_por_operacao': 0.002,
            'stop_diario': 0.01,
            'max_operacoes_dia': 2
        }

        for param, valor in ajustes.items():
            atual = self.config.get(f'trading.{param}')
            novo = atual * (1 - valor) if direcao == 'reduzir' else atual * (1 + valor)
            self.config.set(f'trading.{param}', novo)

    async def _analisar_periodos(self):
        """Analisa performance por período do dia""" 
        for periodo, (inicio, fim) in self.periodos_analise.items():
            operacoes = await self.db.get_operacoes_periodo(inicio, fim)
            if len(operacoes) >= 20:
                metricas = self._calcular_metricas_periodo(operacoes)
                await self._otimizar_periodo(periodo, metricas)

    def _calcular_metricas_periodo(self, operacoes: List[Dict]) -> Dict:
        """Calcula métricas para um período específico"""
        total = len(operacoes)
        wins = len([op for op in operacoes if op['resultado'] == 'WIN'])

        return {
            'win_rate': wins / total if total > 0 else 0,
            'volume_medio': np.mean([op['volume'] for op in operacoes]),
            'tempo_medio': np.mean([op['duracao'].total_seconds() for op in operacoes]),
            'volatilidade': np.std([op['retorno'] for op in operacoes])
        }

    async def _otimizar_periodo(self, periodo: str, metricas: Dict):
        """Otimiza parâmetros para um período específico"""
        estudo = self.estudos[periodo]

        def objetivo(trial):
            params = self._criar_parametros_trial(trial)
            return self._avaliar_parametros_periodo(params, metricas)

        await asyncio.to_thread(
            estudo.optimize,
            objetivo,
            n_trials=50,
            timeout=1800
        )

        melhores_params = estudo.best_params
        self._atualizar_parametros_periodo(periodo, melhores_params)

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

    def _criar_parametros_trial(self, trial) -> Dict:
        """Cria conjunto de parâmetros para teste com ranges otimizados"""
        return {
            'analise': {
                'rsi': {
                    'periodo': trial.suggest_int('rsi_periodo', 7, 21),
                    'sobrevenda': trial.suggest_int('rsi_sobrevenda', 25, 35),
                    'sobrecompra': trial.suggest_int('rsi_sobrecompra', 65, 75)
                },
                'medias_moveis': {
                    'curta': trial.suggest_int('ma_curta', 5, 15),
                    'media': trial.suggest_int('ma_media', 15, 30),
                    'longa': trial.suggest_int('ma_longa', 30, 80)
                },
                'bandas_bollinger': {
                    'periodo': trial.suggest_int('bb_periodo', 12, 26),
                    'desvio': trial.suggest_float('bb_desvio', 1.8, 2.5)
                },
                'momentum': {
                    'periodo': trial.suggest_int('momentum_periodo', 8, 20),
                    'limite': trial.suggest_float('momentum_limite', 0.001, 0.005)
                }
            },
            'operacional': {
                'score_minimo': trial.suggest_float('score_minimo', 0.6, 0.85),
                'min_confirmacoes': trial.suggest_int('min_confirmacoes', 2, 4),
                'tempo_expiracao': trial.suggest_int('tempo_expiracao', 3, 10)
            }
        }

    def _validar_parametros_extendido(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Validação mais completa dos parâmetros"""
        resultados = self._simular_operacoes_completo(params, dados)

        if not resultados['operacoes']:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'drawdown': 1,
                'volatilidade_media': 0,
                'tempo_medio_operacao': 0
            }

        # Cálculos básicos
        total_ops = len(resultados['operacoes'])
        wins = len([op for op in resultados['operacoes'] if op['resultado'] > 0])

        # Cálculos avançados
        ganhos = sum(op['resultado'] for op in resultados['operacoes'] if op['resultado'] > 0)
        perdas = abs(sum(op['resultado'] for op in resultados['operacoes'] if op['resultado'] < 0))

        # Volatilidade
        retornos = [op['resultado'] / op['entrada'] for op in resultados['operacoes']]
        volatilidade = np.std(retornos) if retornos else 0

        # Tempo médio
        tempos = [(op['saida'] - op['entrada']).total_seconds() 
                 for op in resultados['operacoes']]
        tempo_medio = np.mean(tempos) if tempos else 0

        return {
            'win_rate': wins / total_ops,
            'profit_factor': ganhos / perdas if perdas > 0 else float('inf'),
            'drawdown': resultados['max_drawdown'],
            'volatilidade_media': volatilidade,
            'tempo_medio_operacao': int(tempo_medio)
        }

    def _analisar_horarios_otimos(self, dados: pd.DataFrame, params: Dict) -> List[str]:
        """Analisa horários com melhor performance"""
        resultados_hora = {}

        for hora in range(9, 21):  # 9h às 20h
            ops_hora = [op for op in dados['operacoes'] 
                       if op['timestamp'].hour == hora]

            if len(ops_hora) >= 10:  # Mínimo de operações para análise
                wins = len([op for op in ops_hora if op['resultado'] > 0])
                win_rate = wins / len(ops_hora)

                resultados_hora[hora] = {
                    'win_rate': win_rate,
                    'total_ops': len(ops_hora)
                }

        # Seleciona horários com win rate acima de 60%  
        horarios_otimos = [
            f"{hora:02d}:00"
            for hora, res in resultados_hora.items()
            if res['win_rate'] >= 0.6 and res['total_ops'] >= 20
        ]

        return horarios_otimos

    def _validar_melhoria(self, nova_otimizacao: ResultadoOtimizacao) -> bool:
        """Valida se nova otimização representa melhoria significativa"""
        if not self.melhor_resultado:
            return True

        # Critérios de melhoria
        melhorias = {
            'win_rate': nova_otimizacao.win_rate > self.melhor_resultado.win_rate * 1.05,
            'profit_factor': nova_otimizacao.profit_factor > self.melhor_resultado.profit_factor * 1.1,
            'drawdown': nova_otimizacao.drawdown < self.melhor_resultado.drawdown * 0.9,
            'score': nova_otimizacao.score_final > self.melhor_resultado.score_final * 1.05
        }

        # Precisa melhorar em pelo menos 2 critérios
        return sum(melhorias.values()) >= 2

    def _optuna_callback(self, study, trial):
        """Callback para monitoramento da otimização"""
        if trial.number % 10 == 0:
            self.logger.info(f"Trial {trial.number}: score = {trial.value:.4f}")

        # Salva estado intermediário
        if trial.number % 20 == 0:
            self._salvar_estado_otimizacao(study)

    def _salvar_estado_otimizacao(self, study):
        """Salva estado intermediário da otimização"""
        estado = {
            'timestamp': datetime.now().isoformat(),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }

        try:
            with open('data/otimizacao_estado.json', 'w') as f:
                json.dump(estado, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {str(e)}")

    

    def _necessita_otimizacao(self) -> bool:
        """Verifica se é necessário otimizar"""
        if not self.historico_otimizacoes:
            return True

        ultima = self.historico_otimizacoes[-1]
        tempo_passado = (datetime.now() - ultima.data_otimizacao).total_seconds()

        return (tempo_passado > 86400 or  # 24 horas
                ultima.win_rate < self.configuracoes['win_rate_minimo'] or
                ultima.drawdown > self.configuracoes['drawdown_maximo'])

    def _salvar_estado(self):
        """Salva estado atual do auto ajuste"""
        estado = {
            'timestamp': datetime.now().isoformat(),
            'parametros_atuais': self.parametros_atuais,
            'melhor_resultado': self.melhor_resultado._asdict() if self.melhor_resultado else None,
            'configuracoes': self.configuracoes,
            'metricas': self.metricas
        }

        try:
            self.db.salvar_estado_auto_ajuste(estado)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {str(e)}")

    def _simular_operacoes_completo(self, params: Dict, dados: pd.DataFrame) -> Dict:
        """Executa simulação completa para avaliação de parâmetros"""
        resultados = {
            'operacoes': [], 
            'max_drawdown': 0,
            'saldo': 1000 # Saldo inicial
        }

        # Simula operações 
        for i in range(len(dados) - 1):
            sinais = self._calcular_sinais(dados.iloc[i:i+1], params)

            for sinal in sinais:
                if self._validar_entrada(sinal, params):
                    op = self._simular_operacao(
                        sinal,
                        dados.iloc[i+1:i+params['operacional']['tempo_expiracao']]
                    )
                    if op:
                        resultados['operacoes'].append(op)

                        # Atualiza saldo e drawdown
                        resultados['saldo'] += op['resultado'] 
                        dd = (resultados['saldo'] - 1000) / 1000
                        resultados['max_drawdown'] = min(resultados['max_drawdown'], dd)

        return resultados

    def _calcular_sinais(self, dados: pd.DataFrame, params: Dict) -> List[Dict]:
        """Calcula sinais baseado nos parâmetros"""
        sinais = []

        # Análise RSI
        rsi = self._calcular_rsi(dados, params['analise']['rsi'])
        if rsi['sinal']:
            sinais.append(rsi)

        # Análise Médias Móveis
        mm = self._calcular_medias_moveis(dados, params['analise']['medias_moveis']) 
        if mm['sinal']:
            sinais.append(mm)

        # Análise Bollinger
        bb = self._calcular_bollinger(dados, params['analise']['bandas_bollinger'])
        if bb['sinal']:
            sinais.append(bb)

        # Análise Momentum
        mom = self._calcular_momentum(dados, params['analise']['momentum'])
        if mom['sinal']:
            sinais.append(mom)

        return sinais

    def _validar_entrada(self, sinal: Dict, params: Dict) -> bool:
        """Valida se sinal atende critérios mínimos"""
        return (
            sinal['score'] >= params['operacional']['score_minimo'] and
            sinal['confirmacoes'] >= params['operacional']['min_confirmacoes']
        )

    def _simular_operacao(self, sinal: Dict, dados_futuros: pd.DataFrame) -> Optional[Dict]:
        """Simula resultado de uma operação"""
        if dados_futuros.empty:
            return None

        preco_entrada = dados_futuros.iloc[0]['close']
        preco_saida = dados_futuros.iloc[-1]['close']

        resultado = (preco_saida - preco_entrada) if sinal['direcao'] == 'CALL' else (preco_entrada - preco_saida)

        return {
            'entrada': preco_entrada,
            'saida': preco_saida,
            'resultado': resultado,
            'timestamp': dados_futuros.index[0],
            'duracao': dados_futuros.index[-1] - dados_futuros.index[0],
            'retorno': resultado / preco_entrada,
            'volume': dados_futuros['volume'].mean() if 'volume' in dados_futuros else 0
        }

    def _atualizar_parametros(self, otimizacao: ResultadoOtimizacao):
        """Atualiza parâmetros do sistema com resultados otimizados"""
        self.parametros_atuais = otimizacao.parametros
        self.melhor_resultado = otimizacao

        # Atualiza configurações no sistema
        for categoria, params in otimizacao.parametros.items():
            if isinstance(params, dict):
                for param, valor in params.items():
                    self.config.set(f"{categoria}.{param}", valor)
            else:
                self.config.set(categoria, params)

        self._salvar_estado()

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
        
    def _dividir_dados(self, dados: pd.DataFrame, proporcao_treino: float = 0.7) -> (pd.DataFrame, pd.DataFrame):
        """Divide os dados em conjuntos de treino e teste"""
        try:
            tamanho_treino = int(len(dados) * proporcao_treino)
            dados_treino = dados.iloc[:tamanho_treino]
            dados_teste = dados.iloc[tamanho_treino:]
            return dados_treino, dados_teste
        except Exception as e:
            self.logger.error(f"Erro ao dividir dados: {str(e)}")
            return dados, pd.DataFrame()
        
    def _avaliar_parametros(self, dados: pd.DataFrame, parametros: Dict) -> float:
        """Avalia os parâmetros e retorna um score"""
        try:
            # Exemplo de avaliação: média dos scores dos indicadores
            score = (parametros['rsi'] + parametros['medias_moveis'] + parametros['bollinger'] + parametros['momentum']) / 4
            return score
        except Exception as e:
            self.logger.error(f"Erro ao avaliar parâmetros: {str(e)}")
            return 0.0
        
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
