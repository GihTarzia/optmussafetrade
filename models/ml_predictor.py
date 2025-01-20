import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yfinance as yf
from datetime import datetime
import ta
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Optional
from pathlib import Path

class MLPredictor:
    def __init__(self, config, logger):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}  # Novo dicionário para armazenar nomes das features
        self.models_path = Path("models/saved")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.config= config
        self.logger = logger
        ml_config = self.config.get('ml_config')
        self.cache_timeout = self.config.get('ml_parametros.cache_timeout')
        self.min_probabilidade = ml_config['min_probabilidade']
        self.min_accuracy = ml_config['min_accuracy']
        self.max_depth = ml_config['max_depth']
        self.learning_rate = ml_config['learning_rate']
        self.n_estimators = ml_config['n_estimators']
        self.min_confirmacoes = self.config.get('ml_config.min_confirmacoes')
        self.min_training_size = 2000
        
        
        # Novos parâmetros otimizados por tipo de ativo
        self.parametros_modelo = {
            'forex': {
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'scale_pos_weight': 1.0,
                'objective': 'binary:logistic',
                'random_state': 42
            },
        }
        #self.logger.info("\nMLPredictor inicializado com configurações:")
        #self.logger.info(f"Min probabilidade: {self.min_probabilidade}")
        #self.logger.info(f"Min accuracy: {self.min_accuracy}")
        #self.logger.info(f"Min dados treino: {self.min_training_size}")
        
    async def inicializar(self, dados_historicos: pd.DataFrame) -> bool:
        """Método inicial que deve ser chamado para preparar os modelos"""
        try:
            #self.logger.info("\n=== Iniciando MLPredictor ===")
            #self.logger.info(f"Timestamp: {datetime.now()}")
            
            if dados_historicos is None or dados_historicos.empty:
                self.logger.error("Erro: Sem dados históricos para inicialização")
                return False
            
            #self.logger.info(f"\nDados recebidos:")
            #self.logger.info(f"Total registros: {len(dados_historicos)}")
            #self.logger.info(f"Colunas: {dados_historicos.columns.tolist()}")
            #self.logger.info(f"Ativos únicos: {dados_historicos['ativo'].unique()}")
            
            # Treina modelos para cada ativo
            sucesso = await self.treinar(dados_historicos)
            
            if sucesso:
                #self.logger.info("\nStatus dos modelos treinados:")
                for ativo in self.models:
                    metricas = self.models[ativo]['metricas_validacao']
                    #self.logger.info(f"\n{ativo}:")
                    #self.logger.info(f"Accuracy: {metricas.get('accuracy', 0):.2%}")
                    #self.logger.info(f"Features: {len(self.models[ativo]['features'])}")
                    #self.logger.info(f"Última atualização: {self.models[ativo]['ultima_atualizacao']}")
            
            return sucesso
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {str(e)}")
            return False
        

    def _get_modelo_params(self, ativo):
        """Retorna parâmetros específicos para cada tipo de ativo"""
        if any(par in ativo for par in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD']):
            return self.parametros_modelo['forex']

    def criar_features(self, df: pd.DataFrame):
        """Cria features otimizadas para timeframe 1min"""
        try:
            self.logger.debug("Iniciando criação de features...")
            features = pd.DataFrame()   

            if len(df) < 30:
                self.logger.warning("Dados insuficientes para criar features")
                return None 

            # Verifica colunas necessárias
            required_columns = ['close', 'high', 'low', 'open', 'volume']
            df_copy = df.copy()
            df_copy.columns = df_copy.columns.str.lower()

            if not all(col in df_copy.columns for col in required_columns):
                self.logger.error(f"Colunas ausentes. Disponíveis: {df_copy.columns.tolist()}")
                return None 

            self.logger.info("Calculando indicadores técnicos...")

            # RSI períodos curtos
            for periodo in [5, 8, 13]:
                self.logger.debug(f"Calculando RSI período {periodo}")
                features[f'rsi_{periodo}'] = ta.momentum.RSIIndicator(
                    df_copy['close'], 
                    window=periodo
                ).rsi() 

            # MACD otimizado
            for (fast, slow, signal) in [(5,13,3), (3,8,2)]:
                self.logger.debug(f"Calculando MACD {fast},{slow},{signal}")
                macd = ta.trend.MACD(
                    df_copy['close'],
                    window_fast=fast,
                    window_slow=slow,
                    window_sign=signal
                )
                features[f'macd_{fast}_{slow}'] = macd.macd()
                features[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
                features[f'macd_diff_{fast}_{slow}'] = macd.macd_diff() 

            # Bollinger Bands
            for periodo in [8, 13, 21]:
                self.logger.debug(f"Calculando Bollinger período {periodo}")
                bollinger = ta.volatility.BollingerBands(
                    df_copy['close'], 
                    window=periodo,
                    window_dev=2
                )
                features[f'bb_high_{periodo}'] = bollinger.bollinger_hband()
                features[f'bb_low_{periodo}'] = bollinger.bollinger_lband()
                features[f'bb_mid_{periodo}'] = bollinger.bollinger_mavg()
                features[f'bb_bandwidth_{periodo}'] = bollinger.bollinger_wband()   

            # EMAs curtas
            for periodo in [3, 5, 8, 13, 21]:
                self.logger.debug(f"Calculando EMA período {periodo}")
                features[f'ema_{periodo}'] = ta.trend.EMAIndicator(
                    df_copy['close'], 
                    window=periodo
                ).ema_indicator()
                if periodo < 13:
                    features[f'dist_ema_{periodo}'] = (
                        df_copy['close'] - features[f'ema_{periodo}']
                    ) / features[f'ema_{periodo}']  

            # Estocástico rápido
            for k_periodo in [5, 8, 13]:
                for d_periodo in [2, 3]:
                    self.logger.debug(f"Calculando Estocástico K{k_periodo} D{d_periodo}")
                    stoch = ta.momentum.StochasticOscillator(
                        df_copy['high'],
                        df_copy['low'],
                        df_copy['close'],
                        window=k_periodo,
                        smooth_window=d_periodo
                    )
                    features[f'stoch_k_{k_periodo}_{d_periodo}'] = stoch.stoch()
                    features[f'stoch_d_{k_periodo}_{d_periodo}'] = stoch.stoch_signal() 

            # Volatilidade
            self.logger.debug("Calculando métricas de volatilidade")
            atr = ta.volatility.AverageTrueRange(
                df_copy['high'],
                df_copy['low'],
                df_copy['close'],
                window=8
            ).average_true_range()

            features['volatility_ratio'] = atr / df_copy['close']
            features['volatility_normalized'] = (
                atr - atr.rolling(21).mean()
            ) / atr.rolling(21).std()   

            # Volume
            if 'volume' in df_copy.columns:
                self.logger.debug("Calculando indicadores de volume")
                features['volume_ema_3'] = df_copy['volume'].ewm(span=3).mean()
                features['volume_ema_8'] = df_copy['volume'].ewm(span=8).mean()
                features['volume_ratio'] = df_copy['volume'] / features['volume_ema_8']
                features['volume_delta'] = df_copy['volume'].pct_change()   

            # Limpeza final
            self.logger.debug("Realizando limpeza dos dados")
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)    

            self.logger.info(f"Features criadas com sucesso: {len(features.columns)} indicadores")
            return features 

        except Exception as e:
            self.logger.error(f"Erro ao criar features: {str(e)}")
            return None
  
    async def treinar(self, dados_historicos) -> bool:
        """Treina o modelo de ML"""
        try:
            self.logger.info("\n=== Iniciando Treinamento ===")

            if dados_historicos is None:
                self.logger.error("Erro: Dados históricos vazios")
                return False
            
            # Verifica e converte colunas para lowercase
            dados_historicos.columns = dados_historicos.columns.str.lower()
            
            # Verifica colunas necessárias
            colunas_necessarias = ['ativo', 'close', 'high', 'low', 'open', 'volume']
            colunas_presentes = dados_historicos.columns
            
            self.logger.info("\nVerificando colunas:")
            for col in colunas_necessarias:
                presente = col in colunas_presentes
                self.logger.info(f"- {col}: {'OK' if presente else 'NOK'}")
                
            if not all(col in colunas_presentes for col in colunas_necessarias):
                self.logger.error("Erro: Colunas necessárias ausentes")
                return False

            # Para cada ativo
            ativos_processados = []
            for ativo in dados_historicos['ativo'].unique():
                self.logger.info(f"\n--- Processando {ativo} ---")
                dados_ativo = dados_historicos[dados_historicos['ativo'] == ativo].copy()
                self.logger.info(f"Registros: {len(dados_ativo)}")

                if len(dados_ativo) < self.min_training_size:
                    self.logger.warning(f"Dados insuficientes: {len(dados_ativo)} < {self.min_training_size}")
                    continue

                # Cria features
                features = self.criar_features(dados_ativo)
                if features is None:
                    self.logger.error("Erro na criação de features")
                    continue
                
                # Salvar ordem das features
                self.feature_names[ativo] = features.columns.tolist()
                self.logger.info(f"Features criadas: {features.shape[1]}")
                
                # Prepara labels
                retornos = dados_ativo['close'].pct_change().shift(-1)
                labels = (retornos > 0).astype(int)
                
                # Remove NaN
                valid_idx = ~features.isnull().any(axis=1) & ~labels.isnull()
                features = features[valid_idx]
                labels = labels[valid_idx]
                
                self.logger.warning(f"Dados válidos após limpeza: {len(features)}")

                # Treina modelo
                try:
                    # Normaliza dados
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(features)
                    
                    # Treina modelo final
                    model = XGBClassifier(**self._get_modelo_params(ativo))
                    model.fit(X_scaled, labels)
                    
                    # Salva modelo e configurações
                    self.models[ativo] = {
                        'model': model,
                        'features': features.columns.tolist(),
                        'scaler': scaler,
                        'metricas_validacao': {'accuracy': 0.6},  # Valor inicial
                        'ultima_atualizacao': datetime.now()
                    }
                    
                    # Salva scaler
                    self.scalers[ativo] = scaler
                    
                    ativos_processados.append(ativo)
                    self.logger.info(f"Modelo salvo com sucesso para {ativo}")
                    
                except Exception as e:
                    self.logger.error(f"Erro no treino do modelo: {str(e)}")
                    continue

            self.logger.info(f"\n=== Sumário do Treinamento ===")
            self.logger.info(f"Ativos processados: {len(ativos_processados)}")
            self.logger.info(f"Ativos com modelo: {list(self.models.keys())}")
            
            return len(ativos_processados) > 0
        
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            return False
        
    def _validar_dados_entrada(self, dados: pd.DataFrame, ativo: str) -> bool:
        """Valida dados de entrada para previsão"""
        try:
            if dados is None or dados.empty:
                self.logger.warning(f"Dados vazios para {ativo}")
                return False
            
            # Verifica colunas necessárias
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            colunas_presentes = [col.lower() for col in dados.columns]

            self.logger.info(f"Validando dados para {ativo}:")
            self.logger.info(f"Registros disponíveis: {len(dados)}")
            self.logger.info(f"Colunas presentes: {colunas_presentes}")

        
            if not all(col in colunas_presentes for col in required_columns):
                self.logger.warning(f"Colunas necessárias faltando para {ativo}. Presentes: {colunas_presentes}")
                return False
            
            # Verifica quantidade mínima de dados
            if len(dados) < 15:
                self.logger.warning(f"Dados insuficientes para {ativo}: {len(dados)} < 15")
                return False
            
            # Verifica se modelo existe
            if ativo not in self.models:
                self.logger.warning(f"Modelo não encontrado para {ativo}")
                return False
            # Verifica dados válidos
            if dados.isnull().any().any():
                self.logger.warning(f"Dados contêm valores nulos para {ativo}")
                return False
            
            self.logger.info(f"Dados validados com sucesso para {ativo}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação de dados: {str(e)}")
            return False
        
    async def prever(self, dados: pd.DataFrame, ativo: str) -> Optional[Dict]:
        try:
            self.logger.debug(f"Iniciando previsão para {ativo}")
            
            if not self._validar_dados_entrada(dados, ativo):
                return None
            
            features = await self._preparar_features(dados, ativo)
            if features is None:
                return None
            
            # Verificar se temos todas as features necessárias
            if not all(col in features.columns for col in self.feature_names[ativo]):
                self.logger.error("Features ausentes em relação ao treino")
                return None
            
            features_scaled = self.scalers[ativo].transform(features)
            model = self.models[ativo]['model']
            
            probabilidades = model.predict_proba(features_scaled)
            ultima_prob = probabilidades[-1]

            if max(ultima_prob) < self.min_probabilidade:
                return None
            
            
            # Validação mais rigorosa
            if max(ultima_prob) < self.min_probabilidade:
                return None

            # Calcula volatilidade normalizada
            volatilidade = dados['Close'].pct_change().rolling(8).std() * np.sqrt(252)
            volatilidade_norm = (
                volatilidade - volatilidade.rolling(21).mean()
            ) / volatilidade.rolling(21).std()

            if abs(float(volatilidade_norm.iloc[-1])) > 2:
                return None

            return {
                'ativo': ativo,
                'direcao': 'CALL' if ultima_prob[1] > ultima_prob[0] else 'PUT',
                'probabilidade': float(max(ultima_prob)),
                'score': float(max(ultima_prob)) * self.models[ativo]['metricas_validacao'].get('accuracy', 0),
                'timestamp': dados.index[-1],
                'volatilidade': float(volatilidade.iloc[-1]),
                'volatilidade_norm': float(volatilidade_norm.iloc[-1])
            }

        except Exception as e:
            self.logger.error(f"Erro na previsão: {str(e)}")
            return None
        
    def _calcular_volatilidade(self, dados: pd.DataFrame) -> float:
        """Calcula volatilidade dos dados"""
        try:
            return dados['close'].pct_change().std() * np.sqrt(252)
        except Exception:
            self.logger.error('Erro _calcular_volatilidade')
            return 0   
        
    async def _preparar_features(self, dados: pd.DataFrame, ativo) -> pd.DataFrame:
        """Prepara features mantendo consistência com treino"""
        try:
            # Usa o mesmo método de criação de features
            features = self.criar_features(dados)
            if features is None:
                return None
                
            # Garante ordem correta das features
            if ativo in self.feature_names:
                features = features[self.feature_names[ativo]]
                
            return features
     
        except Exception as e:
            self.logger.error(f"Erro ao preparar features: {str(e)}")
            return None
             
    def analisar(self, ativo, periodo='5d', intervalo='1m'):
        """Realiza análise completa de um ativo"""
        try:
            # Baixa dados mais recentes
            df = yf.download(ativo, period=periodo, interval=intervalo, progress=False)
            if df.empty:
                return None
                
             # Verifica condições básicas
            if len(df) < 50:
                return None
            
            # Faz previsão
            previsao = self.prever(df, ativo)
            if previsao is None:
                return None
            
            # Adiciona métricas de mercado
            previsao.update({
                'ativo': ativo,
                'preco_atual': float(df['Close'].iloc[-1]),
                'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0,
                'volatilidade': float(df['Close'].pct_change().std())
            })
            
            return previsao
            
        except Exception as e:
            self.logger.error(f"3Erro na análise de {ativo}: {str(e)}")
            return None
