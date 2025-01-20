import sqlite3
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional
import threading
from pathlib import Path
from queue import Queue
from contextlib import contextmanager
import pytz
import yfinance as yf
import asyncio
from datetime import datetime, time

class ConnectionPool:
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Inicializa o pool
        for _ in range(max_connections):
            conn = sqlite3.connect(db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            self.connections.put(conn)
    
    @contextmanager
    def get_connection(self):
        max_retries = 5
        retry_delay = 0.1  # segundos

        for attempt in range(max_retries):
            try:
                connection = self.connections.get(timeout=1)
                try:
                    yield connection
                finally:
                    self.connections.put(connection)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                asyncio.sleep(retry_delay * (attempt + 1))
    
    def close_all(self):
        while not self.connections.empty():
            conn = self.connections.get()
            conn.close()

class DatabaseManager:
    def __init__(self, logger, db_path: str = 'data/trading_bot.db'):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.lock = threading.Lock()
        self.pool = ConnectionPool(db_path, max_connections=10)  # Aumentado para 1min
        self.logger = logger
        # Cache para otimização
        self.cache = {
            'dados_mercado': {},
            'analises': {},
            'horarios': {}
        }
        self.cache_timeout = 30  
        self.cache_last_update = {}
        
        self._init_database()
        self._optimize_database()
    
    def _optimize_database(self):
        """Otimiza o banco de dados"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')
            cursor.execute('PRAGMA cache_size=-4000')  # Aumentado para 4MB
            cursor.execute('PRAGMA temp_store=MEMORY')
            cursor.execute('PRAGMA mmap_size=268435456')  # 256MB para mmap
            conn.commit()
    
    def _init_database(self):
        """Inicializa as tabelas com otimizações"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de preços com particionamento por data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    date_partition TEXT GENERATED ALWAYS AS (date(timestamp)) VIRTUAL,
                    hour_partition INTEGER GENERATED ALWAYS AS (strftime('%H', timestamp)) VIRTUAL,
                    UNIQUE(ativo, timestamp)
                )
            ''')
  
            # Tabela de sinais com cache de resultados
            cursor.execute('''
CREATE TABLE IF NOT EXISTS sinais (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ativo TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    direcao TEXT NOT NULL,
    preco_entrada REAL NOT NULL,
    preco_saida REAL,
    tempo_expiracao INTEGER NOT NULL,
    score REAL NOT NULL,
    assertividade REAL NOT NULL,
    resultado TEXT,
    lucro REAL,
    padroes TEXT,
    indicadores TEXT,
    ml_prob REAL,
    volatilidade REAL,
    processado BOOLEAN DEFAULT 0,
    data_processamento DATETIME,
    -- Restrição de unicidade para evitar duplicatas
    UNIQUE(ativo, timestamp, direcao, preco_entrada, tempo_expiracao)
);
            ''')

            # Tabela de métricas com resumos pré-calculados
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metricas_resumo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ativo TEXT NOT NULL,
                    periodo TEXT NOT NULL,
                    data_calculo DATETIME NOT NULL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_operacoes INTEGER,
                    media_assertividade REAL,
                    UNIQUE(ativo, periodo, data_calculo)
                )
            ''')
            
            # Nova tabela para resultados de backtest
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_resultados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    drawdown_maximo REAL NOT NULL,
                    retorno_total REAL NOT NULL,
                    metricas TEXT NOT NULL,
                    melhores_horarios TEXT NOT NULL,
                    evolucao_capital TEXT NOT NULL
                )
            ''')           
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analises_detalhadas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sinal_id INTEGER NOT NULL,
                    dados_analise TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY(sinal_id) REFERENCES sinais(id),
                    UNIQUE(sinal_id)
                )
            ''')
            
            # Índices otimizados
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_ativo_date ON precos(ativo, date_partition, hour_partition)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_timestamp_range ON precos(timestamp, ativo)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_timestamp ON sinais(timestamp, ativo)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_processado ON sinais(processado, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analises_sinal_id ON analises_detalhadas(sinal_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_volume ON precos(volume) WHERE volume > 0')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_timestamp_ativo ON precos(timestamp, ativo)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_resultado ON sinais(resultado) WHERE resultado IS NOT NULL')

            conn.commit()
            
    def get_horarios_sucesso(self, ativo: str) -> Dict[str, float]:
        """Análise de horários otimizada"""
        try:
            cache_key = f"horarios_{ativo}"
            if cache_key in self.cache['horarios']:
                return self.cache['horarios'][cache_key]

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    WITH horarios_wins AS (
                        SELECT 
                            strftime('%H:%M', timestamp) AS horario,
                            COUNT(CASE WHEN resultado = 'WIN' THEN 1 END) * 1.0 / COUNT(*) AS taxa_sucesso,
                            COUNT(*) as total_ops
                        FROM sinais 
                        WHERE ativo = ?
                        AND timestamp >= datetime('now', '-7 days')
                        AND resultado IS NOT NULL
                        GROUP BY strftime('%H:%M', timestamp)
                        HAVING total_ops >= 5
                    )
                    SELECT horario, taxa_sucesso
                    FROM horarios_wins
                    WHERE taxa_sucesso >= 0.55
                    ORDER BY taxa_sucesso DESC
                """

                cursor.execute(query, (ativo,))
                resultados = {row[0]: row[1] for row in cursor.fetchall()}

                # Cache por 5 minutos
                self.cache['horarios'][cache_key] = resultados

                self.logger.info(f"Horários de sucesso calculados para {ativo}")
                return resultados

        except Exception as e:
            self.logger.error(f"Erro ao obter horários de sucesso: {str(e)}")
            return {}
      
      
        
    def get_taxa_sucesso_horario(self, hora: int) -> float:
        """Retorna taxa de sucesso para um horário específico"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT 
                        COUNT(CASE WHEN resultado = 'WIN' THEN 1 END) * 1.0 / COUNT(*) as taxa_sucesso,
                        COUNT(*) as total_operacoes
                    FROM sinais
                    WHERE strftime('%H', timestamp) = ?
                    AND resultado IS NOT NULL
                    AND timestamp >= datetime('now', '-30 days')
                """

                cursor.execute(query, (f"{hora:02d}",))
                resultado = cursor.fetchone()

                if resultado and resultado[0] is not None:
                    total_ops = resultado[1]
                    # Só considera taxa se tiver mínimo de operações
                    if total_ops >= 10:  # Mínimo de 10 operações para considerar
                        return float(resultado[0])
                return 0.65  # Taxa neutra se não houver dados

        except Exception as e:
            self.logger.error(f"Erro ao obter taxa de sucesso do horário: {str(e)}")
            return 0.65
        
    def get_assertividade_recente(self, ativo: str, direcao: str, tempo_expiracao: int) -> Optional[float]:
        """Retorna a assertividade média recente para um ativo e direção"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT AVG(assertividade) as assertividade_media,
                       COUNT(*) as total_ops,
                       SUM(CASE WHEN resultado = 'WIN' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
                FROM sinais
                WHERE ativo = ?
                AND direcao = ?
                AND tempo_expiracao = ?
                AND timestamp >= datetime('now', '-3 days')
                AND resultado IS NOT NULL
                HAVING total_ops >= 10
                """
                
                cursor.execute(query, (ativo, direcao, tempo_expiracao))
                result = cursor.fetchone()
                
                if result and result[0] is not None:
                    return float(result[0])
                return 50.0  # Valor padrão se não houver dados
        
        except Exception as e:
            self.logger.error(f"Erro ao obter assertividade recente: {str(e)}")
            return None
        
    # Adicionar novo método para salvar resultados
    async def salvar_resultados_backtest(self, resultados: Dict) -> bool:
        """Salva resultados de backtest com métricas detalhadas"""
        try:
            with await self.transaction() as conn:
                cursor = conn.cursor()

                query = '''
                    INSERT INTO backtest_resultados (
                        timestamp, 
                        total_trades, 
                        win_rate, 
                        profit_factor,
                        drawdown_maximo, 
                        retorno_total, 
                        metricas,
                        melhores_horarios, 
                        evolucao_capital,
                        volatilidade_media,
                        tempo_medio_operacao,
                        sharpe_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''

                metricas = resultados['metricas']

                cursor.execute(query, (
                    datetime.now(),
                    metricas['total_trades'],
                    metricas['win_rate'],
                    metricas['profit_factor'],
                    metricas['drawdown_maximo'],
                    metricas['retorno_total'],
                    json.dumps(metricas),
                    json.dumps(resultados['melhores_horarios']),
                    json.dumps(resultados['evolucao_capital']),
                    metricas.get('volatilidade_media', 0),
                    metricas.get('tempo_medio_operacao', 0),
                    metricas.get('sharpe_ratio', 0)
                ))

                self.logger.info("Resultados do backtest salvos com sucesso")
                return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados do backtest: {str(e)}")
            return False

    async def limpar_dados_antigos(self, dias_retencao: int = 90) -> bool:
        """Limpa dados antigos do banco"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Limpa preços antigos
                cursor.execute("""
                    DELETE FROM precos 
                    WHERE timestamp < datetime('now', ? || ' days')
                """, (-dias_retencao,))
                
                # Limpa sinais antigos
                cursor.execute("""
                    DELETE FROM sinais 
                    WHERE timestamp < datetime('now', ? || ' days')
                """, (-dias_retencao,))
                
                conn.commit()
                self.logger.info(f"Dados anteriores a {dias_retencao} dias removidos")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro na limpeza de dados: {str(e)}")
            return False
        
        
    @contextmanager
    async def transaction(self):
        """Gerenciador de contexto para transações com retry"""
        max_retries = 3
        retry_delay = 1  # segundos

        for attempt in range(max_retries):
            try:
                with self.pool.get_connection() as conn:
                    try:
                        yield conn
                        conn.commit()
                        break
                    except Exception as e:
                        conn.rollback()
                        raise e
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Tentativa {attempt + 1} falhou, tentando novamente...")
                await asyncio.sleep(retry_delay * (attempt + 1))
    
            
    async def get_dados_mercado(self, ativo: str) -> pd.DataFrame:
        cache_key = f"mercado_{ativo}"

        try:
            # Verifica cache
            if cache_key in self.cache['dados_mercado']:
                data, timestamp = self.cache['dados_mercado'][cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                    return data

            dados = await self._baixar_dados_mercado(ativo)
            await self.salvar_precos_novos(ativo, dados)
            
            if not dados.empty:
                self.logger.info(f"Dados obtidos para {ativo}: {len(dados)} registros")
                self.cache['dados_mercado'][cache_key] = (dados, datetime.now())
                return dados

            else:
                self.logger.warning(f"Nenhum dado retornado do yfinance para {ativo}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Erro ao obter dados de mercado: {str(e)}")
            return pd.DataFrame()

    async def _baixar_dados_mercado(self, ativo: str) -> pd.DataFrame:
        max_retries = 3
        retry_delay = 2  # segundos

        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ativo,
                    period="1d",
                    interval="1m",
                    progress=False
                )

                if not df.empty:
                    return df

                if attempt < max_retries - 1:
                    self.logger.warning(f"Tentativa {attempt + 1} falhou para {ativo}, tentando novamente...")
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except Exception as e:
                self.logger.error(f"Erro ao baixar dados de mercado para {ativo}: {str(e)}")
                return pd.DataFrame()

    async def get_preco(self, ativo: str, momento: datetime) -> Optional[float]:
        """Obtém preço com otimização de busca"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Busca exata primeiro
                momento_str = momento.astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("""
                    SELECT close
                    FROM precos
                    WHERE ativo = ? AND timestamp = ?
                    LIMIT 1
                """, (ativo, momento_str))

                resultado = cursor.fetchone()
                if resultado:
                    return resultado[0]

                # Busca mais próximo dentro de 90 segundos
                cursor.execute("""
                    SELECT close, 
                           ABS(STRFTIME('%s', timestamp) - STRFTIME('%s', ?)) as diff
                    FROM precos
                    WHERE ativo = ?
                      AND timestamp BETWEEN datetime(?, '-1 seconds') 
                                      AND datetime(?, '+90 seconds')
                    ORDER BY diff ASC
                    LIMIT 1
                """, (momento_str, ativo, momento_str, momento_str))

                resultado = cursor.fetchone()
                if resultado:
                    return resultado[0]

                return None

        except Exception as e:
            self.logger.error(f"Erro ao recuperar preço: {str(e)}")
            return None


    async def atualizar_resultado_sinal(self, sinal_id: int, **kwargs) -> bool:
        """Atualiza resultado de um sinal"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = '''
                    UPDATE sinais
                    SET resultado = ?,
                        lucro = ?,
                        preco_saida = ?,
                        processado = 1,
                        data_processamento = ?
                    WHERE id = ?
                '''

                cursor.execute(query, (
                    kwargs['resultado'],
                    kwargs['lucro'],
                    kwargs['preco_saida'],
                    kwargs['data_processamento'].strftime('%Y-%m-%d %H:%M:%S'),
                    sinal_id
                ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Erro ao atualizar resultado do sinal: {str(e)}")
            return False

    async def registrar_sinal(self, sinal: Dict) -> Optional[int]:
        """Registra sinal com validações aprimoradas"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                indicadores = json.dumps(sinal.get('indicadores', {}))  # Converte para string JSON


                assertividade = min(100, max(0, sinal['assertividade']))
                padroes = json.dumps(sinal.get('padroes_forca', 0))

                cursor.execute("""
                    INSERT INTO sinais (
                        ativo, timestamp, direcao, preco_entrada,
                        tempo_expiracao, score, assertividade, padroes,
                        ml_prob, volatilidade, indicadores, processado
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sinal['ativo'],
                    sinal['momento_entrada'].strftime('%Y-%m-%d %H:%M:%S'),
                    sinal['direcao'],
                    sinal['preco_entrada'],
                    sinal['tempo_expiracao'],
                    sinal['score'],
                    assertividade,
                    padroes,
                    sinal['ml_prob'],
                    sinal['volatilidade'],
                    indicadores,  # Agora é string JSON
                    False
                ))

                conn.commit()
                sinal_id = cursor.lastrowid
                self.logger.info(f"Sinal {sinal_id} registrado para {sinal['ativo']}")
                return sinal_id

        except Exception as e:
            self.logger.error(f"Erro ao registrar sinal: {str(e)}")
            return None
        
    async def valida_sinal_repetido(self, sinal: Dict) -> Optional[int]:
        """Valida sinal repetido com mesmo ativo, direcao, preco_entrada e timestamp (sem milissegundo)"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Verifica duplicidade
                cursor.execute("""
                    SELECT id FROM sinais
                    WHERE ativo = ? AND 
                          timestamp BETWEEN datetime(?, '-1 seconds') AND datetime(?, '+1 seconds')
                          AND direcao = ? AND preco_entrada = ? AND tempo_expiracao = ?
                """, (sinal['ativo'], sinal['momento_entrada'].strftime('%Y-%m-%d %H:%M:%S'),
                      sinal['momento_entrada'].strftime('%Y-%m-%d %H:%M:%S'),
                      sinal['direcao'], sinal['preco_entrada'], sinal['tempo_expiracao']))

                if cursor.fetchone():
                    self.logger.warning(f"Sinal já registrado para {sinal['ativo']} no mesmo momento e direção.")
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Erro ao registrar sinal: {str(e)}")
            return None
    
    async def get_sinais_sem_resultado(self) -> List[Dict]:
        """Recupera sinais pendentes de forma assíncrona"""
        try:
            with self.pool.get_connection() as conn:
                query = '''
                    SELECT DISTINCT s.* 
                    FROM sinais s
                    WHERE s.resultado IS NULL 
                    AND s.processado = 0
                    AND datetime(s.timestamp, '+' || s.tempo_expiracao || ' minutes') <= datetime('now')
                    AND s.timestamp >= datetime('now', '-1 day')
                    ORDER BY s.timestamp DESC
                '''
                
                cursor = conn.cursor()
                cursor.execute(query)

                sinais = []
                for row in cursor.fetchall():
                    sinal = dict(row)
                    if isinstance(sinal['timestamp'], str):
                        sinal['timestamp'] = datetime.strptime(
                            sinal['timestamp'], 
                            '%Y-%m-%d %H:%M:%S'
                        )
                    sinais.append(sinal)
                
                return sinais

        except Exception as e:
            self.logger.error(f"Erro ao recuperar sinais pendentes: {str(e)}")
            return []
    
    async def salvar_precos_novos(self, ativo: str, dados: pd.DataFrame) -> bool:
        """Salva preços com otimização para 1min"""
        try:
            if dados.empty:
                return False

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Batch insert/update
                batch_size = 1000
                for i in range(0, len(dados), batch_size):
                    batch = dados.iloc[i:i+batch_size]

                    values = []
                    for timestamp, row in batch.iterrows():
                        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        values.append((
                            ativo,
                            timestamp_str,
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            float(row['Volume'])
                        ))

                    cursor.executemany("""
                        INSERT INTO precos (
                            ativo, timestamp, open, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ativo, timestamp) DO UPDATE SET
                            open=excluded.open,
                            high=excluded.high,
                            low=excluded.low,
                            close=excluded.close,
                            volume=excluded.volume
                    """, values)

                conn.commit()
                self.logger.info(f"Salvos {len(dados)} registros para {ativo}")
                return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar preços: {str(e)}")
            return False
    
    # Correção da função salvar_precos
    async def salvar_precos(self, ativo: str, dados: pd.DataFrame) -> bool:
        """Salva dados de preços no banco de dados de forma assíncrona"""
        try:
            # Verifica colunas necessárias
            colunas_requeridas = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in dados.columns for col in colunas_requeridas):
                self.logger.error(f"Erro: DataFrame deve conter as colunas: {colunas_requeridas}")
                return False
    
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
    
                # Prepara dados para inserção
                rows = []
                for timestamp, row in dados.iterrows():
                    # Converte timestamp para string
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    rows.append((
                        ativo,
                        timestamp_str,
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        float(row['Volume'])
                    ))
    
                # Insere dados em lote
                cursor.executemany('''
                    INSERT OR REPLACE INTO precos (
                        ativo, timestamp, open, high, low, close, volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', rows)
    
                conn.commit()
                return True
    
        except Exception as e:
            self.logger.error(f"Erro ao salvar preços para {ativo}: {str(e)}")
            return False  
    # Correção da função get_dados_historicos
    async def get_dados_historicos(self, dias: int = 30) -> pd.DataFrame:
        """Recupera dados históricos do banco de dados"""
        try:
            with self.pool.get_connection() as conn:
                query = '''
                    SELECT 
                        timestamp,
                        ativo,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM precos
                    WHERE timestamp >= datetime('now', ? || ' days')
                    AND timestamp <= datetime('now')
                    ORDER BY timestamp ASC
                '''

                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(-dias,),
                    parse_dates=['timestamp'],
                    index_col='timestamp'
                )

            if not df.empty:
                self.logger.info(f"Dados históricos recuperados: {len(df)} registros")
                return df
            else:
                self.logger.warning("Nenhum dado histórico encontrado")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Erro ao recuperar dados históricos: {str(e)}")
            return pd.DataFrame()
        
    async def get_dados_treino(self) -> pd.DataFrame:
        """Recupera dados de treino otimizados para ML"""
        try:
            with self.pool.get_connection() as conn:
                self.logger.debug("Iniciando recuperação de dados de treino")

                query = """
                    WITH ultimos_dados AS (
                        SELECT 
                            timestamp,
                            ativo,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            ROW_NUMBER() OVER (PARTITION BY ativo ORDER BY timestamp DESC) as rn
                        FROM precos
                        WHERE timestamp >= datetime('now', '-30 days')
                    )
                    SELECT 
                        timestamp,
                        ativo,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM ultimos_dados
                    WHERE rn <= 43200  -- Últimos 30 dias em minutos
                    ORDER BY timestamp ASC
                """

                df = pd.read_sql_query(
                    query,
                    conn,
                    parse_dates=['timestamp']
                )

                if not df.empty:
                    # Pivota dados para formato ML
                    df_pivot = df.pivot(
                        index='timestamp',
                        columns='ativo',
                        values=['open', 'high', 'low', 'close', 'volume']
                    )

                    # Achata os níveis das colunas
                    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]

                    self.logger.info(f"Dados de treino recuperados: {len(df_pivot)} registros")
                    return df_pivot

                self.logger.warning("Nenhum dado de treino encontrado")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Erro ao recuperar dados de treino: {str(e)}")
            return pd.DataFrame()
    
    async def get_operacoes_periodo(self, inicio: str, fim: str) -> List[Dict]:
        """Recupera operações em um período específico"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                query = '''
                    SELECT s.*,
                        julianday(data_processamento) - julianday(timestamp) as duracao
                    FROM sinais s
                    WHERE strftime('%H:%M', timestamp) BETWEEN ? AND ?
                    AND resultado IS NOT NULL
                    AND timestamp >= datetime('now', '-30 days')
                    ORDER BY timestamp DESC
                '''

                cursor.execute(query, (inicio, fim))

                operacoes = []
                for row in cursor.fetchall():
                    operacao = dict(row)
                    # Converte timestamp para datetime se necessário
                    if isinstance(operacao['timestamp'], str):
                        operacao['timestamp'] = datetime.strptime(
                            operacao['timestamp'],
                            '%Y-%m-%d %H:%M:%S'
                        )
                    operacoes.append(operacao)

                return operacoes

        except Exception as e:
            self.logger.error(f"Erro ao recuperar operações do período: {str(e)}")
            return []

    async def salvar_analise_completa(self, sinal: Dict) -> bool:
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                dados_analise = {
                    'metricas_ml': {
                        'probabilidade': sinal['ml_prob'],
                        'score': sinal.get('score', 0),
                    },
                    'analise_tecnica': {
                        'padroes_forca': sinal['padroes_forca'],
                        'tendencia': sinal['tendencia'],
                        'tech_score': sinal.get('tech_score', 0)
                    },
                    'analise_mercado': {
                        'volatilidade': sinal['volatilidade'],
                        'momento_score': sinal.get('momento_score', 0)
                    }
                }

                # Corrigir query
                cursor.execute("""
                    INSERT INTO analises_detalhadas (
                        sinal_id, dados_analise, timestamp
                    ) VALUES (?, ?, ?)
                """, (
                    sinal['id'],
                    json.dumps(dados_analise),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar análise detalhada: {str(e)}")
            return False