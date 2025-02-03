import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import threading
from pathlib import Path
from queue import Queue
from contextlib import contextmanager
import asyncio
from datetime import datetime
import traceback

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
        connection = None
        try:
            connection = self.connections.get(timeout=5)
            yield connection
        finally:
            if connection:
                try:
                    self.connections.put(connection)
                except:
                    connection.close()

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
        """Inicializa o banco de dados com as tabelas necessárias"""
        try:
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
                        assertividade REAL NOT NULL DEFAULT 50.0,
                        resultado TEXT,

                        tendencia TEXT,
                        forca_tendencia REAL,
                        confianca_tendencia REAL,
                    
                        padroes TEXT,
                        forca_padroes REAL,
                    
                        confirmacoes TEXT,
                        peso_validacao REAL,
                    
                        ranking_score REAL,
                        ranking_classificacao TEXT,
                        ranking_confianca REAL,
                        ranking_recomendacao TEXT,
                    
                        detalhes_validacao TEXT,
                        detalhes_tendencia TEXT,
                        ranking_detalhes TEXT,
                    
                        processado BOOLEAN DEFAULT 0,
                        data_processamento DATETIME,

                        UNIQUE(ativo, timestamp, direcao, preco_entrada, tempo_expiracao)
                    )
                ''')     
            
                # Índices otimizados
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_ativo_date ON precos(ativo, date_partition, hour_partition)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_timestamp_range ON precos(timestamp, ativo)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_timestamp ON sinais(timestamp, ativo)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_processado ON sinais(processado, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_volume ON precos(volume) WHERE volume > 0')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_precos_timestamp_ativo ON precos(timestamp, ativo)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sinais_resultado ON sinais(resultado) WHERE resultado IS NOT NULL')

                # Configurações para evitar locks
                cursor.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging
                cursor.execute('PRAGMA busy_timeout=5000')  # 5 segundos de timeout
                cursor.execute('PRAGMA synchronous=NORMAL')  # Menos rigoroso com sync

                conn.commit()
                self.logger.info("Banco de dados inicializado com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def get_taxa_sucesso_horario_ativo(self, hora: int, ativo: str) -> float:
       """Retorna taxa de sucesso para horário e ativo específicos"""
       try:
           with self.pool.get_connection() as conn:
               cursor = conn.cursor()

               query = """
                   SELECT 
                       COUNT(CASE WHEN resultado = 'WIN' THEN 1 END) * 100.0 / COUNT(*) as win_rate
                   FROM sinais
                   WHERE strftime('%H', timestamp) = ?
                   AND ativo = ?
                   AND processado = 1
                   AND datetime(timestamp) >= datetime('now', '-30 days')
               """

               cursor.execute(query, (str(hora).zfill(2), ativo))
               resultado = cursor.fetchone()

               return float(resultado[0]) if resultado and resultado[0] else 50.0

       except Exception as e:
           self.logger.error(f"Erro ao obter taxa de sucesso: {str(e)}")
           return 50.0     
 
    async def limpar_dados_antigos(self, dias: int = 90) -> None:
        """Limpa dados mais antigos que X dias"""
        try:
            self.logger.info(f"Iniciando limpeza de dados antigos ({dias} dias)")
            
            data_limite = datetime.now() - timedelta(days=dias)
            
            # Limpa tabela de preços
            query_precos = """
            DELETE FROM precos 
            WHERE timestamp < :data_limite
            """
            await self.execute(query_precos, {'data_limite': data_limite})
            
            # Limpa tabela de sinais
            query_sinais = """
            DELETE FROM sinais 
            WHERE timestamp < :data_limite
            """
            await self.execute(query_sinais, {'data_limite': data_limite})
            
            self.logger.info(f"Limpeza de dados antigos concluída")
            
        except Exception as e:
            self.logger.error(f"Erro ao limpar dados antigos: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        
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
         
    async def valida_sinal_repetido(self, sinal: Dict) -> Optional[int]:
        """Valida sinal repetido com mesmo ativo, direcao, preco_entrada e timestamp"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Verifica duplicidade
                cursor.execute("""
                    SELECT id FROM sinais
                    WHERE ativo = ? AND 
                          timestamp BETWEEN datetime(?, '-1 seconds') AND datetime(?, '+1 seconds')
                          AND direcao = ? AND preco_entrada = ? AND tempo_expiracao = ?
                """, (sinal['ativo'], sinal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                      sinal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                      sinal['direcao'], sinal['preco_entrada'], sinal['tempo_expiracao']))

                if cursor.fetchone():
                    self.logger.warning(f"Sinal já registrado para {sinal['ativo']} no mesmo momento e direção.")
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Erro ao validar sinal repetido: {str(e)}")
            return None
    
    async def get_sinais_sem_resultado(self):
        try:
            query = """
            SELECT id, ativo, direcao, 
                   strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp, 
                   tempo_expiracao, preco_entrada
            FROM sinais 
            WHERE processado = 0 
            AND timestamp <= datetime('now', '-' || tempo_expiracao || ' minutes')
            """
            
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                sinais = cursor.fetchall()
                
                return [{
                    'id': s[0],
                    'ativo': s[1],
                    'direcao': s[2],
                    'timestamp': datetime.strptime(s[3], '%Y-%m-%d %H:%M:%S'),
                    'tempo_expiracao': s[4],
                    'preco_entrada': float(s[5])
                } for s in sinais]
                
        except Exception as e:
            self.logger.error(f"Erro ao recuperar sinais pendentes: {str(e)}")
            return []
    
    async def salvar_precos_novos(self, ativo: str, dados: pd.DataFrame) -> bool:
        """Salva apenas novos preços na tabela"""
        try:
            if dados.empty:
                self.logger.warning(f"Dados vazios para {ativo}")
                return False

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Pega último timestamp salvo
                cursor.execute("""
                    SELECT MAX(timestamp) 
                    FROM precos 
                    WHERE ativo = ?
                """, (ativo,))
                
                ultimo_registro = cursor.fetchone()[0]
                
                # Converte timestamps para UTC para garantir comparação correta
                if ultimo_registro:
                    ultimo_timestamp = pd.to_datetime(ultimo_registro).tz_localize('UTC')
                else:
                    ultimo_timestamp = pd.Timestamp.min.tz_localize('UTC')
                
                # Garante que o índice dos dados está em UTC
                try:
                    if dados.index.tz is None:
                        dados.index = dados.index.tz_localize('UTC')
                    else:
                        # Usa tz_convert independente do tipo de timezone
                        dados.index = dados.index.tz_convert('UTC')
                except Exception as e:
                    self.logger.error(f"Erro ao converter timezone: {str(e)}")
                    # Se falhar, tenta localizar diretamente
                    dados.index = pd.DatetimeIndex(dados.index).tz_localize('UTC', ambiguous='infer')
                
                # Filtra apenas dados novos
                dados_novos = dados[dados.index > ultimo_timestamp]
                
                if dados_novos.empty:
                    self.logger.debug(f"Nenhum dado novo para {ativo}")
                    return True
                    
                # Prepara valores para inserção
                values = []
                for timestamp, row in dados_novos.iterrows():
                    # Converte timestamp para string UTC
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
                
                # Insere novos dados
                cursor.executemany("""
                    INSERT INTO precos (
                        ativo, timestamp, open, high, low, close, volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ativo, timestamp) DO NOTHING
                """, values)
                
                conn.commit()
                self.logger.info(f"Salvos {len(values)} novos registros para {ativo}")
                return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar preços novos: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
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
        
    async def get_dados_recentes(self, ativo: str, minutos: int = 60) -> pd.DataFrame:
        """Obtém dados recentes com cache otimizado"""
        try:
            cache_key = f"{ativo}_recente"
            
            # Verifica cache
            if (cache_key in self.cache['dados_mercado'] and 
                (datetime.now() - self.cache_last_update.get(cache_key, datetime.min)).total_seconds() < self.cache_timeout):
                return self.cache['dados_mercado'][cache_key]
            
            with self.pool.get_connection() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM precos
                    WHERE ativo = ?
                    AND timestamp >= datetime('now', ? || ' minutes')
                    ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(ativo, -minutos),
                    parse_dates=['timestamp']
                )
                
                if not df.empty:
                    # Atualiza cache
                    self.cache['dados_mercado'][cache_key] = df
                    self.cache_last_update[cache_key] = datetime.now()
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Erro ao obter dados recentes: {str(e)}")
            return pd.DataFrame()
        
    async def salvar_sinal(self, sinal: Dict) -> bool:
        """Salva sinal no banco de dados"""
        try:
            query = """
 INSERT INTO sinais (
            ativo, direcao, timestamp, tempo_expiracao, 
            preco_entrada, score, assertividade,
            tendencia, forca_tendencia, confianca_tendencia,
            padroes, forca_padroes,
            confirmacoes, peso_validacao,
            ranking_score, ranking_classificacao, 
            ranking_confianca, ranking_recomendacao,
            detalhes_validacao, detalhes_tendencia, ranking_detalhes
        ) VALUES (
            :ativo, :direcao, :timestamp, :tempo_expiracao,
            :preco_entrada, :score, :assertividade,
            :tendencia, :forca_tendencia, :confianca_tendencia,
            :padroes, :forca_padroes,
            :confirmacoes, :peso_validacao,
            :ranking_score, :ranking_classificacao,
            :ranking_confianca, :ranking_recomendacao,
            :detalhes_validacao, :detalhes_tendencia, :ranking_detalhes
            )
            """
            
            params = {
            'ativo': sinal['ativo'],
            'direcao': sinal['direcao'],
            'timestamp': sinal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'tempo_expiracao': int(sinal['tempo_expiracao']),
            'preco_entrada': float(sinal['preco_entrada']),
            'score': float(sinal['score']),
            'assertividade': float(sinal['assertividade']),
            
            # Campos de tendência
            'tendencia': sinal.get('tendencia'),
            'forca_tendencia': float(sinal.get('forca_tendencia', 0)),
            'confianca_tendencia': float(sinal.get('confianca_tendencia', 0)),
            
            # Campos de padrões
            'padroes': json.dumps(sinal.get('padroes', [])),
            'forca_padroes': float(sinal.get('forca_padroes', 0)),
            
            # Campos de validação
            'confirmacoes': json.dumps(sinal.get('confirmacoes', [])),
            'peso_validacao': float(sinal.get('peso_validacao', 0)),
            
            # Campos de ranking
            'ranking_score': float(sinal.get('ranking_score', 0)),
            'ranking_classificacao': sinal.get('ranking_classificacao'),
            'ranking_confianca': float(sinal.get('ranking_confianca', 0)),
            'ranking_recomendacao': sinal.get('ranking_recomendacao'),
            
            # Campos de detalhes
            'detalhes_validacao': json.dumps(sinal.get('detalhes_validacao', {})),
            'detalhes_tendencia': json.dumps(sinal.get('detalhes_tendencia', {})),
            'ranking_detalhes': json.dumps(sinal.get('ranking_detalhes', {}))
            }
            
            return await self.execute(query, params)
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar sinal: {str(e)}")
            self.logger.error(f"Sinal recebido: {sinal}")
            return False
        
    async def execute(self, query: str, params: dict = None) -> bool:
        """Executa query com retry automático"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao executar query: {str(e)}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Params: {params}")
            return False

    async def get_dados_periodo(self, ativo: str, inicio: datetime, fim: datetime) -> pd.DataFrame:
        """Obtém dados de um período específico"""
        try:
            with self.pool.get_connection() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM precos
                    WHERE ativo = ?
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(
                        ativo,
                        inicio.strftime('%Y-%m-%d %H:%M:%S'),
                        fim.strftime('%Y-%m-%d %H:%M:%S')
                    ),
                    parse_dates=['timestamp']
                )
                
                return df
                
        except Exception as e:
            self.logger.error(f"Erro ao obter dados do período: {str(e)}")
            return pd.DataFrame()

    async def salvar_resultado(self, sinal_id: int, resultado: Dict) -> bool:
        """Salva resultado do sinal"""
        try:
            query = """
            UPDATE sinais 
            SET resultado = :resultado,
                preco_saida = :preco_saida,
                processado = 1,
                data_processamento = datetime('now')
            WHERE id = :id
            """
            
            params = {
                'id': sinal_id,
                'resultado': resultado['resultado'],
                'preco_saida': resultado['preco_saida']
            }
            
            self.logger.debug(f"Query update resultado: {query}")
            self.logger.debug(f"Params: {params}")
            
            return await self.execute(query, params)
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultado: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        
        
        
    async def analisar_correlacao_scores(self):
        """Analisa correlação entre scores e resultados"""
        query = """
        SELECT 
            score,
            score_tendencia,
            score_momentum,
            score_volume,
            resultado
        FROM sinais
        WHERE resultado IS NOT NULL
        AND timestamp >= datetime('now', '-30 days')
        """
        
        with self.db.pool.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            
            # Converte resultado para binário
            df['win'] = df['resultado'].apply(lambda x: 1 if x == 'WIN' else 0)
            
            # Calcula correlações
            correlacoes = df[[
                'score', 'score_tendencia', 'score_momentum', 
                'score_volume', 'win'
            ]].corr()['win'].sort_values(ascending=False)
            
            self.logger.info(f"Correlações com resultado:\n{correlacoes}")
            
            return correlacoes

    async def ajustar_pesos_scores(self):
        """Ajusta pesos dos scores baseado na performance"""
        correlacoes = await self.analisar_correlacao_scores()
        
        # Normaliza correlações positivas para usar como pesos
        pesos = correlacoes[correlacoes > 0]
        pesos = pesos / pesos.sum()
        
        return dict(pesos)

    async def analisar_performance_por_qualidade(self):
        """Analisa taxa de acerto por nível de qualidade"""
        query = """
        SELECT 
            score_qualidade,
            COUNT(*) as total,
            SUM(CASE WHEN resultado = 'WIN' THEN 1 ELSE 0 END) as wins,
            AVG(CASE WHEN resultado = 'WIN' THEN 1 ELSE 0 END) * 100 as win_rate
        FROM sinais
        WHERE resultado IS NOT NULL
        AND timestamp >= datetime('now', '-30 days')
        GROUP BY score_qualidade
        ORDER BY win_rate DESC
        """
        
        with self.db.pool.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            self.logger.info(f"Performance por qualidade:\n{df}")
            return df


    async def analisar_performance_detalhada(self):
        try:
            """Analisa performance considerando todos os aspectos do sinal"""
            query = """
            SELECT 
                resultado,
                ranking_classificacao,
                ranking_score,
                ranking_confianca,
                forca_tendencia,
                peso_validacao
                FROM sinais
                WHERE resultado IS NOT NULL
            AND timestamp >= datetime('now', '-30 days')
            """

            with self.pool.get_connection() as conn:
                df = pd.read_sql_query(query, conn)

                # Análise por classificação
                performance_classificacao = df.groupby('ranking_classificacao').agg({
                    'resultado': lambda x: (x == 'WIN').mean() * 100
                }).rename(columns={'resultado': 'win_rate'})

                # Análise por faixa de score
                df = df.dropna(subset=['ranking_score'])

                if df['ranking_score'].nunique() >= 5:
                    df['faixa_score'] = pd.qcut(df['ranking_score'], q=5)
                else:
                    self.logger.warning("Poucos valores distintos em ranking_score para aplicar qcut")
                    df['faixa_score'] = pd.cut(df['ranking_score'], bins=5)

                performance_score = df.groupby('faixa_score').agg({
                    'resultado': lambda x: (x == 'WIN').mean() * 100
                })

                return performance_classificacao, performance_score

        except Exception as e:
            self.logger.error(f"Erro ao analisar performance detalhada: {str(e)}")
            return {}, {}