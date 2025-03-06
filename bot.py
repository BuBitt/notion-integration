import os
import time
import glob
import json
import aiohttp
import asyncio
import logging
import requests
import polars as pl

from functools import partial
from dotenv import load_dotenv
from rich.console import Console
from colorlog import ColoredFormatter
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configurar console do Rich
console = Console()

# Criar pastas se não existirem
os.makedirs("logs", exist_ok=True)
os.makedirs("caches", exist_ok=True)

# Configurar logging com cores
log_filename = os.path.join(
    "logs", f"notion_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s%(reset)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
file_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Configurar Polars
pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_rows(30)

# Arquivos para caches persistentes
PAGE_CACHE_FILE = os.path.join("caches", "page_cache.json")
MATERIA_CACHE_FILE = os.path.join("caches", "materia_cache.json")


# Função para verificar e atualizar o cache
def check_and_update_cache(file_path, cache_name, max_age_days=1):
    if os.path.exists(file_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if (datetime.now() - mod_time).days > max_age_days:
            logger.info(
                f"Cache {cache_name} está desatualizado (mais de {max_age_days} dia). Limpando."
            )
            return {}
    return load_cache(file_path, cache_name)


# Funções para carregar e salvar caches
def load_cache(file_path, cache_name):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                cache = json.load(f)
                logger.info(
                    f"Cache {cache_name} carregado de {file_path} com {len(cache)} itens"
                )
                return cache
        except Exception as e:
            logger.error(f"Erro ao carregar cache {cache_name} de {file_path}: {e}")
            return {}
    return {}


def save_cache(cache, file_path, cache_name):
    try:
        with open(file_path, "w") as f:
            json.dump(cache, f)
        logger.info(f"Cache {cache_name} salvo em {file_path} com {len(cache)} itens")
    except Exception as e:
        logger.error(f"Erro ao salvar cache {cache_name} em {file_path}: {e}")


# Função para limpar logs antigos
def clean_old_logs(max_age_days=7):
    log_files = glob.glob("logs/notion_sync_*.log")
    current_time = time.time()
    for log_file in log_files:
        file_time = os.path.getmtime(log_file)
        if (current_time - file_time) > (
            max_age_days * 24 * 60 * 60
        ):  # Converter dias para segundos
            try:
                os.remove(log_file)
                logger.info(f"Arquivo de log antigo removido: {log_file}")
            except Exception as e:
                logger.error(f"Erro ao remover log {log_file}: {e}")


# Carregar caches existentes com verificação de atualização
page_cache = check_and_update_cache(PAGE_CACHE_FILE, "page_cache", max_age_days=1)
materia_cache = check_and_update_cache(
    MATERIA_CACHE_FILE, "materia_cache", max_age_days=1
)

# Carregar variáveis de ambiente
load_dotenv()
notion_api_key = os.getenv("NOTION_API_KEY")
notion_database_id = os.getenv("NOTION_DATABASE_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# Função para checar a API do Notion
def check_notion_api(api_key, database_id):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    url = f"https://api.notion.com/v1/databases/{database_id}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            logger.info("API do Notion está funcionando corretamente.")
            return True
        else:
            logger.error(
                f"Falha na checagem da API: Status {response.status_code} - {response.text}"
            )
            return False
    except requests.RequestException as e:
        logger.error(f"Erro ao conectar à API do Notion: {e}")
        return False


# Checagem inicial
logger.info("Iniciando programa e checando API do Notion...")
if not notion_api_key or not notion_database_id:
    logger.error(
        "As variáveis de ambiente 'NOTION_API_KEY' e 'NOTION_DATABASE_ID' não estão definidas!"
    )
    raise ValueError("Variáveis de ambiente não definidas!")
if not check_notion_api(notion_api_key, notion_database_id):
    logger.error("Checagem da API falhou. Encerrando programa.")
    raise SystemExit("Erro na API do Notion")

logger.info("Variáveis de ambiente carregadas e API validada com sucesso!")

# Configuração da API do Notion
headers = {
    "Authorization": f"Bearer {notion_api_key}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


# Funções auxiliares
def extract_title(props, prop_name):
    try:
        return (
            props.get(prop_name, {}).get("title", [{}])[0].get("plain_text", "").strip()
        )
    except (IndexError, AttributeError):
        logger.debug(f"Erro ao extrair título de '{prop_name}'")
        return ""


def extract_checkbox(props, prop_name):
    return "Yes" if props.get(prop_name, {}).get("checkbox", False) else "No"


def extract_select(props, prop_name):
    return props.get(prop_name, {}).get("select", {}).get("name", "") or ""


async def get_notion_page_async(session, page_id, headers):
    if page_id in page_cache:
        logger.debug(f"Cache hit para página {page_id}")
        return page_cache[page_id]
    page_url = f"https://api.notion.com/v1/pages/{page_id}"
    try:
        async with session.get(page_url, headers=headers, timeout=5) as response:
            response.raise_for_status()
            data = await response.json()
            page_cache[page_id] = data
            logger.debug(f"Página {page_id} carregada e adicionada ao cache")
            return data
    except Exception as e:
        logger.warning(f"Falha ao buscar página {page_id}: {e}")
        return {}


async def extract_relation_titles_async(props, prop_name, headers):
    relations = props.get(prop_name, {}).get("relation", [])
    if not relations:
        return ""
    titles = []
    async with aiohttp.ClientSession() as session:
        for rel in relations:
            rel_id = rel["id"]
            if rel_id in materia_cache:
                logger.debug(f"Cache hit para matéria {rel_id}")
                titles.append(materia_cache[rel_id])
            else:
                try:
                    page_data = await get_notion_page_async(session, rel_id, headers)
                    title = extract_title(page_data.get("properties", {}), "Name")
                    if title:
                        materia_cache[rel_id] = title
                        logger.debug(f"Matéria {rel_id} cached: {title}")
                        titles.append(title)
                except Exception as e:
                    logger.debug(f"Erro ao processar relação {rel_id}: {e}")
    return ", ".join(titles) or "Nenhuma relação encontrada"


def run_async_extract(props, prop_name, headers):
    return asyncio.run(extract_relation_titles_async(props, prop_name, headers))


def extract_date(props, prop_name):
    value = props.get(prop_name, {}).get("date", {}).get("start", "")
    return value


def extract_rich_text(props, prop_name):
    rich_text = props.get(prop_name, {}).get("rich_text", [])
    return rich_text[0].get("text", {}).get("content", "") if rich_text else ""


today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)


def calculate_days_remaining(entrega_date):
    if not entrega_date:
        return None
    try:
        entrega_dt = datetime.fromisoformat(entrega_date)
        return (entrega_dt - today).days
    except ValueError as e:
        logger.error(f"Erro ao calcular dias restantes para '{entrega_date}': {e}")
        return None


def process_result(result, headers):
    props = result["properties"]
    entrega_date = extract_date(props, "Data de Entrega")
    return {
        "Professor": extract_title(props, "Professor"),
        "Feito?": extract_checkbox(props, "Feito?"),
        "Tipo": extract_select(props, "Tipo"),
        "Estágio": extract_select(props, "Estágio"),
        "Matéria": run_async_extract(props, "Matéria", headers),
        "Entrega": entrega_date,
        "Dias Restantes": calculate_days_remaining(entrega_date),
        "Descrição": extract_rich_text(props, "Descrição"),
        "Tópicos": run_async_extract(props, "Tópicos", headers),
    }


def process_batch(batch, headers):
    return [process_result(result, headers) for result in batch]


def fetch_notion_data(database_id, headers, page_size=100):
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    all_results = []
    cursor = None
    while True:
        payload = {"page_size": page_size}
        if cursor:
            payload["start_cursor"] = cursor
        logger.info(f"Buscando página com cursor: {cursor or 'início'}")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            all_results.extend(data.get("results", []))
            cursor = data.get("next_cursor")
            if not cursor:
                break
        except requests.RequestException as e:
            logger.error(f"Erro ao buscar página: {e}")
            raise
    return all_results


# Processamento principal
logger.info("Iniciando requisição ao Notion...")
results = fetch_notion_data(notion_database_id, headers)
logger.info("Dados obtidos com sucesso! Processando...")

# Processar em lotes
batch_size = 50
batches = [results[i : i + batch_size] for i in range(0, len(results), batch_size)]
all_rows = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_batch, batch, headers) for batch in batches]
    for future in as_completed(futures):
        all_rows.extend(future.result())

# Criar DataFrame
logger.info("Criando DataFrame...")
df = pl.DataFrame(all_rows).sort("Dias Restantes", nulls_last=True)
logger.info("DataFrame criado com sucesso!")

# Limpar logs antigos antes de prosseguir
clean_old_logs(max_age_days=7)

# Salvar caches ao final
save_cache(page_cache, PAGE_CACHE_FILE, "page_cache")
save_cache(materia_cache, MATERIA_CACHE_FILE, "materia_cache")

# Filtrar tarefas não feitas com Dias Restantes <= 7
df = df.filter((pl.col("Feito?") == "No") & (pl.col("Dias Restantes") <= 7))

# Transforma o dataframe em uma lista de dicionários
tarefas = df.to_dicts()

# Verificar se as variáveis do Telegram estão definidas
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError(
        "As variáveis 'TELEGRAM_BOT_TOKEN' e 'TELEGRAM_CHAT_ID' precisam estar definidas no .env!"
    )


# Função para formatar a data de YYYY-MM-DD para DD/MM
def formatar_data(data_str):
    data = datetime.strptime(data_str, "%Y-%m-%d")
    return data.strftime("%d/%m")


# Função para escapar todos os caracteres reservados no MarkdownV2
def escapar_markdown_v2(texto):
    caracteres_reservados = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in caracteres_reservados:
        texto = texto.replace(char, f"\\{char}")
    return texto


# Função para gerar a mensagem de uma tarefa com MarkdownV2
def gerar_mensagem_tarefa(tarefa):
    dias_restantes = tarefa.get("Dias Restantes")
    if dias_restantes not in [0, 1, 3, 7]:
        return None

    tipo = tarefa.get("Tipo", "N/D").upper()
    materia = tarefa.get("Matéria", "N/D")
    entrega = tarefa.get("Entrega", "N/D")
    descricao = tarefa.get("Descrição", "N/D") or "Sem descrição"
    data_formatada = formatar_data(entrega) if entrega != "N/D" else "N/D"
    tipo = escapar_markdown_v2(tipo)
    materia = escapar_markdown_v2(materia)
    data_formatada = escapar_markdown_v2(data_formatada)
    descricao = escapar_markdown_v2(descricao)
    topicos = tarefa.get("Tópicos", "N/D") or "Sem Tópicos"

    # Separar tópicos por quebra de linha e aplicar itálico
    topicos_formatados = "\n".join(
        [
            f"\\- _{escapar_markdown_v2(topico.strip())}_"
            for topico in topicos.split(", ")
        ]
    )
    mensagem = (
        f"*{tipo} \\- {materia}*\n"
        f"Dias Restantes: *{dias_restantes} DIA{'S' if dias_restantes > 1 else ''}*\n"
        f"Entrega: `{data_formatada}`\n"
        f"Tópicos:\n{topicos_formatados}\n"
        f"Descrição: _{descricao}_"
    )
    return mensagem


# Função para enviar mensagem ao Telegram
def enviar_mensagem_telegram(mensagem):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensagem,
        "parse_mode": "MarkdownV2",
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logger.info("Mensagem enviada ao Telegram com sucesso!")
        else:
            logger.error(
                f"Erro ao enviar mensagem ao Telegram: {response.status_code} - {response.text}"
            )
    except requests.RequestException as e:
        logger.error(f"Erro de conexão com o Telegram: {e}")


# Gerar mensagens conjuntas e enviar ao Telegram
mensagens = [gerar_mensagem_tarefa(tarefa) for tarefa in tarefas]
mensagens_validas = [msg for msg in mensagens if msg is not None]

if mensagens_validas:
    # Separador estilizado com emojis e texto
    separador = "\n\n*\\-\\-\\-\\-\\-\\-*\n\n"
    mensagem_conjunta = separador.join(mensagens_validas)
    mensagem_conjunta = f"{mensagem_conjunta}"  # Cabeçalho
    logger.debug(f"Mensagem a ser enviada ao Telegram:\n\n{mensagem_conjunta}\n")
    enviar_mensagem_telegram(mensagem_conjunta)
else:
    logger.info("Nenhuma tarefa com Dias Restantes igual a 1, 3 ou 7 encontrada.")
