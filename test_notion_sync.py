import os
import json
import unittest
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime, timedelta
import polars as pl
from bot import (  # Ajustado para importar de "bot"
    load_cache,
    save_cache,
    check_and_update_cache,
    extract_title,
    extract_checkbox,
    extract_select,
    extract_date,
    extract_rich_text,
    calculate_days_remaining,
    process_result,
    formatar_data,
    escapar_markdown_v2,
    gerar_mensagem_tarefa,
)

# Configuração para testes
os.makedirs("test_caches", exist_ok=True)


class TestNotionSync(unittest.TestCase):

    def setUp(self):
        # Dados de exemplo para simular propriedades do Notion
        self.sample_props = {
            "Name": {"title": [{"plain_text": "Test Title"}]},
            "Feito?": {"checkbox": True},
            "Tipo": {"select": {"name": "Assignment"}},
            "Data de Entrega": {"date": {"start": "2025-03-10"}},
            "Descrição": {"rich_text": [{"text": {"content": "Test Description"}}]},
            "Matéria": {"relation": [{"id": "rel1"}]},
        }
        self.today = datetime(2025, 3, 6).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    # Testes para funções de cache
    def test_load_cache_existing(self):
        cache_file = "test_caches/test_cache.json"
        cache_data = {"key": "value"}
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        result = load_cache(cache_file, "test_cache")
        self.assertEqual(result, cache_data)
        os.remove(cache_file)

    def test_load_cache_nonexistent(self):
        result = load_cache("test_caches/nonexistent.json", "nonexistent")
        self.assertEqual(result, {})

    def test_save_cache(self):
        cache_file = "test_caches/save_test.json"
        cache_data = {"key": "value"}
        save_cache(cache_data, cache_file, "save_test")
        with open(cache_file, "r") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, cache_data)
        os.remove(cache_file)

    @patch("os.path.getmtime")
    def test_check_and_update_cache_outdated(self, mock_getmtime):
        cache_file = "test_caches/outdated.json"
        with open(cache_file, "w") as f:
            json.dump({"key": "value"}, f)
        mock_getmtime.return_value = (datetime.now() - timedelta(days=2)).timestamp()
        result = check_and_update_cache(cache_file, "outdated", max_age_days=1)
        self.assertEqual(result, {})
        os.remove(cache_file)

    # Testes para extração de propriedades do Notion
    def test_extract_title(self):
        result = extract_title(self.sample_props, "Name")
        self.assertEqual(result, "Test Title")

    def test_extract_title_empty(self):
        result = extract_title({}, "Name")
        self.assertEqual(result, "")

    def test_extract_checkbox(self):
        result = extract_checkbox(self.sample_props, "Feito?")
        self.assertEqual(result, "Yes")

    def test_extract_checkbox_false(self):
        props = {"Feito?": {"checkbox": False}}
        result = extract_checkbox(props, "Feito?")
        self.assertEqual(result, "No")

    def test_extract_select(self):
        result = extract_select(self.sample_props, "Tipo")
        self.assertEqual(result, "Assignment")

    def test_extract_select_empty(self):
        result = extract_select({}, "Tipo")
        self.assertEqual(result, "")

    def test_extract_date(self):
        result = extract_date(self.sample_props, "Data de Entrega")
        self.assertEqual(result, "2025-03-10")

    def test_extract_rich_text(self):
        result = extract_rich_text(self.sample_props, "Descrição")
        self.assertEqual(result, "Test Description")

    # Teste para cálculo de dias restantes
    @patch("bot.today", new=datetime(2025, 3, 6))  # Ajustado para "bot"
    def test_calculate_days_remaining(self):
        result = calculate_days_remaining("2025-03-10")
        self.assertEqual(result, 4)

    def test_calculate_days_remaining_none(self):
        result = calculate_days_remaining("")
        self.assertIsNone(result)

    # Teste para process_result (mockando run_async_extract)
    @patch("bot.run_async_extract")  # Ajustado para "bot"
    def test_process_result(self, mock_async_extract):
        mock_async_extract.return_value = "Matéria Teste"
        headers = {}
        result = process_result({"properties": self.sample_props}, headers)
        expected = {
            "Professor": "",
            "Feito?": "Yes",
            "Tipo": "Assignment",
            "Estágio": "",
            "Matéria": "Matéria Teste",
            "Entrega": "2025-03-10",
            "Dias Restantes": 4,
            "Descrição": "Test Description",
            "Tópicos": "Matéria Teste",
        }
        self.assertEqual(result["Professor"], expected["Professor"])
        self.assertEqual(result["Feito?"], expected["Feito?"])
        self.assertEqual(result["Dias Restantes"], expected["Dias Restantes"])

    # Testes para formatação de mensagens do Telegram
    def test_formatar_data(self):
        result = formatar_data("2025-03-10")
        self.assertEqual(result, "10/03")

    def test_escapar_markdown_v2(self):
        texto = "Hello *world* _test_ [link]"
        result = escapar_markdown_v2(texto)
        expected = "Hello \\*world\\* \\_test\\_ \\[link\\]"
        self.assertEqual(result, expected)

    def test_gerar_mensagem_tarefa_invalid_days(self):
        tarefa = {
            "Tipo": "Assignment",
            "Dias Restantes": 5,  # Não é 0, 1, 3 ou 7
            "Entrega": "2025-03-11",
        }
        result = gerar_mensagem_tarefa(tarefa)
        self.assertIsNone(result)

    def test_gerar_mensagem_tarefa(self):
        tarefa = {
            "Tipo": "Assignment",
            "Matéria": "Math",
            "Dias Restantes": 3,
            "Entrega": "2025-03-09",
            "Descrição": "Test desc with *special* chars",
            "Tópicos": "Topic1, Topic2",
        }
        result = gerar_mensagem_tarefa(tarefa)
        expected = (
            "*ASSIGNMENT \\- Math*\n"
            "Dias Restantes: *3 DIAS*\n"
            "Entrega: `09/03`\n"  # Corrigido: sem escaping de "/"
            "Tópicos:\n"
            "\\- _Topic1_\n"
            "\\- _Topic2_\n"
            "Descrição: _Test desc with \\*special\\* chars_"
        )
        self.assertEqual(result, expected)

    def test_gerar_mensagem_tarefa_missing_fields(self):
        tarefa = {
            "Dias Restantes": 1,  # Campo válido para gerar mensagem
        }
        result = gerar_mensagem_tarefa(tarefa)
        expected = (
            "*N/D \\- N/D*\n"  # Corrigido: sem escaping de "/"
            "Dias Restantes: *1 DIA*\n"
            "Entrega: `N/D`\n"  # Corrigido: sem escaping de "/"
            "Tópicos:\n"
            "\\- _Sem Tópicos_\n"  # Reflete o "or" no código
            "Descrição: _Sem descrição_"  # Reflete o "or" no código
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
