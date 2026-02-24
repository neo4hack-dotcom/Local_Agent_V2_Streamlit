from __future__ import annotations

from copy import deepcopy

from .models import AgentConfigCreate, AgentTemplate

_TEMPLATE_DATA = [
    {
        "id": "sql_analyst",
        "name": "SQL Analyst",
        "description": "Generate and execute SQL on ClickHouse/Oracle to answer data questions.",
        "defaults": {
            "name": "SQL Analyst",
            "agent_type": "sql_analyst",
            "description": "SQL analytics agent for structured database queries.",
            "system_prompt": "You are a reliable SQL expert and data analyst.",
            "sql_prompt_template": (
                "You must produce a valid SQL query that answers the question.\\n"
                "Constraint: return SQL only (no explanation).\\n"
                "User question: {question}\\n"
                "Allowed tables: {allowed_tables}\\n"
                "Available schema:\\n{schema}\\n"
            ),
            "answer_prompt_template": (
                "Question: {question}\\n"
                "Executed SQL: {sql}\\n"
                "Raw result (JSON): {rows}\\n"
                "Produce a concise, business-oriented answer in English."
            ),
            "allowed_tables": [],
            "max_rows": 200,
            "template_config": {
                "database_id": "",
                "database_name": "",
                "sql_use_case_mode": "llm_sql",
                "sql_query_template": "",
                "sql_parameters": [],
            },
            "enabled": True,
        },
    },
    {
        "id": "clickhouse_table_manager",
        "name": "ClickHouse Table Manager",
        "description": "Create and update ClickHouse tables with configurable safety guards.",
        "defaults": {
            "name": "ClickHouse Table Manager",
            "agent_type": "clickhouse_table_manager",
            "description": (
                "Handles ClickHouse table creation and data write operations "
                "with configurable safety controls."
            ),
            "system_prompt": (
                "You are a ClickHouse data engineer. Produce safe, valid ClickHouse SQL only "
                "and follow the configured safety policy exactly."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "User request: {question}\\n"
                "Executed SQL statements:\\n{sql}\\n"
                "Execution trace: {rows}\\n"
                "Provide a concise operational summary, including what was created/updated and any failures."
            ),
            "allowed_tables": [],
            "max_rows": 100,
            "template_config": {
                "database_id": "",
                "database_name": "",
                "protect_existing_tables": True,
                "allow_row_inserts": True,
                "allow_row_updates": True,
                "allow_row_deletes": False,
                "max_statements": 8,
                "preview_select_rows": 100,
                "stop_on_error": True,
            },
            "enabled": True,
        },
    },
    {
        "id": "unstructured_to_structured",
        "name": "Unstructured -> Structured",
        "description": "Extract structured JSON from free text.",
        "defaults": {
            "name": "Unstructured Extractor",
            "agent_type": "unstructured_to_structured",
            "description": "Extracts key structured fields from unstructured content.",
            "system_prompt": "You extract structured information accurately and return valid JSON.",
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Input text: {question}\\n"
                "Schema definition: {schema}\\n"
                "Return a JSON object matching this schema."
            ),
            "allowed_tables": [],
            "max_rows": 1,
            "template_config": {
                "output_schema": {
                    "summary": "string",
                    "entities": [{"type": "string", "value": "string"}],
                    "priority": "low|medium|high",
                },
                "strict_json": True,
            },
            "enabled": True,
        },
    },
    {
        "id": "email_cleaner",
        "name": "Email Cleaner",
        "description": "Remove noise from emails and keep only essentials.",
        "defaults": {
            "name": "Email Cleaner",
            "agent_type": "email_cleaner",
            "description": "Condenses long emails into key points and actions.",
            "system_prompt": "You clean noisy emails and keep only high-value information.",
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Email content: {question}\\n"
                "Return:\\n"
                "1) Summary\\n"
                "2) Action items\\n"
                "3) Deadlines\\n"
                "4) Risks/Blockers"
            ),
            "allowed_tables": [],
            "max_rows": 1,
            "template_config": {
                "max_bullets": 8,
                "include_sections": [
                    "summary",
                    "action_items",
                    "deadlines",
                    "risks",
                ],
            },
            "enabled": True,
        },
    },
    {
        "id": "file_assistant",
        "name": "File Assistant",
        "description": "Read files from a configured folder and answer questions.",
        "defaults": {
            "name": "File Assistant",
            "agent_type": "file_assistant",
            "description": "Searches and reasons over documents from a configured folder.",
            "system_prompt": "You are a document analyst. Ground answers on provided file context.",
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Question: {question}\\n"
                "Context snippets:\\n{rows}\\n"
                "Answer only from provided context and cite file names."
            ),
            "allowed_tables": [],
            "max_rows": 25,
            "template_config": {
                "folder_path": "",
                "file_extensions": [".txt", ".md", ".json", ".csv", ".log"],
                "max_files": 40,
                "max_file_size_kb": 400,
                "top_k": 6,
            },
            "enabled": True,
        },
    },
    {
        "id": "text_file_manager",
        "name": "Text File Manager",
        "description": "Open/read/create/edit plain text files from a configured folder.",
        "defaults": {
            "name": "Text File Manager",
            "agent_type": "text_file_manager",
            "description": "Performs text file operations (read, create, write, append, list).",
            "system_prompt": (
                "You are a text file operations assistant. "
                "Use file actions precisely and report the exact file path used."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "User request: {question}\\n"
                "File operation output: {rows}\\n"
                "Provide a concise execution summary."
            ),
            "allowed_tables": [],
            "max_rows": 200,
            "template_config": {
                "folder_path": "",
                "default_file_path": "notes.txt",
                "default_encoding": "utf-8",
                "auto_create_folder": True,
                "allow_overwrite": True,
                "max_chars_read": 12000,
            },
            "enabled": True,
        },
    },
    {
        "id": "excel_manager",
        "name": "Excel Manager",
        "description": "Create, read, edit and append data in Excel workbooks (.xlsx).",
        "defaults": {
            "name": "Excel Manager",
            "agent_type": "excel_manager",
            "description": "Handles Excel workbook operations for creation, updates and reads.",
            "system_prompt": (
                "You are an Excel operations assistant. "
                "Use only supported actions and confirm changes with sheet/file details."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "User request: {question}\\n"
                "Workbook operation output: {rows}\\n"
                "Provide a concise execution summary."
            ),
            "allowed_tables": [],
            "max_rows": 500,
            "template_config": {
                "folder_path": "",
                "workbook_path": "workbook.xlsx",
                "default_sheet": "Sheet1",
                "auto_create_folder": True,
                "auto_create_workbook": True,
                "max_rows_read": 200,
            },
            "enabled": True,
        },
    },
    {
        "id": "word_manager",
        "name": "Word Manager",
        "description": "Create, read and edit Word documents (.docx) in a configured folder.",
        "defaults": {
            "name": "Word Manager",
            "agent_type": "word_manager",
            "description": "Handles Word document operations for creation, updates and reads.",
            "system_prompt": (
                "You are a Word document operations assistant. "
                "Use supported actions and confirm file path and paragraph-level changes."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "User request: {question}\\n"
                "Word document operation output: {rows}\\n"
                "Provide a concise execution summary."
            ),
            "allowed_tables": [],
            "max_rows": 300,
            "template_config": {
                "folder_path": "",
                "document_path": "document.docx",
                "auto_create_folder": True,
                "auto_create_document": True,
                "allow_overwrite": True,
                "max_paragraphs_read": 80,
            },
            "enabled": True,
        },
    },
    {
        "id": "elasticsearch_retriever",
        "name": "Elasticsearch Retriever",
        "description": "Fetch evidence from Elasticsearch and summarize it.",
        "defaults": {
            "name": "Elasticsearch Retriever",
            "agent_type": "elasticsearch_retriever",
            "description": "Retrieves relevant documents from Elasticsearch indexes.",
            "system_prompt": "You retrieve and summarize factual evidence from Elasticsearch results.",
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Question: {question}\\n"
                "Search results: {rows}\\n"
                "Produce a concise, evidence-based answer."
            ),
            "allowed_tables": [],
            "max_rows": 20,
            "template_config": {
                "database_id": "",
                "database_name": "",
                "base_url": "http://localhost:9200",
                "index": "",
                "api_key": "",
                "username": "",
                "password": "",
                "verify_ssl": True,
                "top_k": 5,
                "fields": ["*"],
            },
            "enabled": True,
        },
    },
    {
        "id": "rag_context",
        "name": "RAG Context Agent",
        "description": "Retrieve business context from documents before answering.",
        "defaults": {
            "name": "RAG Context Agent",
            "agent_type": "rag_context",
            "description": "Brings external business context from files for grounded answers.",
            "system_prompt": "You are a business assistant that answers using retrieved context chunks.",
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Question: {question}\\n"
                "Retrieved context: {rows}\\n"
                "Answer with citations to sources."
            ),
            "allowed_tables": [],
            "max_rows": 30,
            "template_config": {
                "folder_path": "",
                "file_extensions": [".txt", ".md", ".json", ".csv"],
                "top_k_chunks": 6,
                "chunk_size": 1200,
                "chunk_overlap": 150,
                "max_files": 50,
            },
            "enabled": True,
        },
    },
    {
        "id": "rss_news",
        "name": "RSS / News Briefing Agent",
        "description": "Fetch RSS news from favorite sources, filter by interests and deliver a short briefing.",
        "defaults": {
            "name": "RSS / News Briefing Agent",
            "agent_type": "rss_news",
            "description": (
                "Builds a concise morning news briefing from RSS feeds "
                "filtered by user interests."
            ),
            "system_prompt": (
                "You are a concise newsroom assistant. "
                "Create factual, source-grounded briefings and avoid speculation."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "User request: {question}\\n"
                "Selected news items: {rows}\\n"
                "Produce a short breakfast briefing with key takeaways and source links."
            ),
            "allowed_tables": [],
            "max_rows": 20,
            "template_config": {
                "feed_urls": [
                    "https://www.lemonde.fr/rss/une.xml",
                    "https://www.franceinfo.fr/titres.rss",
                    "https://www.lefigaro.fr/rss/figaro_actualites.xml",
                    "https://www.rfi.fr/fr/rss",
                ],
                "interests": ["economie", "ia", "technologie", "geopolitique"],
                "exclude_keywords": [],
                "top_k": 5,
                "max_items_per_feed": 25,
                "hours_lookback": 24,
                "language_hint": "fr",
                "include_general_if_no_match": True,
            },
            "enabled": True,
        },
    },
    {
        "id": "web_scraper",
        "name": "Web Scraper Agent",
        "description": "Scrape configured websites/pages and extract useful content.",
        "defaults": {
            "name": "Web Scraper Agent",
            "agent_type": "web_scraper",
            "description": "Fetches selected web pages and summarizes extracted content.",
            "system_prompt": (
                "You are a web scraping analyst. Use only scraped content and cite source URLs."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Question: {question}\\n"
                "Scraped pages: {rows}\\n"
                "Answer with key facts and source links."
            ),
            "allowed_tables": [],
            "max_rows": 20,
            "template_config": {
                "start_urls": [],
                "include_urls_from_question": True,
                "search_fallback": True,
                "follow_links": False,
                "same_domain_only": True,
                "allowed_domains": [],
                "max_pages": 3,
                "max_links_per_page": 10,
                "max_chars_per_page": 6000,
                "timeout_seconds": 20,
                "region": "wt-wt",
                "safe_search": "moderate",
            },
            "enabled": True,
        },
    },
    {
        "id": "web_navigator",
        "name": "Web Navigator Agent",
        "description": "Navigate websites step by step (open pages, click, fill forms).",
        "defaults": {
            "name": "Web Navigator Agent",
            "agent_type": "web_navigator",
            "description": (
                "Interactive browser agent that can navigate, click and fill forms "
                "to accomplish web tasks."
            ),
            "system_prompt": (
                "You are a reliable browser automation assistant. "
                "Choose safe actions and explain blockers clearly."
            ),
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Task: {question}\\n"
                "Navigation trace: {rows}\\n"
                "Provide concise status and outcome."
            ),
            "allowed_tables": [],
            "max_rows": 50,
            "template_config": {
                "start_url": "",
                "headless": True,
                "max_steps": 8,
                "timeout_ms": 15000,
                "capture_html_chars": 7000,
            },
            "enabled": True,
        },
    },
    {
        "id": "wikipedia_retriever",
        "name": "Wikipedia Agent",
        "description": "Retrieve and summarize information from Wikipedia.",
        "defaults": {
            "name": "Wikipedia Agent",
            "agent_type": "wikipedia_retriever",
            "description": "Searches Wikipedia and returns grounded summaries with references.",
            "system_prompt": "You are a Wikipedia research assistant. Cite page links in your answer.",
            "sql_prompt_template": "",
            "answer_prompt_template": (
                "Question: {question}\\n"
                "Wikipedia pages: {rows}\\n"
                "Produce a factual answer with citations."
            ),
            "allowed_tables": [],
            "max_rows": 10,
            "template_config": {
                "language": "en",
                "top_k": 5,
                "summary_sentences": 2,
            },
            "enabled": True,
        },
    },
]


def list_agent_templates() -> list[AgentTemplate]:
    templates: list[AgentTemplate] = []
    for item in _TEMPLATE_DATA:
        defaults = AgentConfigCreate.model_validate(deepcopy(item["defaults"]))
        templates.append(
            AgentTemplate(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                defaults=defaults,
            )
        )
    return templates


def template_defaults(template_id: str) -> AgentConfigCreate:
    for template in list_agent_templates():
        if template.id == template_id:
            return template.defaults
    raise ValueError(f"Unknown template id: {template_id}")
