# Отчёт eval — 2026-11-04

Этап 1: проверяется только поиск (LLM нет). Набор: `evals/dataset/howto_config.yaml`,
его писал независимый субагент до появления кода поиска (`evals/dataset/README.md`).

| Параметр | Значение |
|---|---|
| Коммит ассистента | d1767ed |
| Paperless | v3.4.2 (d939d873e11dd14896d1bfe23f65b1baa6513ba7) |
| Модель эмбеддингов | cloud.ru: ai-sage/Giga-Embeddings-instruct-3B, размерность 2048 |
| Инструкция к вопросу | `Instruct: Given a user question about Paperless-ngx, retrieve documentation sections or source code that answer it\nQuery: <вопрос>` |
| Индекс | 3413 кусков: changelog 763, code 2110, doc 540; построен 2026-11-04T15:19:54+00:00 |
| Кейсов | 75 (changelog 13, config 8, error 42, howto 12), прогонов 3 |
| Промпт | не используется |
| Точный поиск по тексту ошибки | выключен (--no-exact) |

## Метрики

| Метрика | Значение | Порог | Статус |
|---|---|---|---|
| recall@5 | 0.75 ± 0.00 | ≥ 0.80 | ❌ |
| MRR | 0.53 ± 0.00 | — | |
| задержка поиска p50 / p95 | 0.20 / 0.35 с | показать | |
| токены запросов к API эмбеддингов | 16824 (≈0.5421 ₽) | показать | |
| срабатывания маскера ПДн | 0 | — | |

## По категориям и ролям

Для error recall@5 — кусок кода со строкой ошибки в топ-5. «Точный поиск» — доля
прогонов, где он сработал; на howto, config и out_of_scope это ложные срабатывания.

| Группа | Кейсов | recall@5 | точный поиск |
|---|---|---|---|
| категория changelog | 13 | 0.85 | 0.00 |
| категория config | 8 | 1.00 | 0.00 |
| категория error | 42 | 0.60 | 0.00 |
| категория howto | 12 | 1.00 | 0.00 |
| категория out_of_scope | 12 | — | 0.00 |
| роль engineer | 23 | 0.58 | 0.00 |
| роль user | 64 | 0.80 | 0.00 |

## По кейсам

| id | роль | ранг по прогонам | топ-1 |
|---|---|---|---|
| changelog-001 | user | 4, 4, 4 | `docs/changelog.md#features_35` |
| changelog-002 | user | 2, 2, 2 | `docs/changelog.md#200` |
| changelog-003 | user | 1, 1, 1 | `docs/faq.md#how-do-i-install-paperless-ngx-on-raspberry-pi` |
| changelog-004 | user | —, —, — | `src/documents/views.py::TasksViewSet.active` |
| changelog-005 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_WEBSERVER_WORKERS` |
| changelog-006 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_AUTO_LOGIN_USERNAME` |
| changelog-007 | user | —, —, — | `docs/configuration.md#PAPERLESS_OCR_LANGUAGES` |
| changelog-008 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_ARCHIVE_FILE_GENERATION` |
| changelog-009 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_OCR_MODE` |
| changelog-010 | user | 1, 1, 1 | `docs/configuration.md#tika` |
| changelog-011 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_CONVERT_MEMORY_LIMIT` |
| changelog-012 | user | 2, 2, 2 | `docs/usage.md#document-searches` |
| changelog-013 | user | 4, 4, 4 | `docs/usage.md#the-consumption-directory` |
| error_report-001 | user | 3, 3, 3 | `src/documents/barcodes.py::BarcodePlugin._apply_detected_asn` |
| error_report-002 | user | —, —, — | `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` |
| error_report-003 | engineer | 3, 3, 3 | `docs/advanced_usage.md#pre-consume-script` |
| error_report-004 | user | —, —, — | `docs/configuration.md#PAPERLESS_OCR_IMAGE_DPI` |
| error_report-005 | engineer | 2, 2, 2 | `docs/troubleshooting.md#consumption-fails-with-ghostscript-pdfa-rendering-failed` |
| error_report-006 | user | 1, 1, 1 | `src/documents/barcodes.py::BarcodePlugin.detect` |
| error_report-007 | user | 2, 2, 2 | `docs/advanced_usage.md#things-that-could-go-wrong` |
| error_report-008 | user | 3, 3, 3 | `src/paperless/auth.py::PaperlessBasicAuthentication` |
| error_report-009 | user | 1, 1, 1 | `src/paperless_mail/mail.py::MailAccountHandler._handle_mail_rule` |
| error_report-010 | engineer | 5, 5, 5 | `docs/administration.md#importer` |
| error_report-011 | user | 3, 3, 3 | `docs/troubleshooting.md#classifier-error-no-training-data-available` |
| error_report-012 | engineer | —, —, — | `docs/configuration.md#PAPERLESS_TIKA_ENDPOINT` |
| error-001 | engineer | —, —, — | `src/paperless_ai/vector_store.py:L57-L61` |
| error-002 | user | —, —, — | `src/documents/barcodes.py::scannable_mime_types` |
| error-003 | user | 1, 1, 1 | `src/paperless_mail/oauth.py::PaperlessMailOAuth2Manager.refresh_account_oauth_token` |
| error-004 | user | 2, 2, 2 | `src/paperless_ai/indexing.py::update_llm_index` |
| error-005 | engineer | —, —, — | `src/documents/parsers.py::DocumentParser.get_page_count` |
| error-006 | user | 4, 4, 4 | `docs/advanced_usage.md#post-consume-script` |
| error-007 | user | 4, 4, 4 | `docs/administration.md#index` |
| error-008 | user | 1, 1, 1 | `src/documents/filters.py::TitleContentFilter.filter` |
| error-009 | engineer | —, —, — | `docs/changelog.md#bug-fixes_98` |
| error-010 | user | 4, 4, 4 | `docs/configuration.md#PAPERLESS_POST_CONSUME_SCRIPT` |
| error-011 | user | 1, 1, 1 | `src/documents/workflows/actions.py::execute_password_removal_action` |
| error-012 | user | 3, 3, 3 | `docs/configuration.md#PAPERLESS_SHARE_LINK_BUNDLE_CLEANUP_CRON` |
| error-013 | engineer | —, —, — | `docs/configuration.md#PAPERLESS_WEBHOOKS_ALLOWED_SCHEMES` |
| error-014 | user | —, —, — | `docs/changelog.md#bug-fixes_44` |
| error-015 | user | —, —, — | `docs/development.md#creating-a-date-parser-plugin` |
| error-016 | engineer | —, —, — | `src/paperless/serialisers.py::ApplicationConfigurationSerializer._require_json_object` |
| error-017 | user | 4, 4, 4 | `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` |
| error-018 | user | 1, 1, 1 | `src/documents/workflows/actions.py::execute_password_removal_action` |
| error-019 | engineer | 2, 2, 2 | `src/documents/management/commands/document_exporter.py::Command.generate_share_link_bundle_target` |
| error-020 | user | 2, 2, 2 | `src/documents/classifier.py::ClassifierModelCorruptError` |
| error-021 | user | 2, 2, 2 | `src/documents/file_handling.py::UnsafeFilePathError` |
| error-022 | engineer | —, —, — | `docs/configuration.md#PAPERLESS_MEDIA_ROOT` |
| error-023 | user | 1, 1, 1 | `src/paperless/checks.py::check_v3_minimum_upgrade_version` |
| error-024 | user | —, —, — | `docs/changelog.md#all-app-changes_60` |
| error-025 | engineer | 2, 2, 2 | `docs/configuration.md#PAPERLESS_AI_LLM_CONTEXT_SIZE` |
| error-026 | user | —, —, — | `docs/development.md#2-registering-via-entry-point` |
| error-027 | user | —, —, — | `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` |
| error-028 | engineer | —, —, — | `docs/changelog.md#dependencies_14` |
| error-029 | user | —, —, — | `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` |
| error-030 | user | 1, 1, 1 | `src/paperless/settings/custom.py::parse_dateparser_languages` |
| howto-001 | user | 1, 1, 1 | `docs/usage.md#incoming-mail` |
| howto-002 | user | 4, 4, 4 | `src/paperless_mail/mail.py::TagMailAction` |
| howto-003 | user | 5, 5, 5 | `docs/usage.md#global-search` |
| howto-004 | user | 2, 2, 2 | `docs/usage.md#sharing-documents-from-paperless-ngx` |
| howto-005 | user | 1, 1, 1 | `docs/usage.md#sharing-documents-from-paperless-ngx` |
| howto-006 | user | 2, 2, 2 | `docs/administration.md#migrating-restoring` |
| howto-007 | user | 3, 3, 3 | `docs/configuration.md#PAPERLESS_OCR_ROTATE_PAGES` |
| howto-008 | engineer | 1, 1, 1 | `docs/administration.md#backup` |
| howto-009 | engineer | 1, 1, 1 | `docs/administration.md#docker-updating` |
| howto-010 | engineer | 1, 1, 1 | `docs/administration.md#importer` |
| howto-011 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_CONSUMER_ENABLE_BARCODES` |
| howto-012 | user | 1, 1, 1 | `docs/faq.md#im-using-docker-where-are-my-documents` |
| config-001 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_OCR_LANGUAGES` |
| config-002 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_TIME_ZONE` |
| config-003 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_CONSUMER_SUBDIRS_AS_TAGS` |
| config-004 | engineer | 1, 1, 1 | `docs/configuration.md#PAPERLESS_DBENGINE` |
| config-005 | engineer | 1, 1, 1 | `docs/advanced_usage.md#file-name-handling` |
| config-006 | engineer | 1, 1, 1 | `docs/configuration.md#PAPERLESS_CONSUMER_POLLING_INTERVAL` |
| config-007 | user | 1, 1, 1 | `docs/setup.md#less-powerful-devices` |
| config-008 | user | 1, 1, 1 | `docs/configuration.md#PAPERLESS_CONSUMER_DELETE_DUPLICATES` |

## Промахи (19)

### changelog-004 — «Где посмотреть, какие фоновые задачи сейчас в очереди и что выполняется? Искал раздел Django Q в админке — не нашёл»

Ожидалось: `docs/advanced_usage.md#celery-monitoring`, `docs/configuration.md#PAPERLESS_ENABLE_FLOWER`

Найдено:
1. `src/documents/views.py::TasksViewSet.active` (0.552)
2. `src/documents/views.py::TasksViewSet.run` (0.515)
3. `src/documents/views.py::TasksViewSet.status_counts` (0.486)
4. `src/documents/models.py::PaperlessTask` (0.476)
5. `src/documents/views.py::SystemStatusView` (0.468)

### changelog-007 — «Штрихкоды-разделители на сканах часто не распознаются. Можно переключить paperless на другую библиотеку распознавания штрихкодов?»

Ожидалось: `docs/migration-v3.md#barcode-scanner-changes`

Найдено:
1. `docs/configuration.md#PAPERLESS_OCR_LANGUAGES` (0.480)
2. `docs/configuration.md#ocr` (0.476)
3. `docs/configuration.md#PAPERLESS_CONSUMER_ENABLE_BARCODES` (0.469)
4. `docs/changelog.md#dependencies_26` (0.461)
5. `docs/changelog.md#dependencies_24` (0.460)

### error_report-002 — «после переезда на новый сервер документы из папки consume не добавляются, в логе такое: [2026-11-04 09:12:44,031] [ERROR] [paperless.consumer] [a91c03e7] The following error occurred while storing document scan_0012.pdf after parsing: [Errno 13] Permission denied: '/usr/src/paperless/consume/scan_0012.pdf' помогите»

Ожидалось: 

Найдено:
1. `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` (0.658)
2. `docs/configuration.md#PAPERLESS_CONSUMPTION_DIR` (0.598)
3. `docs/usage.md#the-consumption-directory` (0.592)
4. `docs/troubleshooting.md#no-files-are-added-by-the-consumer` (0.532)
5. `docs/troubleshooting.md#operation-not-permitted` (0.515)

### error_report-004 — «кидаю картинку png, задача обработки падает. в ошибке что-то про DPI, мол в картинке нет информации о DPI и архивный pdf из неё сделать нельзя. что делать?»

Ожидалось: 

Найдено:
1. `docs/configuration.md#PAPERLESS_OCR_IMAGE_DPI` (0.521)
2. `src/paperless/parsers/tesseract.py::RasterisedDocumentParser.get_dpi` (0.507)
3. `docs/configuration.md#PAPERLESS_CONSUMER_BARCODE_DPI` (0.479)
4. `src/documents/parsers.py::rasterize_pdf_page_to_png` (0.476)
5. `src/paperless/parsers/tesseract.py::RasterisedDocumentParser.calculate_a4_dpi` (0.465)

### error_report-012 — «Включили PAPERLESS_TIKA_ENABLED для офисных файлов, docx не обрабатываются: `[2026-11-04 13:27:51,006] [ERROR] [paperless.consumer] [7d3e21b8] Error occurred while consuming document contract.docx: Could not parse /tmp/paperless/paperless-ngx4h1q2z9w/contract.docx with tika server at http://tika:9998: [Errno 111] Connection refused`. Где это бросается и что проверить?»

Ожидалось: 

Найдено:
1. `docs/configuration.md#PAPERLESS_TIKA_ENDPOINT` (0.552)
2. `docs/configuration.md#PAPERLESS_TIKA_ENABLED` (0.529)
3. `docs/configuration.md#PAPERLESS_TIKA_GOTENBERG_ENDPOINT` (0.519)
4. `docs/configuration.md#tika` (0.514)
5. `src/paperless/parsers/tika.py::TikaDocumentParser` (0.509)

### error-001 — «[2026-11-04 10:15:02,123] [ERROR] [paperless.vector_store] [3f2a9c1d] Unsupported filter column: 42»

Ожидалось: 

Найдено:
1. `src/paperless_ai/vector_store.py:L57-L61` (0.495)
2. `src/paperless_ai/indexing.py::_exclude_document_id_filter` (0.449)
3. `src/paperless_ai/indexing.py::document_id_filters` (0.445)
4. `src/paperless_ai/indexing.py::exclude_document_ids_filter` (0.440)
5. `src/documents/search/_errors.py::InvalidNumberQuery` (0.422)

### error-002 — «scan_0042.pdf: Unsupported mime type application/octet-stream»

Ожидалось: 

Найдено:
1. `src/documents/barcodes.py::scannable_mime_types` (0.542)
2. `src/paperless/parsers/tesseract.py::RasterisedDocumentParser.supported_mime_types` (0.534)
3. `src/documents/serialisers.py::PostDocumentSerializer.validate_document` (0.532)
4. `docs/faq.md#what-file-types-does-paperless-ngx-support` (0.528)
5. `src/paperless/parsers/text.py::TextDocumentParser.supported_mime_types` (0.493)

### error-005 — «document page count is unknown»

Ожидалось: 

Найдено:
1. `src/documents/parsers.py::DocumentParser.get_page_count` (0.600)
2. `src/paperless/parsers/tesseract.py::RasterisedDocumentParser.get_page_count` (0.599)
3. `src/paperless/parsers/__init__.py::ParserProtocol.get_page_count` (0.590)
4. `src/paperless/parsers/text.py::TextDocumentParser.get_page_count` (0.583)
5. `src/paperless/parsers/remote.py::RemoteDocumentParser.get_page_count` (0.561)

### error-009 — «Rule scan_0042.pdf: Error while searching folder /usr/src/paperless/me»

Ожидалось: 

Найдено:
1. `docs/changelog.md#bug-fixes_98` (0.461)
2. `docs/changelog.md#all-app-changes_99` (0.437)
3. `docs/troubleshooting.md#consumer-fails-with-a-filenotfounderror` (0.428)
4. `docs/configuration.md#PAPERLESS_CONSUMPTION_DIR` (0.406)
5. `docs/changelog.md#bug-fixes_114` (0.393)

### error-013 — «В логах paperless вот такое: «URI scheme '/usr/src/paperless/media/documents/originals/0000042.pdf' is not allowed. Allowed schemes: scan_0042.pdf». Что это значит и что делать?»

Ожидалось: 

Найдено:
1. `docs/configuration.md#PAPERLESS_WEBHOOKS_ALLOWED_SCHEMES` (0.491)
2. `docs/changelog.md#bug-fixes_98` (0.440)
3. `docs/configuration.md#PAPERLESS_FILENAME_FORMAT` (0.439)
4. `src/documents/barcodes.py::scannable_mime_types` (0.438)
5. `docs/faq.md#what-file-types-does-paperless-ngx-support` (0.435)

### error-014 — «Error while emptying trash: 42»

Ожидалось: 

Найдено:
1. `docs/changelog.md#bug-fixes_44` (0.546)
2. `docs/changelog.md#all-app-changes_44` (0.521)
3. `docs/changelog.md#dependencies_59` (0.462)
4. `docs/changelog.md#bug-fixes_75` (0.461)
5. `docs/changelog.md#bug-fixes_98` (0.458)

### error-015 — «Plugin scan_0042.pdf does not subclass DateParser.»

Ожидалось: 

Найдено:
1. `docs/development.md#creating-a-date-parser-plugin` (0.631)
2. `docs/development.md#developing-date-parser-plugins` (0.608)
3. `docs/development.md#1-implementing-the-parser-class_1` (0.601)
4. `src/documents/plugins/date_parsing/base.py::DateParserPluginBase` (0.581)
5. `docs/development.md#plugin-discovery` (0.575)

### error-016 — «[2026-11-04 10:15:02,123] [ERROR] [paperless.serialisers] [3f2a9c1d] Value must be a list»

Ожидалось: 

Найдено:
1. `src/paperless/serialisers.py::ApplicationConfigurationSerializer._require_json_object` (0.544)
2. `src/documents/serialisers.py::DocumentListSerializer` (0.488)
3. `src/paperless/serialisers.py::ApplicationConfigurationSerializer.validate_user_args` (0.468)
4. `src/documents/serialisers.py::DocumentListSerializer._validate_document_id_list` (0.463)
5. `src/documents/serialisers.py::AcknowledgeTasksViewSerializer._validate_task_id_list` (0.457)

### error-022 — «scan_0042.pdf: URI scheme not allowed in /usr/src/paperless/media/documents/originals/0000042.pdf: must be #anchor, relative path, or data:image/*»

Ожидалось: 

Найдено:
1. `docs/configuration.md#PAPERLESS_MEDIA_ROOT` (0.461)
2. `docs/configuration.md#PAPERLESS_GS_BINARY` (0.439)
3. `docs/changelog.md#bug-fixes_142` (0.436)
4. `docs/configuration.md#PAPERLESS_CONVERT_BINARY` (0.430)
5. `docs/usage.md#the-consumption-directory` (0.427)

### error-024 — «An error occurred performing bulk edit: /usr/src/paperless/media»

Ожидалось: 

Найдено:
1. `docs/changelog.md#all-app-changes_60` (0.480)
2. `docs/changelog.md#all-app-changes_76` (0.468)
3. `docs/changelog.md#bug-fixes_36` (0.467)
4. `src/documents/views.py::BulkEditView` (0.464)
5. `docs/changelog.md#all-app-changes_98` (0.462)

### error-026 — «[2026-11-04 10:15:02,123] [ERROR] [paperless.registry] [3f2a9c1d] Failed to load parser entrypoint 'scan_0042.pdf' — skipping.»

Ожидалось: 

Найдено:
1. `docs/development.md#2-registering-via-entry-point` (0.580)
2. `docs/development.md#making-custom-parsers` (0.562)
3. `src/paperless/parsers/registry.py::ParserRegistry.register_defaults` (0.547)
4. `docs/development.md#extending-paperless-ngx` (0.542)
5. `docs/advanced_usage.md#parser-plugins` (0.538)

### error-027 — «scan_0042.pdf: Could not determine PDF page size for [Errno 13] Permission denied: /usr/src/paperless/media/documents/originals/0000042.pdf»

Ожидалось: 

Найдено:
1. `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` (0.524)
2. `docs/troubleshooting.md#operation-not-permitted` (0.483)
3. `docs/troubleshooting.md#consumer-fails-with-a-filenotfounderror` (0.444)
4. `src/documents/permissions.py::PaperlessAdminPermissions` (0.441)
5. `docs/changelog.md#bug-fixes_118` (0.428)

### error-028 — «В логах paperless вот такое: «42 must be a multiple of 90». Что это значит и что делать?»

Ожидалось: 

Найдено:
1. `docs/changelog.md#dependencies_14` (0.419)
2. `docs/changelog.md#dependencies_59` (0.410)
3. `docs/changelog.md#dependencies_43` (0.402)
4. `docs/changelog.md#bug-fixes_145` (0.401)
5. `docs/changelog.md#dependencies_11` (0.401)

### error-029 — «Rule [Errno 13] Permission denied: Error while processing mail /usr/src/paperless/media/documents/originals/0000042.pd»

Ожидалось: 

Найдено:
1. `docs/troubleshooting.md#permission-denied-errors-in-the-consumption-directory` (0.575)
2. `docs/troubleshooting.md#operation-not-permitted` (0.552)
3. `src/documents/permissions.py::PaperlessAdminPermissions` (0.496)
4. `src/paperless_mail/serialisers.py::MailRuleSerializer.validate_account` (0.489)
5. `docs/changelog.md#bug-fixes_118` (0.483)
