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
| Точный поиск по тексту ошибки | включён |

## Метрики

| Метрика | Значение | Порог | Статус |
|---|---|---|---|
| recall@5 | 0.96 ± 0.00 | ≥ 0.80 | ✅ |
| MRR | 0.86 ± 0.00 | — | |
| задержка поиска p50 / p95 | 0.20 / 0.33 с | показать | |
| токены запросов к API эмбеддингов | 16824 (≈0.5421 ₽) | показать | |
| срабатывания маскера ПДн | 0 | — | |

## По категориям и ролям

Для error recall@5 — кусок кода со строкой ошибки в топ-5. «Точный поиск» — доля
прогонов, где он сработал; на howto, config и out_of_scope это ложные срабатывания.

| Группа | Кейсов | recall@5 | точный поиск |
|---|---|---|---|
| категория changelog | 13 | 0.85 | 0.00 |
| категория config | 8 | 1.00 | 0.00 |
| категория error | 42 | 0.98 | 0.90 |
| категория howto | 12 | 1.00 | 0.00 |
| категория out_of_scope | 12 | — | 0.00 |
| роль engineer | 23 | 1.00 | 0.52 |
| роль user | 64 | 0.95 | 0.41 |

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
| error_report-001 | user | 1, 1, 1 | `src/documents/consumer.py::AsnCheckPlugin.pre_check_asn_value` |
| error_report-002 | user | 1, 1, 1 | `src/documents/consumer.py::ConsumerPlugin.run` |
| error_report-003 | engineer | 1, 1, 1 | `src/documents/consumer.py::ConsumerPlugin.run_pre_consume_script` |
| error_report-004 | user | —, —, — | `docs/configuration.md#PAPERLESS_OCR_IMAGE_DPI` |
| error_report-005 | engineer | 1, 1, 1 | `src/paperless/parsers/tesseract.py::RasterisedDocumentParser._handle_subprocess_output_error` |
| error_report-006 | user | 1, 1, 1 | `src/documents/barcodes.py::BarcodePlugin.detect` |
| error_report-007 | user | 1, 1, 1 | `src/documents/double_sided.py::CollatePlugin.run` |
| error_report-008 | user | 1, 1, 1 | `src/paperless_mail/mail.py::mailbox_login` |
| error_report-009 | user | 1, 1, 1 | `src/paperless_mail/mail.py::MailAccountHandler._handle_mail_rule` |
| error_report-010 | engineer | 1, 1, 1 | `src/documents/management/commands/document_importer.py::Command.load_metadata` |
| error_report-011 | user | 3, 3, 3 | `docs/troubleshooting.md#classifier-error-no-training-data-available` |
| error_report-012 | engineer | 2, 2, 2 | `src/documents/consumer.py::ConsumerPlugin.run` |
| error-001 | engineer | 1, 1, 1 | `src/paperless_ai/vector_store.py::_build_where` |
| error-002 | user | 1, 1, 1 | `src/documents/consumer.py::ConsumerPlugin.run` |
| error-003 | user | 1, 1, 1 | `src/paperless_mail/oauth.py::PaperlessMailOAuth2Manager.refresh_account_oauth_token` |
| error-004 | user | 1, 1, 1 | `src/paperless_ai/indexing.py::llm_index_add_or_update_document` |
| error-005 | engineer | 1, 1, 1 | `src/documents/serialisers.py::BulkEditSerializer._validate_parameters_split` |
| error-006 | user | 1, 1, 1 | `src/documents/consumer.py::ConsumerPlugin.run_post_consume_script` |
| error-007 | user | 1, 1, 1 | `src/documents/search/_schema.py::open_or_rebuild_index` |
| error-008 | user | 1, 1, 1 | `src/documents/filters.py::TitleContentFilter.filter` |
| error-009 | engineer | 1, 1, 1 | `src/paperless_mail/mail.py::MailAccountHandler._handle_mail_rule` |
| error-010 | user | 1, 1, 1 | `src/documents/consumer.py::ConsumerPlugin.run_post_consume_script` |
| error-011 | user | 1, 1, 1 | `src/documents/workflows/actions.py::execute_password_removal_action` |
| error-012 | user | 1, 1, 1 | `src/documents/tasks.py::cleanup_expired_share_link_bundles` |
| error-013 | engineer | 1, 1, 1 | `src/documents/validators.py::uri_validator` |
| error-014 | user | 1, 1, 1 | `src/documents/tasks.py::empty_trash` |
| error-015 | user | 1, 1, 1 | `src/documents/plugins/date_parsing/__init__.py::_discover_parser_class` |
| error-016 | engineer | 1, 1, 1 | `src/documents/serialisers.py::CustomFieldInstanceSerializer.validate` |
| error-017 | user | 1, 1, 1 | `src/documents/bulk_edit.py::remove_password` |
| error-018 | user | 1, 1, 1 | `src/documents/workflows/actions.py::execute_password_removal_action` |
| error-019 | engineer | 2, 2, 2 | `src/documents/management/commands/document_exporter.py::Command.generate_share_link_bundle_target` |
| error-020 | user | 1, 1, 1 | `src/documents/classifier.py::load_classifier` |
| error-021 | user | 1, 1, 1 | `src/documents/templating/filepath.py::validate_filepath_template_and_render` |
| error-022 | engineer | 1, 1, 1 | `src/paperless/validators.py::reject_dangerous_svg` |
| error-023 | user | 1, 1, 1 | `src/paperless/checks.py::check_v3_minimum_upgrade_version` |
| error-024 | user | 1, 1, 1 | `src/documents/views.py::BulkEditView.post` |
| error-025 | engineer | 1, 1, 1 | `src/paperless/settings/__init__.py:L1198-L1250` |
| error-026 | user | 1, 1, 1 | `src/paperless/parsers/registry.py::ParserRegistry.discover` |
| error-027 | user | 1, 1, 1 | `src/paperless/parsers/utils.py::get_pdf_first_page_size_points` |
| error-028 | engineer | 1, 1, 1 | `src/documents/serialisers.py::_validate_rotation_degrees` |
| error-029 | user | 1, 1, 1 | `src/paperless_mail/mail.py::MailAccountHandler._handle_mail_rule` |
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

## Промахи (3)

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

### error_report-004 — «кидаю картинку png, задача обработки падает. в ошибке что-то про DPI, мол в картинке нет информации о DPI и архивный pdf из неё сделать нельзя. что делать?»

Ожидалось: 

Найдено:
1. `docs/configuration.md#PAPERLESS_OCR_IMAGE_DPI` (0.521)
2. `src/paperless/parsers/tesseract.py::RasterisedDocumentParser.get_dpi` (0.507)
3. `docs/configuration.md#PAPERLESS_CONSUMER_BARCODE_DPI` (0.479)
4. `src/documents/parsers.py::rasterize_pdf_page_to_png` (0.476)
5. `src/paperless/parsers/tesseract.py::RasterisedDocumentParser.calculate_a4_dpi` (0.465)
