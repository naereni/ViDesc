### Настройка окружения

Сначала создать и активировать venv:

```bash
python3 -m venv venv
. venv/bin/activate
```

Дальше поставить зависимости:

```bash
make install
```
### Команды

#### Подготовка
* `make install` - установка библиотек
* `make download_weights` - скачать веса моделек

#### Запуск сервиса
* `make run_app` - запустить сервис. Можно с аргументом `APP_PORT`

#### Сборка образа
* `make build` - собрать образ. Можно с аргументами `DOCKER_TAG`, `DOCKER_IMAGE`

#### Статический анализ
* `make lint` - запуск линтеров

#### Тестирование
* `make run_unit_tests` - запуск юнит-тестов
* `make run_integration_tests` - запуск интеграционных тестов
* `make run_all_tests` - запуск всех тестов
* `make generate_coverage_report` - сгенерировать coverage-отчет


### Ссылка на материалы
* [Презентация](https://docs.google.com/presentation/d/1L8e7dveCTyvu1tBL6xgf2uLFQuouJLypOnZqh8IBoVw/edit?usp=sharing)
* [GitLab CI/CD](https://docs.gitlab.com/ee/ci/)
* [Ansible](https://docs.ansible.com/)
