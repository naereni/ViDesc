# ViDesc - Video Description Service
Проект по разработке сервиса для генерации описаний к видео на русском. Упор в проекте сделан на внутреннее качество продакшена. Будут использованы технологии GIT, FastAPI, Docker, MLOps, DeepLearning.

## Start develop
```bash
    # if poetry not installed
    curl -sSL https://install.python-poetry.org | python3 -
    # need python 3.9
    git clone git@github.com:naereni/ViDesc.git && cd ViDesc
    poetry install
    pre-commit install && pre-commit autoupdate
    pre-commit run --all-files
```

## Authors & Contributors
- [@naereni](https://github.com/naereni)
- [@mike-yasnov](https://github.com/mike-yasnov)
