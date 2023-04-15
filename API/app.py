import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf
from src.app_containers.app_container1 import AppContainer
from src.routes import predictor
from src.routes.routers import router as app_router


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load("config/config.yml")
    container.config.from_dict(cfg)
    container.wire([predictor])

    app = FastAPI()
    set_routers(app)
    return app


def set_routers(app: FastAPI):
    app.include_router(app_router, prefix="/videodescription", tags=["video"])


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, port=5000, host="0.0.0.0")
