from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from src.app_containers.app_container1 import AppContainer
from src.routes.routers import router
from src.services.utils import read_video
from src.services.videsc import ViDesc


@router.get("/predict")
@inject
def predict(
    video_path: str,
    service: ViDesc = Depends(Provide[AppContainer.videsc]),
):
    video = read_video(video_path)
    caption = service.predict(video)

    return caption
