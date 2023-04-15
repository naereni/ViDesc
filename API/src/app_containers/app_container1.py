from dependency_injector import containers, providers
from src.services.video_descriptor import VideoDescription
from src.services.videsc import ViDesc


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    video_describer = providers.Factory(
        VideoDescription,
        config=config.services.video_description,
    )

    videsc = providers.Singleton(
        ViDesc,
        descriptor=video_describer,
    )
