from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.audio_beep.pipeline import AudioBeepPipeline

    register(AudioBeepPipeline)
