"""EmoNarrify: Emotional Image-to-Audio Story Generation"""


def __getattr__(name):
    if name == "EmoNarrifyPipeline":
        from emonarrify.pipeline import EmoNarrifyPipeline
        return EmoNarrifyPipeline
    raise AttributeError(f"module 'emonarrify' has no attribute {name!r}")
