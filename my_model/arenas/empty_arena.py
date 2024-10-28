from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


class EmptyArena(Arena):
    """This is just a placeholder for trajectory optimization. It is an empty arena."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/empty_arena.xml"))
