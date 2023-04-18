from dm_control import composer
from dm_control import mjcf
import os


class Obstacle(composer.Entity):
    def _build(self, pos, name='sphere'):
        self._mjcf_root = mjcf.RootElement()
        self._mjcf_root.worldbody.add('geom',
                                      type="sphere",
                                      mass="1.2",
                                      contype="1",
                                      friction="0.4 0.005 0.00001",
                                      conaffinity="1",
                                      size="0.08",
                                      rgba=(0.8, 0.3, 0.3, 1),
                                      pos=pos)

    @property
    def mjcf_model(self):
        return self._mjcf_root
