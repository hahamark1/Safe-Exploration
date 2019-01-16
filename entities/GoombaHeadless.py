from classes.Sprites import Sprites
from classes.Animation import Animation
from classes.Maths import vec2D
from traits.leftrightwalk import LeftRightWalkTrait
from entities.EntityBase import EntityBase
import pygame


class GoombaHeadless(EntityBase):
    def __init__(self, x, y, level):
        super(GoombaHeadless, self).__init__(x, y - 1, 1.25)
        self.leftrightTrait = LeftRightWalkTrait(self, level)
        self.type = "Mob"
        self.alive = True

    def update(self):
        if(self.alive):
            self.applyGravity()
            self.leftrightTrait.update()
        else:
            if(self.timer < self.timeAfterDeath):
                pass
            else:
                self.alive = None
            self.timer += 0.1