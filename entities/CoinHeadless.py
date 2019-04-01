from entities.EntityBase import EntityBase
from classes.Animation import Animation
import pygame
from copy import copy


class CoinHeadless(EntityBase):
    def __init__(self, x, y, gravity=0):
        super(CoinHeadless, self).__init__(x, y, gravity)
        self.type = "Item"

    def update(self):
        if(self.alive):
            pass
