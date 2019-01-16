import pygame
from pygame.transform import flip


class goTraitHeadless():
    def __init__(self, ent):
        self.direction = 0
        self.heading = 1
        self.accelVel = 0.64
        self.decelVel = 0.15
        self.maxVel = 3.2
        self.boost = False
        self.entity = ent

    def update(self):
        if(self.boost):
            self.maxVel = 5.5
        else:
            if(abs(self.entity.vel.x) > 3.2):
                self.entity.vel.x = 3.2 * self.heading
            self.maxVel = 3.2

        if(self.direction != 0):
            self.heading = self.direction
            if(self.heading == 1):
                if(self.entity.vel.x < self.maxVel):
                    self.entity.vel.x += self.accelVel * self.heading
            else:
                if(self.entity.vel.x > -self.maxVel):
                    self.entity.vel.x += self.accelVel * self.heading
        else:
            if(self.entity.vel.x >= 0):
                self.entity.vel.x -= self.decelVel
            else:
                self.entity.vel.x += self.decelVel
            if(int(self.entity.vel.x) == 0):
                self.entity.vel.x = 0
