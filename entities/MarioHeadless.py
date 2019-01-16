from classes.Sprites import Sprites
import pygame
from pygame.locals import *
import classes.Maths
import numpy as np
import random
from traits import go, jump
from traits.go import goTrait
from traits.goHeadless import goTraitHeadless
from traits.jump import jumpTrait
from classes.Animation import Animation
from classes.Collider import Collider
from classes.Camera import Camera
from entities.EntityBase import EntityBase
from classes.EntityCollider import EntityCollider
from traits.bounce import bounceTrait
from classes.Sound import Sound
from classes.Input import Input


class MarioHeadless(EntityBase):
    def __init__(self, x, y, level, gravity=1.25):
        super(MarioHeadless, self).__init__(x, y, gravity)
        self.input = Input(self)

        self.traits = {
            "jumpTrait": jumpTrait(self),
            "goTrait": goTraitHeadless(self),
            "bounceTrait": bounceTrait(self)
        }

        self.levelObj = level
        self.collision = Collider(self, level)
        self.EntityCollider = EntityCollider(self)
        self.restart = False

    def update(self):
        self.updateTraits()
        self.moveMario()
        self.applyGravity()
        self.checkEntityCollision()

    def moveMario(self):
        self.rect.y += self.vel.y
        self.collision.checkY()
        self.rect.x += self.vel.x
        self.collision.checkX()

    def checkEntityCollision(self):
        for ent in self.levelObj.entityList:
            collisionState = self.EntityCollider.check(ent)
            if collisionState.isColliding:
                if (ent.type == "Item"):
                    self._onCollisionWithItem(ent)
                elif (ent.type == "Block"):
                    self._onCollisionWithBlock(ent)
                elif (ent.type == "Mob"):
                    self._onCollisionWithMob(ent, collisionState)

    def _onCollisionWithItem(self, item):
        self.levelObj.entityList.remove(item)
        self.levelObj.points += 100
        self.levelObj.coins += 1

    def _onCollisionWithBlock(self, block):
        block.triggered = True

    def _onCollisionWithMob(self, mob, collisionState):
        if collisionState.isTop and (mob.alive is True or mob.alive == "shellBouncing"):
            self.rect.bottom = mob.rect.top
            self.bounce()
            self.killEntity(mob)
        elif collisionState.isTop and mob.alive == "sleeping":
            self.rect.bottom = mob.rect.top
            mob.timer = 0
            self.bounce()
            mob.alive = False
        elif collisionState.isTop and mob.alive == "sleeping":
            if (mob.rect.x < self.rect.x):
                mob.leftrightTrait.direction = -1
            else:
                mob.leftrightTrait.direction = 1
            mob.alive = "shellBouncing"
        elif collisionState.isColliding and mob.alive == True:
            self.gameOver()

    def bounce(self):
        self.traits['bounceTrait'].jump = True

    def killEntity(self, ent):
        if ent.__class__.__name__ != "Koopa":
            ent.alive = False
        else:
            ent.timer = 0
            ent.alive = "sleeping"
        self.levelObj.points += -1000

    def gameOver(self):
        self.restart = True

    def getPos(self):
        return (self.rect.x, self.rect.y)

    def doRandomMove(self):
        moves = ['moveLeft', 'moveRight', 'jump', 'doNothing']
        random_move = random.choice(moves)
        if random_move == 'moveLeft':
            self.traits['goTrait'].direction = -1
        elif random_move == 'moveRight':
            self.traits['goTrait'].direction = 1
        elif random_move == 'jump':
            self.traits['jumpTrait'].start()
        elif random_move == 'doNothing':
            self.traits['goTrait'].direction = 0