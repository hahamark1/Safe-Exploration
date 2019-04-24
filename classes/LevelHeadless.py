from classes.Sprites import Sprites
import pygame
import json
from entities.Goomba import Goomba
from entities.Koopa import Koopa
from entities.GoombaHeadless import GoombaHeadless
from entities.RandomBox import RandomBox
from classes.Tile import Tile
from entities.CoinHeadless import CoinHeadless
import numpy as np
from copy import copy


class LevelHeadless():
    def __init__(self, levelname):
        self.level = None
        self.levelLength = 0
        self.points = 0
        self.coins = 0
        self.clock = 0
        self.entityList = []
        self.groundList = []
        self.loadLevel(levelname)

    def loadLevel(self, levelname):
        with open("./levels/{}".format(levelname)) as jsonData:
            data = json.load(jsonData)
            self.loadObjects(data)
            self.loadEntities(data)
            self.loadLayers(data)
            self.levelLength = data['length']

    def loadEntities(self, data):
        if 'entities' in data['level']:
            for entity in data['level']['entities']:
                for position in entity['positions']:
                    if entity['name'] == "Goomba":
                        self.addGoomba(position[0], position[1])
                    elif entity['name'] == "Koopa":
                        self.addKoopa(position[0], position[1])
                    elif entity['name'] == "coin":
                        self.addCoin(position[0], position[1])
                        # pass
                    elif entity['name'] == "randomBox":
                        self.addRandomBox(position[0], position[1])



    def loadObjects(self, data):
        for obj in data['level']['objects']:
            for position in obj['positions']:
                if(obj['name'] == "randomBox"):
                    self.addRandomBox(position[0], position[1])
                elif(obj['name'] == "pipe"):
                    self.addPipeSprite(position[0], position[1], position[2])
                elif(obj['name'] == "coin"):
                    self.addCoin(position[0], position[1])
                elif(obj['name'] == "ground"):
                    self.groundList.append([position[0], position[1]])


    def loadLayers(self, data):
        levely = {}
        for layer in data['level']['layers']:
            for y in range(layer['ranges']['y'][0], layer['ranges']['y'][1]):
                if y not in levely:
                    levely[y] = {}

                for x in range(layer['ranges']['x'][0],
                               layer['ranges']['x'][1]):
                    if(layer['spritename'] == 'sky'):
                        levely[y][x]=Tile(None,
                                None)
                    elif(layer['spritename'] == 'ground'):
                        levely[y][x]=Tile(None, pygame.Rect(
                                    x * 32, (y - 1) * 32, 32, 32))
                        self.groundList.append([x,y])
                    else:
                        levely[y][x]=Tile(None, pygame.Rect(
                            x * 32, (y - 1) * 32, 32, 32))
        # print(levely)
        level_setup = [[levely[key2][key1] for key1 in sorted(levely[key2])] for key2 in sorted(levely)]

        self.level = level_setup



    def updateEntities(self):
        for entity in self.entityList:
            entity.update()
            if(entity.alive is None):
                self.entityList.remove(entity)


    def addRandomBox(self, x, y):
        self.level[y][x] = Tile(
            None,
            pygame.Rect(x * 32, y * 32 - 1, 32, 32)
        )
        self.entityList.append(
            RandomBox(
                x,
                y,)
        )

    def addCoin(self, x, y):
        self.entityList.append(
            CoinHeadless(x, y)
        )

    def addGoomba(self, x, y):
        self.entityList.append(
            GoombaHeadless(x, y, self)
        )

    def addKoopa(self, x, y):
        self.entityList.append(
            Koopa(x, y, self)
        )
