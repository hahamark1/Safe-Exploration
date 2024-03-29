from classes.Maths import vec2D


class Camera():
    def __init__(self, pos, entity):
        self.pos = vec2D(pos.x, pos.y)
        self.entity = entity
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32

    def move(self):
        xPosFloat = self.entity.getPosIndex(True).x
        if xPosFloat > 10 and xPosFloat < 140:
            self.pos.x = -xPosFloat + 10
        self.x = self.pos.x * 32
        self.y = self.pos.y * 32
