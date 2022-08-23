"""
Calculate the distance of eyes and mouth of the face.
"""
X = 0
Y = 1


class EyeMouthDistance:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.eye_distance = self.get_eye_distance()
        self.mouth_distance = self.get_mouth_distance()

    def left_eye_distance(self):
        distance = (self.landmarks[61][Y] - self.landmarks[67][Y]) + \
                   (self.landmarks[62][Y] - self.landmarks[66][Y]) + \
                   (self.landmarks[63][Y] - self.landmarks[65][Y])

        return abs(distance / (3 * (self.landmarks[60][X] - self.landmarks[64][X])))

    def right_eye_distance(self):
        distance = (self.landmarks[69][Y] - self.landmarks[75][Y]) + \
                   (self.landmarks[70][Y] - self.landmarks[74][Y]) + \
                   (self.landmarks[71][Y] - self.landmarks[73][Y])

        return abs(distance / (3 * (self.landmarks[68][X] - self.landmarks[72][X])))

    def get_eye_distance(self):
        return (self.left_eye_distance() + self.right_eye_distance()) / 2

    def get_mouth_distance(self):
        distance = (self.landmarks[89][Y] - self.landmarks[95][Y]) + \
                   (self.landmarks[90][Y] - self.landmarks[94][Y]) + \
                   (self.landmarks[91][Y] - self.landmarks[93][Y])

        return abs(distance / (3 * (self.landmarks[88][X] - self.landmarks[92][X])))
