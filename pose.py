# Pose Class
import numpy as np

class Pose:
    def __init__(self, lower_body_parts):
        # lower_body_parts = [(0)RAnkle.value, (1) RKnee.value, (2)RHip.value, (3)LAnkle.value, (4)LKnee.value, (5)LHip.value, (6)Neck.value]
        self.lower_body_parts = lower_body_parts        

    def get_rhip_angle(self):
        # angle between the RKnee, RHip, Neck
        return joint_angle(self.lower_body_parts[1], self.lower_body_parts[2], self.lower_body_parts[6])
    def get_rknee_angle(self):
        # angle between RAnkle, RKnee, RHip
        return joint_angle(self.lower_body_parts[0], self.lower_body_parts[1], self.lower_body_parts[2])
    def get_lhip_angle(self):
        # angle between LKnee, LHip, Neck
        return joint_angle(self.lower_body_parts[4], self.lower_body_parts[5], self.lower_body_parts[6])
    def get_lknee_angle(self):
        # angle between LAnkle, LKnee, LHip
        return joint_angle(self.lower_body_parts[3], self.lower_body_parts[4], self.lower_body_parts[5])
    
    def plant_side(self):
        # which side of the body's ankle is closest to the ground
        if self.lower_body_parts[0] < self.lower_body_parts[3]:
            return "right"
        return "left"
    
    def get_PHip_angle(self):
        if self.plant_side == "right":
            return self.get_rhip_angle()
        return self.get_lhip_angle()

    def get_PKNee_angle(self):
        if self.plant_side == "right":
            return self.get_rknee_angle()
        return self.get_lknee_angle()

    def get_SHip_angle(self):
        if self.plant_side == "right":
            return self.get_lhip_angle()
        return self.get_rhip_angle()

    def get_SKnee_angle(self):
        if self.plant_side == "right":
            return self.get_lknee_angle()
        return self.get_rknee_angle()

def joint_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # normalization of vectors
    limb1 = a - b
    limb2 = c - b

    # calculate dot product between two lines
    dot_limbs = np.dot(limb1, limb2)

    # calculate cosine angle
    cos_angle = dot_limbs / (np.linalg.norm(limb1) * np.linalg.norm(limb2))

    # radians to degrees 
    angle = np.arccos(cos_angle)
    return angle


