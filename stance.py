# Stance Class
class Stance:
    def __init__(self,ideal_PHip_angle,ideal_PKnee_angle,ideal_SHip_angle,ideal_SKnee_angle): 
        # ideal plant side angles in degrees
        self.ideal_PHip_angle = ideal_PHip_angle
        self.ideal_PKnee_angle = ideal_PKnee_angle
        # ideal swing side angles in degrees
        self.ideal_SHip_angle = ideal_SHip_angle
        self.ideal_SKnee_angle = ideal_SKnee_angle
    
    def get_ideal_PHip_angle(self):
        return self.ideal_PHip_angle
    def get_ideal_PKnee_angle(self):
        return self.ideal_PKnee_angle    
    def get_ideal_SHip_angle(self):
        return self.ideal_SHip_angle
    def get_ideal_SKnee_angle(self):
        return self.ideal_SKnee_angle    
