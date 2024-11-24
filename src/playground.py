from options import Options
from painter import Painter

opt = Options()
opt.gather_options()
opt.writer=None
opt.robot=None

painter = Painter(opt) 