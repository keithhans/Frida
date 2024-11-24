from options import Options
from painter import Painter
import datetime
from my_tensorboard import TensorBoard



opt = Options()
opt.gather_options()

# Setup Tensorboard
date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
opt.writer = TensorBoard('{}/{}'.format('tensorboard', run_name))
#opt.writer.add_text('args', str(sys.argv), 0)

opt.robot=None

painter = Painter(opt) 