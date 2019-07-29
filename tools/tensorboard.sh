#!/bin/bash

srun -t 1:00:00 --qos=cbmm ss tensorboard --port=6058 \
--logdir=/om/user/scasper/workspace/models/init_scheme_tests/

#srun -t 2:00:00 --qos=cbmm /
#singularity exec -B /om:/om --nv /om/user/scasper/singularity/xboix-tensorflow.simg tensorboard \
#--port=6058 --logdir=/om/user/scasper/redundancy_workspace/models/init_scheme_tests/

# You don't really need this. Just use:
# tb /om/user/scasper/redundancy_workspace/models/replication/
# ssh -NL 16006:polestar:6099 scasper@polestar.mit.edu
# http://127.0.0.1:16006

# Similarly, for a jupyter nb:
# first go to redundancy dir
# jn
# ssh -N -f -L 1229:polestar:9000 scasper@polestar.mit.edu
# https://127.0.0.1:1229
