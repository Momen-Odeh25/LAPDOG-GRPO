import datetime
import os
import signal
import socket
import subprocess
import sys
from logging import getLogger

import torch

logger = getLogger()

GLOO_GROUP = None


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))

    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = "SLURM_JOB_ID" in os.environ and not "WORLD_SIZE" in os.environ
    has_local_rank = hasattr(params, "local_rank")

    # SLURM job
    if params.is_slurm_job and has_local_rank:
        assert params.local_rank == -1  # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS",
            "SLURM_TASKS_PER_NODE",
            "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
            "SLURM_NODEID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_TASK_PID",
        ]

        for name in SLURM_VARIABLES:
            _ = os.environ.get(name, None)

        # number of nodes / node ID
        params.n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        params.node_id = int(os.environ["SLURM_NODEID"])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ["SLURM_LOCALID"])
        params.global_rank = int(os.environ["SLURM_PROCID"])

        # number of processes / GPUs per node
        params.world_size = int(os.environ["SLURM_NTASKS"])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        params.main_addr = hostnames.split()[0].decode("utf-8")
        assert 10001 <= params.main_port <= 20000 or params.world_size == 1

        # set environment variables for 'env://'
        os.environ["MASTER_ADDR"] = params.main_addr
        os.environ["MASTER_PORT"] = str(params.main_port)
        os.environ["WORLD_SIZE"] = str(params.world_size)
        os.environ["RANK"] = str(params.global_rank)

        params.is_distributed = True

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    # LOGIC FIX: Check environment variable if argument is default -1
    elif has_local_rank and (params.local_rank != -1 or "LOCAL_RANK" in os.environ):
        
        if "LOCAL_RANK" in os.environ:
            params.local_rank = int(os.environ["LOCAL_RANK"])

        assert params.main_port == -1 or True

        # read environment variables
        if "RANK" in os.environ:
            params.global_rank = int(os.environ["RANK"])
        if "WORLD_SIZE" in os.environ:
            params.world_size = int(os.environ["WORLD_SIZE"])
        
        if "NGPU" in os.environ:
            params.n_gpu_per_node = int(os.environ["NGPU"])
        else:
            params.n_gpu_per_node = torch.cuda.device_count()

        # number of nodes / node ID
        if params.n_gpu_per_node > 0:
            params.n_nodes = params.world_size // params.n_gpu_per_node
            params.node_id = params.global_rank // params.n_gpu_per_node
        else:
            params.n_nodes = 1
            params.node_id = 0
            
        params.is_distributed = True

    else:
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.is_distributed = False
        params.n_nodes = 1
        params.node_id = 0
        params.n_gpu_per_node = 1

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:
        # Fix for if gloo sockets are inconsistent or hang on detection
        if "GLOO_SOCKET_IFNAME" in os.environ:
            gloo_socket_ifname = os.environ["GLOO_SOCKET_IFNAME"]
        else:
            try:
                p1 = subprocess.Popen(["ip", "r"], stdout=subprocess.PIPE)
                p2 = subprocess.Popen(["grep", "default"], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                gloo_socket_ifname = subprocess.check_output(["awk", "{print $5}"], stdin=p2.stdout).decode("utf-8").strip()
                p2.stdout.close()
                os.environ["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname
            except Exception:
                # Fallback to loopback if detection fails/hangs
                os.environ["GLOO_SOCKET_IFNAME"] = "lo"

        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )

        global GLOO_GROUP

        GLOO_GROUP = torch.distributed.new_group(
            list(range(params.world_size)), backend="gloo", timeout=datetime.timedelta(0, 600)
        )


def get_gloo_group():
    global GLOO_GROUP
    assert GLOO_GROUP is not None
    return GLOO_GROUP