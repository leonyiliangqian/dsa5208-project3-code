# Two-"node" local simulation: the dev image plus sshd, so Open MPI's rsh
# launcher can start ranks in a second container. Throwaway test containers
# on a private Docker network with a generated key; this tests mapping,
# binding and process cleanup across hosts, not PBS (no tm here). ssh
# forwards PBS_JOBID so remote daemons carry it, as the cleanup check assumes
# PBS tm does on Vanda (node_report.sh records whether it really does).
FROM dsa5208-p3-dev:bookworm
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server openssh-client procps \
 && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd \
 && echo 'AcceptEnv PBS_JOBID' >> /etc/ssh/sshd_config
USER dev
RUN mkdir -p /home/dev/.ssh && ssh-keygen -q -t ed25519 -N "" -f /home/dev/.ssh/id_ed25519 \
 && cp /home/dev/.ssh/id_ed25519.pub /home/dev/.ssh/authorized_keys \
 && printf 'Host node*\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n  SendEnv PBS_JOBID\n' > /home/dev/.ssh/config \
 && chmod 700 /home/dev/.ssh && chmod 600 /home/dev/.ssh/*
USER root
CMD ["/usr/sbin/sshd", "-D", "-e"]
