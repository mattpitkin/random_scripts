jobs = []

# get job names from DAG file
dagfile = "mydag.dag"  # examples dag name
with open(dagfile, "r") as fp:
    for line in fp.readlines():
        if "JOB" == line[0:3]:
            jobs.append(line.split()[1])

# get log data (split on ellipsis)
with open("{}.nodes.log".format(dagfile), "r") as fp:
    logdata = fp.read().split("...")

# get cluster ids associated with each job
nodeid = {}
for log in logdata:
    # cluster
    l = log.split()

    if len(l) < 1:
        continue

    cluster = l[1].strip("(").strip(")")  # cluster id    

    if "DAG Node" in log:
        nodeid[cluster] = {}
        dagjobid = l[-1]
        nodeid[cluster]["job"] = dagjobid


# get jobs that exited with "Normal termination (return value 0)"
success = []
for log in logdata:
    l = log.split()

    if len(l) < 1:
        continue

    cluster = l[1].strip("(").strip(")")  # cluster id
    if "Normal termination (return value 0)" in log:
        success.append(cluster)


# get list of DONE jobs and list of not done jobs
success = set(success)  # convert to set for quicker searching
done = []
todo = []
for job in jobs:
    isdone = False
    for cluster in nodeid:
        if nodeid[cluster]["job"] == job:
            if cluster in success:
                done.append(job)
                isdone = True
            break

    if not isdone:
        todo.append(job)

# create rescue dag
rescuefile = "{}.rescue001".format(dagfile)

with open(rescuefile, "w") as fp:
    fp.write("# Total number of Nodes: {}\n".format(len(jobs)))
    fp.write("# Nodes premarked DONE: {}\n".format(len(done)))
    fp.write("# Nodes that failed: {}\n\n".format(len(nodeid) - len(done)))

    for donejob in done:
        fp.write("DONE {}\n".format(donejob))
